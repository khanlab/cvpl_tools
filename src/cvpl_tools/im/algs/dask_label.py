"""
dask-image's label function encounters memory error when in large dataset. This file defines a distributed, on-disk
version of the label() function of scipy.ndimage
"""
import dask
import dask.array as da
import numcodecs
import numpy as np
import numpy.typing as npt
import numcodecs.abc

from cvpl_tools.im.ndblock import NDBlock
from cvpl_tools.tools.dask_utils import get_dask_client
from cvpl_tools.fsspec import RDirFileSystem, ensure_rdir_filesystem
import cvpl_tools.tools.fs as tlfs
from scipy.ndimage import label as scipy_label
import cvpl_tools.im.algorithms as cvpl_algorithms
from dask.distributed import print as dprint
from collections import defaultdict


def find_connected_components(edges: set[tuple[int, int]]) -> list[set[int, ...], ...]:
    graph = defaultdict(set)

    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    visited = set()
    components = []

    def dfs(node, component):
        # avoid recursive implementation, which fails if maximum recursion limits is exceeded on large datasets
        nodes = {node}
        while len(nodes) > 0:
            node = nodes.pop()
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    nodes.add(neighbor)

    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)

    return components


async def compute_lower_adj_set(locally_labeled: da.Array, compressor: numcodecs.abc.Codec) -> set[tuple[int, int]]:
    """From a SQL db of neighboring slices, compute their corresponding lower adjacency matrix as a set

    Args:
        locally_labeled: Path to the SQL database on disk
        compressor: The compressor used to compress chunks during persist() call

    Returns:
        The adjacency edge set
    """
    numblocks = locally_labeled.numblocks

    def find_neib(block, block_info):
        if block_info is None:
            return np.zeros(block.shape, dtype=np.int64)
        block_index = tuple(block_info[0]['chunk-location'])

        neibs = set()
        for ax in range(block.ndim):
            if block_index[ax] == 0:
                continue
            # flatten two flat surfaces into two 1d arrays
            sli = np.moveaxis(np.take(block, indices=(0, 1), axis=ax), ax, 0).reshape(2, -1)
            tups = cvpl_algorithms.np_unique(sli, axis=1)
            assert tups.shape[0] == 2, (
                f'Expected tups 2 rows, got tups.shape={tups.shape}, at block_index={block_index}, '
                f'over ax={ax}. \nsli.shape={sli.shape}, '
                f'block.shape={block.shape}, numblocks={numblocks}. \nsli={sli}')
            for row in tups.transpose().tolist():
                i1, i2 = row
                if i2 < i1:
                    tmp = i2
                    i2 = i1
                    i1 = tmp
                if i1 == 0:
                    continue
                assert i1 < i2, f'i1={i1} and i2={i2}!'  # can not be equal because indices are globally unique here
                tup = (i2, i1)
                neibs.add(tup)
        if len(neibs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        neibs = np.array(tuple(neibs), dtype=np.int64)
        assert neibs.ndim == 2 and neibs.shape[1] == 2, neibs.shape
        return neibs

    depth = {dim: (1, 0) for dim in range(locally_labeled.ndim)}
    padded = locally_labeled.map_overlap(func=lambda x: x, depth=depth, trim=False).persist(compressor=compressor)
    result = NDBlock(padded)
    result = await NDBlock.map_ndblocks((result,), find_neib, out_dtype=np.int64)
    neibs = await result.reduce(force_numpy=True)

    lower_adj = set()
    for pair in neibs:
        lower_adj.add(tuple(pair))
    return lower_adj


async def label(im: npt.NDArray | da.Array | NDBlock,
                output_dtype: np.dtype = None,
                context_args: dict = None
                ) -> npt.NDArray | da.Array | NDBlock:
    """Dask array version of scipy.ndimage.label

    Args:
        im: The image to be labeled
        output_dtype: Output label data type; choose one you will be sure to accommodate the max number of contours
            found in the image
        context_args: extra contextual arguments
            - cache_url (str | RDirFileSystem): Points to directory under which cache will be stored
            - logging (bool, optional): If provided and True, print some debugging info to coiled logger

    Returns:
        Tuple (lbl_im, nlbl) where lbl_im is a globally labeled image of the same type/chunk size as the input
    """
    if isinstance(im, np.ndarray):
        return scipy_label(im, output=output_dtype)

    if context_args is None:
        context_args = {}
    cache_url = context_args.get('cache_url')
    is_logging = context_args.get('logging', False)

    if cache_url is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(cache_url)

    ndim = im.ndim
    compressor = numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
    vargs = dict(compressor=compressor)  # this is for compressing labels of uint8 or int32 types

    is_dask = isinstance(im, da.Array)
    if not is_dask:
        assert isinstance(im, NDBlock)
        im = im.as_dask_array(storage_options=vargs)

    def map_block(block: npt.NDArray, block_info: dict):
        lbl_im = scipy_label(block, output=output_dtype)[0]
        return lbl_im

    def to_max(block: npt.NDArray, block_info: dict):
        return block.max(keepdims=True)

    # compute locally labelled chunks and save their bordering slices
    if is_logging:
        print('Locally label the image')

    locally_labeled = await tlfs.cache_im(
        lambda: im.map_blocks(map_block, meta=np.zeros(tuple(), dtype=output_dtype)),
        context_args=dict(cache_url=None if fs is None else fs['locally_labeled_without_cumsum'], viewer_args=vargs)
    )

    async def compute_nlbl_np_arr():
        if is_logging:
            print('Taking the max of each chunk to obtain number of labels')
        locally_labeled_ndblock = NDBlock(locally_labeled)
        new_slices = list(tuple(slice(0, 1) for _ in range(ndim))
                          for _ in NDBlock(locally_labeled_ndblock).get_slices_list())
        nlbl_ndblock_arr = await NDBlock.map_ndblocks([locally_labeled_ndblock], fn=to_max, out_dtype=output_dtype,
                                                      new_slices=new_slices)
        if is_logging:
            print('Convert number of labels of chunks to numpy array')
        nlbl_np_arr = await nlbl_ndblock_arr.as_numpy()
        return nlbl_np_arr

    nlbl_np_arr = await tlfs.cache_im(fn=compute_nlbl_np_arr, context_args=dict(
        cache_url=None if fs is None else fs['nlbl_np_arr']))

    def compute_cumsum_np_arr():
        if is_logging:
            print('Compute prefix sum and reshape back')
        cumsum_np_arr = np.cumsum(nlbl_np_arr)
        return cumsum_np_arr

    cumsum_np_arr = await tlfs.cache_im(fn=compute_cumsum_np_arr, context_args=dict(
        cache_url=None if fs is None else fs['cumsum_np_arr']
    ))
    assert cumsum_np_arr.ndim == 1
    total_nlbl = cumsum_np_arr[-1].item()
    cumsum_np_arr[1:] = cumsum_np_arr[:-1]
    cumsum_np_arr[0] = 0
    cumsum_np_arr = cumsum_np_arr.reshape(nlbl_np_arr.shape)
    if is_logging:
        print(f'total_nlbl={total_nlbl}, Convert prefix sum to a dask array then to NDBlock')
    cumsum_da_arr = da.from_array(cumsum_np_arr, chunks=(1,) * cumsum_np_arr.ndim)

    def compute_locally_labeled():
        if is_logging:
            print(f'Computing locally labeled')

        def compute_slices(block: npt.NDArray, block2: npt.NDArray, block_info: dict = None):
            # block is the local label, block2 is the single element prefix summed number of labels

            block = block + (block != 0).astype(block.dtype) * block2
            return block

        nonlocal locally_labeled
        locally_labeled = da.map_blocks(compute_slices, locally_labeled, cumsum_da_arr,
                                        meta=np.zeros(tuple(), dtype=output_dtype))
        return locally_labeled

    locally_labeled = await tlfs.cache_im(
        compute_locally_labeled,
        context_args=dict(cache_url=None if fs is None else fs['locally_labeled_with_cumsum'], viewer_args=vargs)
    )

    comp_i = 0

    async def compute_globally_labeled():
        if is_logging:
            print('Process locally to obtain a lower triangular adjacency matrix')
        lower_adj = await compute_lower_adj_set(locally_labeled, compressor=compressor)
        print(f'The set of adjacency edges is of size {len(lower_adj)}')
        connected_components = find_connected_components(lower_adj)
        if is_logging:
            print('Compute final indices remap array')
        ind_map_np = np.arange(total_nlbl + 1, dtype=output_dtype)
        assigned_mask = np.zeros((total_nlbl + 1), dtype=np.uint8)
        assigned_mask[0] = 1  # we don't touch background class
        nonlocal comp_i
        while comp_i < len(connected_components):
            comp = connected_components[comp_i]
            comp_i += 1
            for j in comp:
                ind_map_np[j] = comp_i
                assigned_mask[j] = 1
        for i in range(assigned_mask.shape[0]):
            if assigned_mask[i] == 0:
                comp_i += 1
                ind_map_np[i] = comp_i

        if is_logging:
            print(f'comp_i={comp_i}, Remapping the indices array to be globally consistent')
        ind_map_scatter = await get_dask_client().scatter(ind_map_np, broadcast=True)

        def local_to_global(block, block_info, ind_map_scatter):
            return ind_map_scatter[block]

        return locally_labeled.map_blocks(func=local_to_global, meta=np.zeros(tuple(), dtype=output_dtype),
                                          ind_map_scatter=ind_map_scatter)

    comp_i = (await tlfs.cache_im(lambda: np.array(comp_i, dtype=np.int64)[None], context_args=dict(
        cache_url=None if fs is None else fs['comp_i']
    ))).item()

    globally_labeled = await tlfs.cache_im(
        fn=compute_globally_labeled,
        context_args=dict(
            cache_url=None if fs is None else fs['globally_labeled'],
            viewer_args=vargs
        )
    )
    result_arr = globally_labeled
    if not is_dask:
        if is_logging:
            print('converting the result to NDBlock')
        result_arr = NDBlock(result_arr)
    if is_logging:
        print('Function ends')

    im = await tlfs.cache_im(lambda: result_arr, context_args=dict(
        cache_url=None if fs is None else fs['global_os'],
        viewer_args=vargs | dict(is_label=True)
    ))

    return result_arr, comp_i
