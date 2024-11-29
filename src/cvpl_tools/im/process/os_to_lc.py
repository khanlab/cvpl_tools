from typing import Sequence

from cvpl_tools.tools.dask_utils import compute, get_dask_client
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.im.algorithms as algorithms
from cvpl_tools.im.process.base import lc_interpretable_napari
import cvpl_tools.tools.fs as tlfs
from cvpl_tools.fsspec import ensure_rdir_filesystem
import dask.array as da
from scipy.ndimage import find_objects
import dask
from dask.distributed import print as dprint


def os_to_lc_direct_np_features(block: npt.NDArray[np.int32], full_statistics_map, min_size, block_info=None
) -> npt.NDArray[np.float64]:
    if block_info is not None:
        mat_width = block.ndim + len(full_statistics_map)
        idx_max = np.array(tuple(d - 1 for d in block.shape), dtype=np.int64)

        slices = block_info[0]['array-location']
        _contours_argwhere, _ids = algorithms.npindices_from_os(block, is_sparse=True)
        contours_argwhere, ids = [], []
        for i in range(len(_contours_argwhere)):
            contour = _contours_argwhere[i]
            assert contour.shape[1] == block.ndim, (f'Contour shape {contour.shape} returned by npindices_from_os '
                                                    f'does not match block.ndim={block.ndim}')
            if len(contour) > min_size:
                contours_argwhere.append(contour)
                ids.append(_ids[i])
        ids = np.array(ids, dtype=_ids.dtype)
        lc = [contour.astype(np.float64).mean(axis=0) for contour in contours_argwhere]

        is_empty = len(lc) == 0
        if is_empty:
            lc = np.zeros((0, mat_width), dtype=np.float64)
        else:
            tmp = lc
            lc = np.zeros((len(lc), mat_width), dtype=np.float64)
            lc[:, :block.ndim] = tmp
        if slices is not None and not is_empty:
            start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float64)
            assert not np.isnan(start_pos).any(), 'nan should not be present in slice() objects for this!'
            lc[:, :block.ndim] += start_pos[None, :]

        # append extra statistics columns
        for name, j in full_statistics_map.items():
            if name == 'nvoxel':
                col = [contour.shape[0] for contour in contours_argwhere]
            elif name == 'edge_contact':
                col = []
                for contour in contours_argwhere:
                    on_edge = (contour == 0).astype(np.uint8) + (contour == idx_max[None, :]).astype(np.uint8)
                    col.append(on_edge.sum().item() > 0)
            elif name == 'id':
                col = ids
            else:
                raise ValueError(f'Unrecognized name at index {j}: {name}')
            lc[:, block.ndim + j] = col

        return lc
    else:
        return np.zeros(block.shape, dtype=np.float64)


async def os_to_lc_direct(os,
                          min_size: int = 0,
                          reduce: bool = False,
                          is_global: bool = False,
                          ex_statistics: Sequence[str] = tuple(),
                          context_args: dict = None
                          ):
    if context_args is None:
        context_args = {}
    curl = context_args.get('cache_url')
    viewer_args = context_args.get('viewer_args', {})
    if curl is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(curl)
    viewer = viewer_args.get('viewer', None)

    ex_statistics = ex_statistics
    full_statistics_map = dict(
        nvoxel=0,
        edge_contact=1,
        id=2
    )  # these columns are appended to feature array; ex_statistics selects from these to return
    global_full_statistics_map = dict(
        nvoxel=0,
        id=1
    )
    ret_cols = tuple(full_statistics_map[name] for name in ex_statistics)

    _ndblock: NDBlock = None
    _reduced_features = None
    _reduced_np_features = None

    async def reduced_features():
        nonlocal _reduced_features
        if _reduced_features is None:
            _reduced_features = await _ndblock.reduce(force_numpy=False)
        return _reduced_features

    async def reduced_np_features():
        nonlocal _reduced_np_features
        if _reduced_np_features is None:
            rf = await reduced_features()
            if isinstance(rf, da.Array):
                rf = await compute(get_dask_client(), rf)
            _reduced_np_features = rf
        return _reduced_np_features

    async def feature_forward(im: npt.NDArray[np.int32] | da.Array) -> NDBlock[np.float64]:
        return await NDBlock.map_ndblocks([NDBlock(im)], os_to_lc_direct_np_features, out_dtype=np.float64,
                                          fn_args=dict(full_statistics_map=full_statistics_map, min_size=min_size))

    async def aggregate_by_id():
        """Aggregate _ndblock by id"""

        nonlocal _ndblock, full_statistics_map, global_full_statistics_map
        ref_ndblock: NDBlock = _ndblock
        ndim = ref_ndblock.get_ndim()
        block_indices = ref_ndblock.get_block_indices()
        slices_list = ref_ndblock.get_slices_list()
        chunk_shape = np.array(tuple(s.stop - s.start for s in slices_list[0]), dtype=np.float64)
        recons = {ind: [] for ind in block_indices}

        rf = await reduced_np_features()
        rf = rf[np.argsort(rf[:, -1])]
        if rf.shape[0] == 0:
            cnt_ranges = []
        else:
            cnt_ranges = list(find_objects(rf[:, -1].astype(np.int32)))

        nvoxel_ind = full_statistics_map['nvoxel']
        for i, rg in enumerate(cnt_ranges):
            if rg is None:
                continue
            lbl = i + 1
            subrf = rf[rg]
            nvoxel = subrf[:, ndim + nvoxel_ind]
            nvoxel_tot = nvoxel.sum()
            centroid = (subrf[:, :ndim] * nvoxel[:, None]).sum(axis=0) / nvoxel_tot

            row = centroid.tolist()
            row.append(nvoxel_tot)
            row.append(lbl)
            row = np.array(row, dtype=np.float64)
            ind = tuple(np.floor(centroid / chunk_shape).astype(np.int32).tolist())
            recons[ind].append(row)

        for i in range(len(slices_list)):
            ind = block_indices[i]
            rows = recons[ind]
            if len(rows) == 0:
                rows = np.zeros((0, ndim + len(global_full_statistics_map)), dtype=np.float64)
            else:
                rows = np.array(recons[ind], dtype=np.float64)
            recons[ind] = (rows, slices_list[i])
        _ndblock = NDBlock.create_from_dict_and_properties(recons, ref_ndblock.get_properties() | dict(is_numpy=True))

    async def ndblock_compute():
        nonlocal _ndblock, _reduced_features, _reduced_np_features
        _ndblock = await tlfs.cache_im(fn=feature_forward(os), context_args=dict(
            cache_url=None if fs is None else fs['block_level_lc_ndblock']
        ))
        if is_global:
            # update and aggregate the rows in ndblock that correspond to the same contour
            await aggregate_by_id()
            _reduced_features = None
            _reduced_np_features = None
        return _ndblock

    _ndblock = await tlfs.cache_im(fn=ndblock_compute, context_args=dict(
        cache_url=None if fs is None else fs['lc_ndblock']
    ))

    if viewer and viewer_args.get('display_points', True):
        if is_global:
            extras = list(global_full_statistics_map.keys())
        else:
            extras = list(full_statistics_map.keys())
        lc_interpretable_napari(
            'os_to_lc_centroids',
            await reduced_np_features(),
            viewer,
            os.ndim,
            extras
        )

    ret_cols = tuple(range(os.ndim)) + tuple(r + os.ndim for r in ret_cols)
    if reduce:
        ret = (await reduced_features())[:, ret_cols]
    else:
        ret = _ndblock.select_columns(ret_cols)
    _ndblock = None
    _reduced_features = None
    _reduced_np_features = None

    return ret
