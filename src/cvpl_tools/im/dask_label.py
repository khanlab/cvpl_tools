"""
dask-image's label function encounters memory error when in large dataset. This file defines a distributed, on-disk
version of the label() function of scipy.ndimage
"""

import dask.array as da
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
from cvpl_tools.im.partd_server import SQLiteKVStore, SQLitePartd, SqliteServer
from cvpl_tools.im.fs import CacheDirectory
import os
from scipy.ndimage import label as scipy_label
import partd
import pickle
import cvpl_tools.im.algorithms as cvpl_algorithms
from dask.distributed import print as dprint
from collections import defaultdict


class PairKVStore(SQLiteKVStore):
    def init_db(self):
        if not self.is_exists:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS kv_store (
                id TEXT PRIMARY KEY,
                value1 TEXT,
                value2 TEXT
            )
            ''')
            self.write_row_stmt = """
            INSERT INTO kv_store (id, value1, value2) VALUES (?, ?, NULL)
            ON CONFLICT(id) DO UPDATE SET value2=excluded.value1;
            """

    def read_all(self):
        self.cursor.execute("""
        SELECT id FROM kv_store
        """)
        ids = self.cursor.fetchall()
        for row in ids:
            self.cursor.execute("""
            SELECT value1, value2 FROM kv_store WHERE id = ?
            """, row)
            yield self.cursor.fetchone()


def find_connected_components(edges: set[tuple[int, int]]) -> list[set[int, ...], ...]:
    graph = defaultdict(set)

    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)

    return components


def label(im: npt.NDArray | da.Array | NDBlock, cache_dir: CacheDirectory, output_dtype: np.dtype = None,
          viewer_args: dict = None
          ) -> npt.NDArray | da.Array | NDBlock:
    """Return (lbl_im, nlbl) where lbl_im is a globally labeled image of the same type/chunk size as the input"""

    ndim = im.ndim
    if viewer_args is None:
        viewer_args = {}
    is_logging = viewer_args.get('logging', False)

    if isinstance(im, np.ndarray):
        return scipy_label(im, output=output_dtype)
    is_dask = isinstance(im, da.Array)
    if not is_dask:
        assert isinstance(im, NDBlock)
        im = im.as_dask_array(tmp_dirpath=cache_dir.path)

    def map_block(block: npt.NDArray, block_info: dict):
        lbl_im = scipy_label(block, output=output_dtype)[0]
        return lbl_im

    def to_max(block: npt.NDArray, block_info: dict):
        return block.max(keepdims=True)

    # compute locally labelled chunks and save their bordering slices
    if is_logging:
        print('Locally label the image')
    locally_labeled = cache_dir.cache_im(
        lambda: im.map_blocks(map_block, meta=np.zeros(tuple(), dtype=output_dtype)),
        cid='locally_labeled_without_cumsum'
    )

    def compute_nlbl_np_arr():
        if is_logging:
            print('Taking the max of each chunk to obtain number of labels')
        locally_labeled_ndblock = NDBlock(locally_labeled)
        new_slices = list(tuple(slice(0, 1) for _ in range(ndim))
                          for _ in NDBlock(locally_labeled_ndblock).get_slices_list())
        nlbl_ndblock_arr = NDBlock.map_ndblocks([locally_labeled_ndblock], fn=to_max, out_dtype=output_dtype,
                                                new_slices=new_slices)
        if is_logging:
            print('Convert number of labels of chunks to numpy array')
        nlbl_np_arr = nlbl_ndblock_arr.as_numpy()
        return nlbl_np_arr

    nlbl_np_arr = cache_dir.cache_im(fn=compute_nlbl_np_arr, cid='nlbl_np_arr')

    def compute_cumsum_np_arr():
        if is_logging:
            print('Compute prefix sum and reshape back')
        cumsum_np_arr = np.cumsum(nlbl_np_arr)
        return cumsum_np_arr

    cumsum_np_arr = cache_dir.cache_im(fn=compute_cumsum_np_arr, cid='cumsum_np_arr')
    assert cumsum_np_arr.ndim == 1
    total_nlbl = cumsum_np_arr[-1].item()
    cumsum_np_arr[1:] = cumsum_np_arr[:-1]
    cumsum_np_arr[0] = 0
    cumsum_np_arr = cumsum_np_arr.reshape(nlbl_np_arr.shape)
    if is_logging:
        print(f'total_nlbl={total_nlbl}, Convert prefix sum to a dask array then to NDBlock')
    cumsum_da_arr = da.from_array(cumsum_np_arr, chunks=(1,) * cumsum_np_arr.ndim)

    # Prepare cache file to be used
    if is_logging:
        print('Setting up cache sqlite database')
    _, cache_file = cache_dir.cache(cid='border_slices')
    os.makedirs(cache_file.path, exist_ok=True)
    db_path = f'{cache_file.path}/border_slices.db'

    def create_kv_store():
        kv_store = PairKVStore(db_path)
        return kv_store

    def get_sqlite_partd():
        partd = SQLitePartd(cache_file.path, create_kv_store=create_kv_store)
        return partd

    if is_logging:
        print('Setting up partd server')
    server = SqliteServer(cache_file.path, get_sqlite_partd=get_sqlite_partd)
    server_address = server.address

    # compute edge slices
    if is_logging:
        print('Computing edge slices, writing to database')

    def compute_slices(block: npt.NDArray, block2: npt.NDArray, block_info: dict = None):
        # block is the local label, block2 is the single element prefix summed number of labels

        client = partd.Client(server_address)
        block_index = list(block_info[0]['chunk-location'])
        block = block + (block != 0).astype(block.dtype) * block2
        for ax in range(block.ndim):
            for face in range(2):
                block_index[ax] += face
                indstr = '_'.join(str(index) for index in block_index) + f'_{ax}'
                sli_idx = face * (block.shape[ax] - 1)
                sli = np.take(block, indices=sli_idx, axis=ax)
                client.append({
                    indstr: pickle.dumps(sli)
                })
                block_index[ax] -= face
        client.close()
        return block

    locally_labeled = cache_dir.cache_im(
        lambda: da.map_blocks(compute_slices, locally_labeled, cumsum_da_arr,
                              meta=np.zeros(tuple(), dtype=output_dtype)),
        cid='locally_labeled_with_cumsum'
    )

    if is_logging:
        print('Process locally to obtain a lower triangular adjacency matrix')
    read_kv_store = PairKVStore(db_path)
    lower_adj = set()
    for value1, value2 in read_kv_store.read_all():
        if value1 is None or value2 is None:
            continue
        sli1, sli2 = pickle.loads(value1).flatten(), pickle.loads(value2).flatten()
        sli = np.stack((sli1, sli2), axis=1)
        tups = cvpl_algorithms.np_unique(sli, axis=0)
        for row in tups.tolist():
            i1, i2 = row
            if i2 < i1:
                tmp = i2
                i2 = i1
                i1 = tmp
            if i1 == 0:
                continue
            assert i1 < i2, f'i1={i1} and i2={i2}!'  # can not be equal because indices are globally unique here
            tup = (i2, i1)
            if tup not in lower_adj:
                lower_adj.add(tup)
    connected_components = find_connected_components(lower_adj)
    if is_logging:
        print('Compute final indices remap array')
    ind_map_np = np.arange(total_nlbl + 1, dtype=output_dtype)
    assigned_mask = np.zeros((total_nlbl + 1), dtype=np.uint8)
    assigned_mask[0] = 1  # we don't touch background class
    comp_i = 0
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

    read_kv_store.close()

    if is_logging:
        print(f'comp_i={comp_i}, Remapping the indices array to be globally consistent')
    client = viewer_args['client']
    ind_map_scatter = client.scatter(ind_map_np, broadcast=True)

    def local_to_global(block, block_info, ind_map_scatter):
        return ind_map_scatter[block]

    globally_labeled = cache_dir.cache_im(
        lambda: locally_labeled.map_blocks(func=local_to_global, meta=np.zeros(tuple(), dtype=output_dtype),
                                           ind_map_scatter=ind_map_scatter),
        cid='globally_labeled'
    )
    result_arr = globally_labeled
    if not is_dask:
        if is_logging:
            print('converting the result to NDBlock')
        result_arr = NDBlock(result_arr)
    if is_logging:
        print('Function ends')

    server.close()  # TODO: find way to move this to where it should be

    return result_arr, comp_i
