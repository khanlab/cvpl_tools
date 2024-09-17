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
    if is_dask:
        im = NDBlock(im)

    def map_block(block: npt.NDArray, block_info: dict):
        lbl_im = scipy_label(block, output=output_dtype)[0]
        return lbl_im

    def to_max(block: npt.NDArray, block_info: dict):
        return block.max(keepdims=True)

    # compute locally labelled chunks and save their bordering slices
    if is_logging:
        print('Locally label the image')
    locally_labeled = cache_dir.cache_im(
        lambda: NDBlock.map_ndblocks([im], fn=map_block, out_dtype=output_dtype)
    )
    if is_logging:
        print('Taking the max of each chunk to obtain number of labels')
    new_slices = list(tuple(slice(0, 1) for _ in range(ndim)) for _ in locally_labeled.get_slices_list())
    nlbl_ndblock_arr = NDBlock.map_ndblocks([locally_labeled], fn=to_max, out_dtype=output_dtype,
                                            new_slices=new_slices)
    if is_logging:
        print('Convert number of labels of chunks to numpy array')
    print(nlbl_ndblock_arr.get_repr_format())
    print(nlbl_ndblock_arr.is_numpy())
    print(type(nlbl_ndblock_arr.arr))
    nlbl_np_arr = nlbl_ndblock_arr.as_numpy()
    if is_logging:
        print('Compute prefix sum and reshape back')
    cumsum_np_arr = np.cumsum(nlbl_np_arr)
    assert cumsum_np_arr.ndim == 1
    total_nlbl = cumsum_np_arr[-1].item()
    cumsum_np_arr[1:] = cumsum_np_arr[:-1]
    cumsum_np_arr[0] = 0
    cumsum_np_arr = cumsum_np_arr.reshape(nlbl_np_arr.shape)
    if is_logging:
        print(f'total_nlbl={total_nlbl}, Convert prefix sum to a dask array then to NDBlock')
    cumsum_ndblock_arr = NDBlock(da.from_array(cumsum_np_arr, chunks=(1,) * cumsum_np_arr.ndim))

    # Prepare cache file to be used
    if is_logging:
        print('Setting up cache sqlite database')
    _, cache_file = cache_dir.cache()
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

    def compute_slices(block1: npt.NDArray, block2: npt.NDArray, block_info: dict = None):
        # block1 is the local label, block2 is the single element prefix summed number of labels

        client = partd.Client(server_address)
        block_index = list(block_info[0]['chunk-location'])
        block1 = block1 + (block1 != 0).astype(block1.dtype) * block2
        for ax in range(block1.ndim):
            for face in range(2):
                block_index[ax] += face
                indstr = '_'.join(str(index) for index in block_index) + f'_{ax}'
                sli_idx = face * (block1.shape[ax] - 1)
                sli = np.take(block1, indices=sli_idx, axis=ax)
                client.append({
                    indstr: pickle.dumps(sli)
                })
                block_index[ax] -= face
        client.close()
        return block1
    locally_labeled = cache_dir.cache_im(
        lambda: NDBlock.map_ndblocks([locally_labeled, cumsum_ndblock_arr],
                                     compute_slices,
                                     out_dtype=output_dtype,
                                     use_input_index_as_arrloc=0)
    )
    server.close()

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
    ind_map = {i: i for i in range(1, total_nlbl + 1)}
    for i2, i1 in lower_adj:
        if ind_map[i2] > i1:
            ind_map[i2] = i1
    if is_logging:
        print('Compute final indices remap array')
    ind_map_np = np.zeros((total_nlbl + 1,), dtype=output_dtype)
    for i in range(1, total_nlbl + 1):
        direct_connected = ind_map[i]
        ind_map_np[i] = direct_connected
        ind_map_np[i] = ind_map_np[direct_connected]
    ind_map_np = ind_map_np[ind_map_np]

    read_kv_store.close()

    if is_logging:
        print('Remapping the indices array to be globally consistent')
    client = viewer_args['client']
    ind_map_scatter = client.scatter(ind_map_np, broadcast=True)

    def local_to_global(block, block_info, ind_map_scatter):
        return ind_map_scatter[block]
    globally_labeled = cache_dir.cache_im(
        fn=lambda: NDBlock.map_ndblocks([locally_labeled],
                                        fn=local_to_global,
                                        out_dtype=output_dtype,
                                        fn_args=dict(ind_map_scatter=ind_map_scatter)),
        cid='globally_labeled'
    )
    result_arr = globally_labeled.as_dask_array(tmp_dirpath=f'{cache_file.path}/to_dask_array')
    if not is_dask:
        result_arr = NDBlock(result_arr)
    if is_logging:
        print('Function ends')
    return result_arr, total_nlbl

