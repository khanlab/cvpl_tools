from typing import Sequence

from cvpl_tools.im.fs import CachePointer
from cvpl_tools.im.seg_process import SegProcess, lc_interpretable_napari
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.im.algorithms as algorithms
import dask.array as da
from scipy.ndimage import find_objects
import dask
from dask.distributed import print as dprint


class DirectOSToLC(SegProcess):
    """Convert a 0-N contour mask to a list of N centroids, one for each contour

    The converted list of centroids is in the same order as the original contour order (The contour labeled
    1 will come first and before contour labeled 2, 3 and so on)
    """

    def __init__(self,
                 min_size: int = 0,
                 reduce: bool = False,
                 is_global: bool = False,
                 ex_statistics: Sequence[str] = tuple()):
        super().__init__()
        self.reduce = reduce
        self.min_size = min_size
        self.is_global = is_global
        self.ex_statistics = ex_statistics
        self.full_statistics_map = dict(
            nvoxel=0,
            edge_contact=1,
            id=2
        )  # these columns are appended to feature array; self.ex_statistics selects from these to return
        self.global_full_statistics_map = dict(
            nvoxel=0,
            id=1
        )
        self.ret_cols = tuple(self.full_statistics_map[name] for name in self.ex_statistics)

        self._ndblock: NDBlock = None
        self._reduced_features = None
        self._reduced_np_features = None

    @property
    def reduced_features(self):
        if self._reduced_features is None:
            self._reduced_features = self._ndblock.reduce(force_numpy=False)
        return self._reduced_features

    @property
    def reduced_np_features(self):
        if self._reduced_np_features is None:
            rf = self.reduced_features
            if isinstance(rf, da.Array):
                rf = rf.compute()
            self._reduced_np_features = rf
        return self._reduced_np_features

    def np_features(self, block: npt.NDArray[np.int32], block_info=None) \
            -> npt.NDArray[np.float64]:
        if block_info is not None:
            mat_width = block.ndim + len(self.full_statistics_map)
            idx_max = np.array(tuple(d - 1 for d in block.shape), dtype=np.int64)

            slices = block_info[0]['array-location']
            _contours_argwhere, _ids = algorithms.npindices_from_os(block, is_sparse=True)
            contours_argwhere, ids = [], []
            for i in range(len(_contours_argwhere)):
                contour = _contours_argwhere[i]
                assert contour.shape[1] == block.ndim, (f'Contour shape {contour.shape} returned by npindices_from_os '
                                                        f'does not match block.ndim={block.ndim}')
                if len(contour) > self.min_size:
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
            for name, j in self.full_statistics_map.items():
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

    def feature_forward(self, im: npt.NDArray[np.int32] | da.Array) -> NDBlock[np.float64]:
        return NDBlock.map_ndblocks([NDBlock(im)], self.np_features, out_dtype=np.float64)

    def aggregate_by_id(self):
        """Aggregate self._ndblock by id"""

        ref_ndblock: NDBlock = self._ndblock
        ndim = ref_ndblock.get_ndim()
        block_indices = ref_ndblock.get_block_indices()
        slices_list = ref_ndblock.get_slices_list()
        chunk_shape = np.array(tuple(s.stop - s.start for s in slices_list[0]), dtype=np.float64)
        recons = {ind: [] for ind in block_indices}

        rf = self.reduced_np_features
        rf = rf[np.argsort(rf[:, -1])]
        if rf.shape[0] == 0:
            cnt_ranges = []
        else:
            cnt_ranges = list(find_objects(rf[:, -1].astype(np.int32)))

        nvoxel_ind = self.full_statistics_map['nvoxel']
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
                rows = np.zeros((0, ndim + len(self.global_full_statistics_map)), dtype=np.float64)
            else:
                rows = np.array(recons[ind], dtype=np.float64)
            recons[ind] = (rows, slices_list[i])
        self._ndblock = NDBlock.create_from_dict_and_properties(recons,
                                                                ref_ndblock.get_properties() | dict(is_numpy=True))

    def forward(self,
                im: npt.NDArray[np.int32] | da.Array,
                cptr: CachePointer,
                viewer_args: dict = None) -> NDBlock[np.float64] | npt.NDArray[np.float64]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)

        cdir = cptr.subdir()

        def compute():
            self._ndblock = cdir.cache_im(fn=lambda: self.feature_forward(im),
                                          cache_level=1,
                                          cid='block_level_lc_ndblock')
            if self.is_global:
                # update and aggregate the rows in ndblock that correspond to the same contour
                self.aggregate_by_id()
                self._reduced_features = None
                self._reduced_np_features = None
            return self._ndblock

        self._ndblock = cdir.cache_im(fn=compute,
                                      cache_level=1,
                                      cid='lc_ndblock')

        if viewer and viewer_args.get('display_points', True):
            if self.is_global:
                extras = list(self.global_full_statistics_map.keys())
            else:
                extras = list(self.full_statistics_map.keys())
            lc_interpretable_napari(
                'os_to_lc_centroids',
                self.reduced_np_features,
                viewer,
                im.ndim,
                extras
            )

        ret_cols = tuple(range(im.ndim)) + tuple(r + im.ndim for r in self.ret_cols)
        if self.reduce:
            ret = self.reduced_features[:, ret_cols]
        else:
            ret = self._ndblock.select_columns(ret_cols)
        self._ndblock = None
        self._reduced_features = None
        self._reduced_np_features = None

        return ret
