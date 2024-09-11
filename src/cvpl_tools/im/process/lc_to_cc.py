from typing import Sequence

from cvpl_tools.im import seg_process
from cvpl_tools.im.seg_process import SegProcess, lc_interpretable_napari, heatmap_logging, map_ncell_vector_to_total
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io


class CountLCBySize(SegProcess):
    """Counting list of cells by size

    Several features:
    1. A size threshold, below which each contour is counted as a single cell (or part of a single cell,
    in the case it is neighbor to boundary of the image)
    2. Above size threshold, the contour is seen as a cluster of cells an estimate of cell count is given
    based on the volume of the contour
    3. For cells on the boundary location, their estimated ncell is penalized according to the distance
    between the cell centroid and the boundary of the image; if the voxels of the cell do not touch
    edge, this penalty does not apply
    4. A min_size threshold, below (<=) which the contour is simply discarded because it's likely just
    an artifact
    """

    def __init__(self,
                 size_threshold: int | float = 25.,
                 volume_weight: float = 6e-3,
                 border_params: tuple[float, float, float] = (3., -.5, 2.),
                 min_size: int | float = 0,
                 reduce: bool = False):
        super().__init__()
        self.size_threshold = size_threshold
        self.volume_weight = volume_weight
        self.border_params = border_params
        self.min_size = min_size
        self.reduce = reduce

    def cc_list(self, lc: npt.NDArray[np.float64], ndim: int, os_shape: tuple) -> npt.NDArray[np.float64]:
        """Assumption: lc[:, ndim] is nvoxel and lc[:, ndim + 1] is edge_contact"""
        ncells = {}
        dc = []
        dc_idx_to_centroid_idx = {}
        for i in range(lc.shape[0]):
            nvoxel = lc[i, ndim].item()
            if nvoxel <= self.min_size:
                ncells[i] = 0.
            else:
                ncells[i] = 0.

                # if no voxel touch the boundary, we do not want to apply the edge penalty
                on_edge = lc[i, ndim + 1].item()
                if on_edge:
                    dc_idx_to_centroid_idx[len(dc)] = i
                    dc.append(lc[i, :ndim])
                else:
                    ncells[i] = 1

                if nvoxel > self.size_threshold:
                    ncells[i] += (nvoxel - self.size_threshold) * self.volume_weight
        ps = CountLCEdgePenalized(os_shape, self.border_params)

        if len(dc) == 0:
            dc_centroids = np.zeros((0, ndim), dtype=np.float64)
        else:
            dc_centroids = np.array(dc, dtype=np.float64)
        dc_ncells = ps.cc_list(dc_centroids, (0,) * ndim)
        for dc_idx in dc_idx_to_centroid_idx:
            i = dc_idx_to_centroid_idx[dc_idx]
            ncells[i] += dc_ncells[dc_idx]
        ncells = np.array([ncells[i] for i in range(len(ncells))], dtype=np.float64)
        return ncells

    def np_features(self,
                    lc: npt.NDArray[np.float64],
                    block_info=None) -> npt.NDArray[np.float64]:
        if block_info is None:
            return np.zeros(tuple(), dtype=np.float64)

        slices = block_info[0]['array-location']
        ndim = len(slices)
        os_shape = tuple(s.stop for s in slices)
        cc_list = self.cc_list(lc, ndim, os_shape)
        features = np.concatenate((lc[:, :ndim], cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, lc: NDBlock[np.float64]) -> NDBlock[np.float64]:
        return NDBlock.map_ndblocks([NDBlock(lc)], self.np_features, out_dtype=np.float64)

    def forward(self,
                lc: NDBlock[np.float64],
                ndim: int,
                cid: str = None,
                viewer_args: dict = None
                ) -> npt.NDArray[np.float64]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)
        ndblock = cache_dir.cache_im(fn=lambda: self.feature_forward(lc),
                                     cid='lc_by_size_features',
                                     cache_level=1)

        dp = viewer_args.get('display_points', True)
        if viewer:
            if dp:
                features = ndblock.reduce(force_numpy=True)
                features = features[features[:, -1] > 0., :]
                lc_interpretable_napari('bysize_ncells',
                                        features, viewer, ndim, ['ncells'])

            aggregate_ndblock: NDBlock[np.float64] = cache_dir.cache_im(
                fn=lambda: map_ncell_vector_to_total(ndblock), cid='aggregate_ndblock',
                cache_level=2
            )
            if dp:
                aggregate_features: npt.NDArray[np.float64] = cache_dir.cache_im(
                    fn=lambda: aggregate_ndblock.reduce(force_numpy=True), cid='block_cell_count',
                    cache_level=2
                )
                lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                        ndim, ['ncells'], text_color='red')

            heatmap_cache_exists, heatmap_cache_dir = cache_dir.cache(is_dir=True, cid='cell_density_map')
            chunk_size = lc.get_chunksize()
            heatmap_logging(aggregate_ndblock, heatmap_cache_exists, heatmap_cache_dir, viewer_args, chunk_size)

        ndblock = ndblock.select_columns([-1])

        if self.reduce:
            ndblock = ndblock.reduce(force_numpy=False)
        ndblock = ndblock.sum(keepdims=True)
        return ndblock


class CountLCEdgePenalized(SegProcess):
    """From a list of cell centroid locations, calculate a cell count estimate

    You need to provide an image_shape parameter due to the fact that lc does not contain
    information about input image shape

    Each centroid is simply treated as 1 cell when they are sufficiently far from the edge,
    but as they get closer to the edge the divisor becomes >1. and their estimate decreases
    towards 0, since cells near the edge may be double-counted (or triple or more counted
    if at a corner etc.)
    """

    def __init__(self,
                 chunks: Sequence[Sequence[int]] | Sequence[int],
                 border_params: tuple[float, float, float] = (3., -.5, 2.),
                 reduce: bool = False):
        """Initialize a CountLCEdgePenalized object

        Args:
            chunks: Shape of the blocks over each axis
            border_params: Specify how the cells on the border gets discounted. Formula is:
                intercept, dist_coeff, div_max = self.border_params
                mults = 1 / np.clip(intercept - border_dists * dist_coeff, 1., div_max)
                cc_list = np.prod(mults, axis=1)
            reduce: If True, reduce the results into a Numpy 2d array calling forward()
        """
        super().__init__()
        if isinstance(chunks[0], int):
            # Turn Sequence[int] to Sequence[Sequence[int]]
            # assume single numpy block, at index (0, 0, 0)
            chunks = tuple((chunks[i],) for i in range(len(chunks)))
        self.chunks = chunks
        self.numblocks = tuple(len(c) for c in chunks)
        self.border_params = border_params
        self.reduce = reduce

        intercept, dist_coeff, div_max = border_params
        assert intercept >= 1., f'intercept has to be >= 1. as divisor must be >= 1! (intercept={intercept})'
        assert dist_coeff <= 0., (f'The dist_coeff needs to be non-positive so divisor decreases as cell is further '
                                  f'from the edge')
        assert div_max >= 1., f'The divisor is >= 1, but got div_max < 1! (div_max={div_max})'

    def cc_list(self,
                lc: npt.NDArray[np.float64],
                block_index: tuple) -> npt.NDArray[np.float64]:
        """Returns a cell count estimate for each contour in the list of centroids

        Args:
            lc: The list of centroids to be given cell estimates for
            block_index: The index of the block which this lc corresponds to

        Returns:
            A 1-d list, each element is a scalar cell count for the corresponding contour centroid in lc
        """
        block_shape = np.array(
            tuple(self.chunks[i][block_index[i]] for i in range(len(self.chunks))),
            dtype=np.float64
        )
        midpoint = (block_shape * .5)[None, :]

        # compute border distances in each axis direction
        border_dists = np.abs((lc[:, :len(self.chunks)] + midpoint) % block_shape - (midpoint - .5))

        intercept, dist_coeff, div_max = self.border_params
        mults = 1 / np.clip(intercept + border_dists * dist_coeff, 1., div_max)
        cc_list = np.prod(mults, axis=1)
        return cc_list

    def np_features(self, lc: npt.NDArray[np.float64], block_info=None) -> npt.NDArray[np.float64]:
        """Calculate cell counts, then concat centroid locations to the left of cell counts"""
        cc_list = self.cc_list(lc, block_info[0]['chunk-location'])
        features = np.concatenate((lc[:, :len(self.chunks)], cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, lc: NDBlock[np.float64]) -> NDBlock[np.float64]:
        return NDBlock.map_ndblocks([lc], self.np_features, out_dtype=np.float64)

    def forward(self,
                lc: NDBlock[np.float64],
                cid: str = None,
                viewer_args: dict = None) -> NDBlock[np.float64]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        assert lc.get_numblocks() == self.numblocks, ('numblocks could not match up for the chunks argument '
                                                      f'provided, expected {self.numblocks} but got '
                                                      f'{lc.get_numblocks()}')
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)

        print('1')

        ndblock = cache_dir.cache_im(fn=lambda: self.feature_forward(lc), cid='lc_cc_edge_penalized',
                                     cache_level=1)

        print('2')

        dp = viewer_args.get('display_points', True)
        if viewer:
            print('3')
            if viewer_args.get('display_checkerboard', True):
                checkerboard = cache_dir.cache_im(fn=lambda: cvpl_ome_zarr_io.dask_checkerboard(self.chunks),
                                                  cid='checkerboard',
                                                  cache_level=2,
                                                  viewer_args=viewer_args | dict(is_label=True))
            print('4')

            if dp:
                print('5')
                features = ndblock.reduce(force_numpy=True)
                lc_interpretable_napari('lc_cc_edge_penalized', features, viewer,
                                        len(self.chunks), ['ncells'])
                print('6')

            aggregate_ndblock: NDBlock[np.float64] = cache_dir.cache_im(
                fn=lambda: map_ncell_vector_to_total(ndblock), cid='aggregate_ndblock',
                cache_level=2
            )
            print('7')
            if dp:
                print('8')
                aggregate_features: npt.NDArray[np.float64] = cache_dir.cache_im(
                    fn=lambda: aggregate_ndblock.reduce(force_numpy=True), cid='block_cell_count',
                    cache_level=2
                )
                lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                        len(self.chunks), ['ncells'], text_color='red')
                print('9')

            heatmap_cache_exists, heatmap_cache_dir = cache_dir.cache(is_dir=True, cid='cell_density_map')
            chunk_size = tuple(ax[0] for ax in self.chunks)
            heatmap_logging(aggregate_ndblock, heatmap_cache_exists, heatmap_cache_dir, viewer_args, chunk_size)
            print('10')

        ndblock = ndblock.select_columns([-1])
        print('11')
        if self.reduce:
            print('12')
            ndblock = ndblock.reduce(force_numpy=False)
            print('13')
        ndblock = ndblock.sum(keepdims=True)
        print('14')
        return ndblock

