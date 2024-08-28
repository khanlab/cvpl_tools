"""
Segmentation and post-processing.

This file is for methods generating (dask) single class segmentation masks of binary or ordinal types, the latter of
which is a 0-N segmentation of N objects of the same class in an image.

Methods in this file are designed such that they can be easily swapped out and whose performance is then compared
against each other to manual segmentations over a dataset.

The input to these methods are either input 3d single-channel image of type np.float32, or input image paired with
a deep learning segmentation algorithm. The output may be cell count #, binary mask
(np.uint8) or ordinal mask (np.int32).

Conceptually, we define the following input/output types:
IN - Input Image (np.float32) between min=0 and max=1, this is the brightness dask image as input
BS - Binary Segmentation (3d, np.uint8), this is the binary mask single class segmentation
OS - Ordinal Segmentation (3d, np.int32), this is the 0-N where contour 1-N each denotes an object; also single class
LC - List of Centroids, this contains a list of centroids for each block in the original image
CC - Cell Count Map (3d, np.float32), a cell count number (estimate, can be float) for each block
CD - Cell Density Map (3d, np.float32), this is a down sampled map of the cell density of the brain
ATLAS_MAP - A function that maps from the pixel location within brain to brain regions
ATLAS_CD - Atlas, this summarizes cell density about the brain grouped by region of the brain

Each type above may have associated parameters of them that is not changed.
And we work with the following types of methods as steps to obtain final segmentations from the input mask:
preprocess: IN -> IN
e.g. gaussian_blur

predict_bs: IN -> BS
e.g. cellseg3d_predict, simple_threshold

predict_lc: IN -> LC
e.g. blob_dog

predict_cc: IN -> CC
e.g. scaled_sum_intensity

binary_to_inst: BS -> OS
e.g. direct_bs_to_os, watershed

binary_and_centroids_to_inst: (BS, LC) -> OS
e.g. in_contour_voronoi

os_to_lc: OS -> LC
e.g. direct_os_to_centroids

count_from_lc: LC -> CC
e.g. count_lc_ncentroid, count_edge_penalized_lc_ncentroid

count_from_os: OS -> CC
e.g. count_os_ncontour, count_edge_penalized_os_ncontour, count_os_byvolume

count_to_atlas_cell_density: CC -> ATLAS_CD
e.g. average_by_region

Each method is an object implementing the SegStage interface that has the following methods:
- forward(*args, cid=None, viewer=None) -> out

About viewer visualization: They need not be interpretable for large images but should handle small sized dask
images for debugging purpose.
"""

import abc
import logging
from typing import Callable, Any, Sequence, final, Iterator

import cvpl_tools.im.fs as imfs
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
import cvpl_tools.im.algorithms as algorithms
import cvpl_tools.im.ndblock as ndblock
from cvpl_tools.im.ndblock import NDBlock
import dask.array as da
from dask.distributed import print as dprint
import napari
import numpy as np
import numpy.typing as npt

import skimage
from scipy.ndimage import (
    gaussian_filter as gaussian_filter,
    label as instance_label,
    find_objects as find_objects
)


# ------------------------------------Helper Functions---------------------------------------


def lc_interpretable_napari(layer_name: str,
                            lc: npt.NDArray,
                            viewer: napari.Viewer,
                            ndim: int,
                            extra_features: Sequence,
                            text_color: str = 'green'):
    """This function is used to display feature points for LC-typed output

    Args:
        layer_name: displayed name of the layer
        lc: The list of features, each row of length (ndim + nextra)
        viewer: Napari viewer to add points to
        ndim: dimension of the image
        extra_features: extra features to be displayed as text
        text_color: to be used as display text color
    """
    # reference: https://napari.org/stable/gallery/add_points_with_features.html
    nextra = len(extra_features)
    assert isinstance(lc, np.ndarray), 'lc should be of type np.ndarray!'
    assert lc.ndim == 2, (f'Wrong dimension for list of centroids, expected ndim=2, but got lc={lc} and '
                          f'lc.shape={lc.shape}')
    assert lc.shape[1] == nextra + ndim, (f'Wrong number of features for list of centroids, expected length along '
                                          f'first dimension to be nextra + ndim={nextra + ndim} but got lc={lc} and '
                                          f'lc.shape={lc.shape}')

    features = {
        extra_features[i]: lc[:, ndim + i] for i in range(nextra)
    }

    strings = [extra_features[i] + '={' + extra_features[i] + ':.2f}' for i in range(nextra)]
    text_parameters = {
        'string': '\n'.join(strings),
        'size': 9,
        'color': text_color,
        'anchor': 'center',
    }
    import time
    stime = time.time()
    viewer.add_points(lc[:, :ndim],
                      size=1.,
                      ndim=ndim,
                      name=layer_name,
                      features=features,
                      text=text_parameters,
                      visible=False)


# ---------------------------------------Interfaces------------------------------------------


logger = logging.getLogger('SEG_PROCESSES')


# class CacheType(enum.Enum):
#     """Useful for determining the amount and what kinds of cache we need"""
#     TEMPORARY_FILE = 0
#     INTERMEDIATE_FILE = 1
#     VIEWER_FILE = 2  # images to be viewed by napari Viewer or other kinds of visualization
#     INTERMEDIATE_RESULT = 3
#     FINAL_RESULT = 4


class SegProcess(abc.ABC):
    def __init__(self):
        self.tmpdir: imfs.CacheDirectory | None = None
        self.cid_prefix: str | None = None

    def set_tmpdir(self, tmpdir: imfs.CacheDirectory):
        """Use this to cache output results

        For output images that involves interpretation with napari, this is necessary for a smooth display of the
        results.

        Args:
            tmpdir: Any cvpl_tools.im.fs.CacheDirectory object that allows many temporary objects to be written
                as cache in the directory
        """
        assert isinstance(tmpdir, imfs.CacheDirectory), \
            f'Expected type cvpl_tools.im.fs.CacheDirectory, got {type(tmpdir)}'
        self.tmpdir = tmpdir

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:  # removed viewer explicitly since that seems to cause issue in IDE docs
        """Compute outputs from inputs (args)

        Args:
            *args: args inputs to forward()
            viewer_args: If not specified, this is just a forward step; if specified, visualization of the step
                underlying working mechanism will be displayed in Napari viewer object viewer_args['viewer'] for
                interpretation of results and debugging purposes
            **kwargs: kwargs inputs to forward

        Returns:
            computed output
        """
        raise NotImplementedError("SegProcess base class does not implement forward()!")


class BlockToBlockProcess(SegProcess):
    def __init__(self, out_dtype: np.dtype, is_label=False):
        super().__init__()
        self.out_dtype = out_dtype
        self.is_label = is_label

    @final
    def forward(self, im: npt.NDArray | da.Array, cid: str | None = None, viewer_args=None) \
            -> npt.NDArray | da.Array:
        if viewer_args is None:
            viewer_args = {}
        if isinstance(im, np.ndarray):
            result = self.np_forward(im)
        elif isinstance(im, da.Array):
            result = im.map_blocks(
                self.np_forward,
                meta=np.array(tuple(), dtype=self.out_dtype),
                dtype=self.out_dtype
            )
        elif isinstance(im, NDBlock):
            result = NDBlock.map_ndblocks([im],
                                          self.np_forward,
                                          out_dtype=self.out_dtype,
                                          use_input_index_as_arrloc=0)
        else:
            raise TypeError(f'Invalid im type: {type(im)}')

        result = self.tmpdir.cache_im(lambda: result,
                                      cid=cid,
                                      cache_level=1,
                                      viewer_args=viewer_args | dict(is_label=self.is_label))
        return result

    @abc.abstractmethod
    def np_forward(self, im: npt.NDArray, block_info=None) -> npt.NDArray:
        raise NotImplementedError('Subclass of BlockToBlockProcess should implement np_forward!')


# ---------------------------------------Preprocess------------------------------------------


class GaussianBlur(BlockToBlockProcess):
    def __init__(self, sigma: float):
        super().__init__(np.float32, is_label=False)
        self.sigma = sigma

    def np_forward(self, im: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.float32]:
        return gaussian_filter(im, sigma=self.sigma)


# -------------------------------------Predict Binary----------------------------------------


class BSPredictor(BlockToBlockProcess):
    def __init__(self, pred_fn: Callable):
        super().__init__(np.uint8, is_label=True)
        self.pred_fn = pred_fn

    def np_forward(self, im: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.uint8]:
        return self.pred_fn(im)


class SimpleThreshold(BlockToBlockProcess):
    def __init__(self, threshold: float):
        super().__init__(np.uint8, is_label=True)
        self.threshold = threshold

    def np_forward(self, im: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.uint8]:
        return (im > self.threshold).astype(np.uint8)


# --------------------------------Predict List of Centroids-----------------------------------


class BlobDog(SegProcess):
    def __init__(self, min_sigma=1, max_sigma=2, threshold: float = 0.1, reduce=False):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.reduce = reduce

    def np_features(self, block: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.float32]:
        if block_info is not None:
            slices = block_info[0]['array-location']
            lc = skimage.feature.blob_dog(np.array(block * 255, dtype=np.uint8),
                                          min_sigma=self.min_sigma,
                                          max_sigma=self.max_sigma,
                                          threshold=self.threshold).astype(np.float32)  # N * (ndim + 1) ndarray
            start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float32)
            lc[:, :block.ndim] += start_pos[None, :]
            return lc
        else:
            return block

    def feature_forward(self, im: npt.NDArray[np.float32] | da.Array) -> NDBlock[np.float32]:
        return NDBlock.map_ndblocks([NDBlock(im)], self.np_features, out_dtype=np.float32)

    def forward(self,
                im: npt.NDArray[np.float32] | da.Array,
                cid: str = None,
                viewer_args: dict = None
                ) -> NDBlock:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        ndblock = self.tmpdir.cache_im(lambda: self.feature_forward(im), cid=cid, cache_level=1)

        if viewer and viewer_args.get('display_points', True):
            blobdog = ndblock.reduce(force_numpy=True)
            lc_interpretable_napari('blobdog_centroids', blobdog, viewer, im.ndim, ['sigma'])

        ndblock = ndblock.select_columns(slice(im.ndim))
        if self.reduce:
            ndblock = ndblock.reduce(force_numpy=False)
        return ndblock


# -------------------------------Direct Cell Count Prediction----------------------------------


class SumScaledIntensity(SegProcess):
    def __init__(self, scale: float = .008, min_thres: float = 0., reduce: bool = True,
                 spatial_box_width: int | None = None):
        """Summing up the intensity and scale it to obtain number of cells, directly

        Args:
            scale: Scale the sum of intensity by this to get number of cells
            min_thres: Intensity below this threshold is excluded (set to 0 before summing)
            reduce: reduce result when calling forward
            spatial_box_width: If not None, will use this as the box width for adding points to Napari
        """
        assert scale >= 0, f'{scale}'
        assert 0. <= min_thres <= 1., f'{min_thres}'
        super().__init__()
        self.scale = scale
        self.min_thres = min_thres
        self.reduce = reduce
        self.spatial_box_width = spatial_box_width

    def np_features(self, block: npt.NDArray[np.float32], block_info=None, spatial_box_width=None) \
            -> npt.NDArray[np.float32]:
        if block_info is not None:
            slices = block_info[0]['array-location']
            if spatial_box_width is not None:
                padded_block = algorithms.pad_to_multiple(block, spatial_box_width)
            else:
                padded_block = block
            masked = padded_block * (padded_block > self.min_thres)
            if slices is not None:
                startpoint = np.array([slices[i].start for i in range(len(slices))],
                                      dtype=np.float32)
            else:
                startpoint = np.zeros((block.ndim,), dtype=np.float32)

            if spatial_box_width is not None:
                subblock_shape = (spatial_box_width,) * block.ndim
                masked = algorithms.np_map_block(masked, block_sz=subblock_shape)
                masked: npt.NDArray = masked.sum(axis=tuple(range(block.ndim, block.ndim * 2)))

                features = np.zeros((masked.size, block.ndim + 1), dtype=np.float32)
                inds = np.array(np.indices(masked.shape, dtype=np.float32))
                transpose_axes = tuple(range(1, block.ndim + 1)) + (0,)
                inds = inds.transpose(transpose_axes).reshape(-1, block.ndim)
                features[:, -1] = masked.flatten() * self.scale
                features[:, :-1] = startpoint[None, :] + (inds + .5) * np.array(subblock_shape, dtype=np.float32)
            else:
                features = np.zeros((1, block.ndim + 1), dtype=np.float32)
                features[0, -1] = masked.sum() * self.scale
                features[:, :-1] = startpoint[None, :] + np.array(block.shape, dtype=np.float32) / 2
            return features
        else:
            return block

    def feature_forward(self, im: npt.NDArray[np.float32] | da.Array, spatial_block_width: int = None) \
            -> NDBlock[np.float32]:
        def map_fn(b, block_info):
            return self.np_features(b, block_info, spatial_block_width)

        return NDBlock.map_ndblocks([NDBlock(im)], map_fn, out_dtype=np.float32)

    def forward(self, im: npt.NDArray | da.Array, cid: str = None, viewer_args: dict = None) \
            -> NDBlock | npt.NDArray:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)

        forwarded = cache_dir.cache_im(
            fn=lambda: self.feature_forward(im),
            cid='forwarded',
            cache_level=2
        )

        if viewer:
            mask = cache_dir.cache_im(fn=lambda: im > self.min_thres, cid='ssi_mask',
                                       cache_level=2,
                                       viewer_args=viewer_args | dict(is_label=True))
            if self.spatial_box_width is not None and viewer_args.get('display_points', True):
                ssi = cache_dir.cache_im(
                    fn=lambda: self.feature_forward(im, spatial_block_width=self.spatial_box_width).reduce(force_numpy=True),
                    cid='ssi',
                    cache_level=2
                )
                lc_interpretable_napari('ssi_block', ssi, viewer, im.ndim, ['ncells'])

            aggregate_ndblock: NDBlock[np.float32] = cache_dir.cache_im(
                fn=lambda: map_ncell_vector_to_total(forwarded), cid='aggregate_ndblock',
                cache_level=2
            )
            heatmap_cache_exists, heatmap_cache_dir = cache_dir.cache(is_dir=True, cid='cell_density_map')
            chunk_size = NDBlock(im).get_chunksize()
            heatmap_logging(aggregate_ndblock, heatmap_cache_exists, heatmap_cache_dir, viewer_args, chunk_size)

        def fn():
            ndblock = forwarded.select_columns([-1])
            if self.reduce:
                ndblock = ndblock.reduce(force_numpy=False)
            return ndblock

        ssi_result = cache_dir.cache_im(fn=fn, cid='ssi_result', cache_level=1)

        return ssi_result


# ---------------------------Convert Binary Mask to Instance Mask------------------------------


class DirectBSToOS(BlockToBlockProcess):
    def __init__(self):
        super().__init__(np.int32, is_label=True)

    def np_forward(self, bs: npt.NDArray[np.uint8], block_info=None) -> npt.NDArray[np.int32]:
        lbl_im, nlbl = instance_label(bs)
        return lbl_im


class Watershed3SizesBSToOS(BlockToBlockProcess):
    def __init__(self,
                 size_thres=60.,
                 dist_thres=1.,
                 rst=None,
                 size_thres2=100.,
                 dist_thres2=1.5,
                 rst2=60.):
        super().__init__(np.int32, is_label=True)
        self.size_thres = size_thres
        self.dist_thres = dist_thres
        self.rst = rst
        self.size_thres2 = size_thres2
        self.dist_thres2 = dist_thres2
        self.rst2 = rst2

    def np_forward(self, bs: npt.NDArray[np.uint8], block_info=None) -> npt.NDArray[np.int32]:
        lbl_im = algorithms.round_object_detection_3sizes(bs,
                                                          size_thres=self.size_thres,
                                                          dist_thres=self.dist_thres,
                                                          rst=self.rst,
                                                          size_thres2=self.size_thres2,
                                                          dist_thres2=self.dist_thres2,
                                                          rst2=self.rst2,
                                                          remap_indices=True)
        return lbl_im
    # TODO: better visualization of this stage


# ---------------------------Convert Binary Mask to Instance Mask------------------------------


class BinaryAndCentroidListToInstance(SegProcess):
    """Defines a SegProcess

    This class' instance forward() takes two inputs: The binary mask segmenting objects, and centroids
    detected by the blobdog algorithm. Then it splits the pixels in the binary mask into instance
    segmentation based on the centroids found by blobdog. The result is a more finely segmented mask
    where objects closer together are more likely to be correctly segmented as two
    """

    def __init__(self, maxSplit: int = 10):
        """Initialize a BinaryAndCentroidListToInstance object

        Args:
            maxSplit: If a contour has number of corresponding centroids above (>) this number in lc,
                then the contour is left as is; this parameter exists for optimization purpose, since
                the larger contours have a time complexity O(N * S) to its spatial size S and the
                number of contours N
        """
        super().__init__()
        self.maxSplit = maxSplit

    def split_ndarray_by_centroid(
            self,
            centroids: list[npt.NDArray[np.int64]],
            indices: list[int],
            X: tuple[npt.NDArray]
    ) -> npt.NDArray[np.int32]:
        N = len(centroids)
        assert N >= 2
        assert N == len(indices)
        arr_shape = X[0].shape
        indices = np.array(indices, dtype=np.int32)
        if N < 10:
            X = np.array(X)

            idxD = np.zeros(arr_shape, dtype=np.int32)
            minD = np.ones(arr_shape, dtype=np.float32) * 1e10
            for i in range(N):
                centroid = centroids[i]
                D = X - np.expand_dims(centroid, list(range(1, X.ndim)))
                D = np.linalg.norm(D.astype(np.float32), axis=0)  # euclidean distance
                new_mask = D < minD
                idxD = new_mask * indices[i] + ~new_mask * idxD
                minD = new_mask * D + ~new_mask * minD
            return idxD
        else:
            centroids = np.array(centroids, dtype=np.int32)
            idxD = algorithms.voronoi_ndarray(arr_shape, centroids)
            return indices[idxD]

    def bacl_forward(self,
                     bs: npt.NDArray[np.uint8],
                     lc: npt.NDArray[np.float32],
                     block_info=None) -> npt.NDArray[np.int32]:
        """For a numpy block and list of centroids in block, return segmentation based on centroids"""

        assert isinstance(bs, np.ndarray) and isinstance(lc, np.ndarray), \
            f'Error: inputs must be numpy for the forward() of this class, got bs={type(bs)} and lc={type(lc)}'

        # first sort each centroid into contour they belong to
        input_slices = block_info[0]['array-location']
        lc = lc.astype(np.int64) - np.array(tuple(s.start for s in input_slices), dtype=np.int64)[None, :]
        lbl_im, max_lbl = instance_label(bs)

        lbl_im: npt.NDArray[np.int32] = lbl_im.astype(np.int32)
        max_lbl: int

        # Below, index 0 is background - centroids fall within this are discarded
        contour_centroids = [[] for _ in range(max_lbl + 1)]

        for centroid in lc:
            c_ord = int(lbl_im[tuple(centroid)])
            contour_centroids[c_ord].append(centroid)

        # now we compute the contours, and brute-force calculate what centroid each pixel is closest to
        object_slices = list(find_objects(lbl_im))
        new_lbl = max_lbl + 1

        for i in range(1, max_lbl + 1):
            slices = object_slices[i - 1]
            if slices is None:
                continue

            # if there are 0 or 1 centroid in the contour, we do nothing
            centroids = contour_centroids[i]  # centroids fall within the current contour
            ncentroid = len(centroids)
            if ncentroid <= 1 or ncentroid > self.maxSplit:
                continue

            # otherwise, divide the contour and map pixels to each
            indices = [i] + [lbl for lbl in range(new_lbl, new_lbl + ncentroid - 1)]
            new_lbl += ncentroid - 1
            mask = lbl_im[slices] == i
            stpt = np.array(tuple(s.start for s in slices), dtype=np.int64)
            centroids = [centroid - stpt for centroid in centroids]
            divided = algorithms.coord_map(mask.shape,
                                           lambda *X: self.split_ndarray_by_centroid(centroids, indices, X))

            lbl_im[slices] = lbl_im[slices] * ~mask + divided * mask

        return lbl_im

    def forward(self,
                bs: npt.NDArray[np.uint8] | da.Array,
                lc: NDBlock[np.float32],
                cid: str | None = None,
                viewer_args: dict = None
                ) -> npt.NDArray[np.int32] | da.Array:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)

        if viewer:
            def compute_lbl():
                if isinstance(bs, np.ndarray):
                    lbl_im = instance_label(bs)[0]
                else:
                    lbl_im = bs.map_blocks(
                        lambda block: instance_label(block)[0].astype(np.int32), dtype=np.int32
                    )
                return lbl_im

            lbl_im = cache_dir.cache_im(fn=compute_lbl, cid='before_split',
                                         cache_level=2,
                                         viewer_args=viewer_args | dict(is_label=True))

        def compute_result():
            nonlocal bs, lc
            bs = NDBlock(bs)
            is_numpy = bs.is_numpy() or lc.is_numpy()
            if is_numpy:
                # if one is numpy but not both numpy, force both of them to be numpy
                if not lc.is_numpy():
                    lc = lc.reduce(force_numpy=True)
                elif not bs.is_numpy():
                    bs = bs.as_numpy()

            ndblock = NDBlock.map_ndblocks([bs, lc], self.bacl_forward,
                                           out_dtype=np.int32, use_input_index_as_arrloc=0)
            if is_numpy:
                result = ndblock.as_numpy()
            else:
                tmp_path = self.tmpdir.cache(is_dir=False, cid='blocks')[1].path
                result = ndblock.as_dask_array(tmp_path)
            return result

        result = cache_dir.cache_im(fn=compute_result, cid='result',
                                     cache_level=1,
                                     viewer_args=viewer_args | dict(is_label=True))
        return result


# ---------------------------Ordinal Segmentation to List of Centroids-------------------------


class DirectOSToLC(SegProcess):
    """Convert a 0-N contour mask to a list of N centroids, one for each contour

    The converted list of centroids is in the same order as the original contour order (The contour labeled
    1 will come first and before contour labeled 2, 3 and so on)
    """

    def __init__(self, min_size: int = 0, reduce=False):
        super().__init__()
        self.reduce = reduce
        self.min_size = min_size

    def np_features(self, block: npt.NDArray[np.int32], block_info=None) \
            -> npt.NDArray[np.float32]:
        if block_info is not None:
            slices = block_info[0]['array-location']
            contours_np3d = algorithms.npindices_from_os(block)
            lc = [contour.astype(np.float32).mean(axis=0) for contour in contours_np3d
                  if len(contour) > self.min_size]

            if len(lc) == 0:
                lc = np.zeros((0, block.ndim), dtype=np.float32)
            else:
                lc = np.array(lc, dtype=np.float32)
            if slices is not None:
                start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float32)
                lc[:, :block.ndim] += start_pos[None, :]
            return lc
        else:
            return np.zeros(block.shape, dtype=np.float32)

    def feature_forward(self, im: npt.NDArray[np.int32] | da.Array) -> NDBlock[np.float32]:
        return NDBlock.map_ndblocks([NDBlock(im)], self.np_features, out_dtype=np.float32)

    def forward(self,
                im: npt.NDArray[np.int32] | da.Array,
                cid: str = None,
                viewer_args: dict = None) -> NDBlock[np.float32]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        ndblock = self.tmpdir.cache_im(fn=lambda: self.feature_forward(im),
                                       cache_level=1,
                                       cid=cid)

        if viewer and viewer_args.get('display_points', True):
            features = ndblock.reduce(force_numpy=True)
            lc_interpretable_napari('os_to_lc_centroids', features, viewer, im.ndim, [])

        if self.reduce:
            ndblock = ndblock.reduce(force_numpy=False)
        return ndblock


# -----------------------Convert List of Centroids to Cell Count Estimate----------------------


def map_ncell_vector_to_total(ndblock: NDBlock[np.float32]) -> NDBlock[np.float32]:
    """Aggregate the counts in ncell vector to get a single ncell estimate

    Args:
        ndblock: Each block contains a ncell vector

    Returns:
        The summed ncell by block; coordinates of each ncell is the center of the block
    """
    def map_fn(block: npt.NDArray[np.float32], block_info):
        slices = block_info[0]['array-location']
        midpoint = np.array(tuple((s.stop - s.start + 1) / 2 + s.start for s in slices), dtype=np.float32)
        ndim = midpoint.shape[0]
        result = np.zeros((1, ndim + 1), dtype=np.float32)
        result[0, :-1] = midpoint
        result[0, -1] = block[:, -1].sum()
        return result
    return NDBlock.map_ndblocks([ndblock], map_fn, out_dtype=np.float32)


def heatmap_logging(aggregate_ndblock: NDBlock[np.float32],
                    cache_exists: bool,
                    cache_dir: imfs.CacheDirectory,
                    viewer_args: dict,
                    chunk_size: tuple):
    np_arr_path = f'{cache_dir.path}/density_map.npy'
    if not cache_exists:
        block = aggregate_ndblock.select_columns([-1])
        ndim = block.get_ndim()
        def map_fn(block: npt.NDArray[np.float32], block_info=None):
            return block.reshape((1,) * ndim)
        block = block.map_ndblocks([block], map_fn, out_dtype=np.float32)
        block = block.as_numpy()
        np.save(np_arr_path, block)

    if viewer_args is None:
        viewer_args = {}
    viewer: napari.Viewer = viewer_args.get('viewer', None)

    block = np.load(np_arr_path)
    if viewer:
        block = np.log2(block + 1.)
        viewer.add_image(block, name='cell_density_map', scale=chunk_size, blending='additive', colormap='red',
                         translate=tuple(sz / 2 for sz in chunk_size))


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
                lc: npt.NDArray[np.float32],
                block_index: tuple) -> npt.NDArray[np.float32]:
        """Returns a cell count estimate for each contour in the list of centroids

        Args:
            lc: The list of centroids to be given cell estimates for
            block_index: The index of the block which this lc corresponds to

        Returns:
            A 1-d list, each element is a scalar cell count for the corresponding contour centroid in lc
        """
        block_shape = np.array(
            tuple(self.chunks[i][block_index[i]] for i in range(len(self.chunks))),
            dtype=np.float32
        )
        midpoint = (block_shape * .5)[None, :]

        # compute border distances in each axis direction
        border_dists = np.abs((lc + midpoint) % block_shape - (midpoint - .5))

        intercept, dist_coeff, div_max = self.border_params
        mults = 1 / np.clip(intercept + border_dists * dist_coeff, 1., div_max)
        cc_list = np.prod(mults, axis=1)
        return cc_list

    def np_features(self, lc: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.float32]:
        """Calculate cell counts, then concat centroid locations to the left of cell counts"""
        cc_list = self.cc_list(lc, block_info[0]['chunk-location'])
        features = np.concatenate((lc, cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, lc: NDBlock[np.float32]) -> NDBlock[np.float32]:
        return NDBlock.map_ndblocks([lc], self.np_features, out_dtype=np.float32)

    def forward(self,
                lc: NDBlock[np.float32],
                cid: str = None,
                viewer_args: dict = None) -> NDBlock[np.float32]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        assert lc.get_numblocks() == self.numblocks, ('numblocks could not match up for the chunks argument '
                                                      f'provided, expected {self.numblocks} but got '
                                                      f'{lc.get_numblocks()}')
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)

        import time
        ndblock = cache_dir.cache_im(fn=lambda: self.feature_forward(lc), cid='lc_cc_edge_penalized',
                                      cache_level=1)
        if viewer and viewer_args.get('display_points', True):
            if viewer_args.get('display_checkerboard', True):
                checkerboard = cache_dir.cache_im(fn=lambda: cvpl_ome_zarr_io.dask_checkerboard(self.chunks),
                                                   cid='checkerboard',
                                                   cache_level=2,
                                                   viewer_args=viewer_args | dict(is_label=True))

            features = ndblock.reduce(force_numpy=True)
            lc_interpretable_napari('lc_cc_edge_penalized', features, viewer,
                                    len(self.chunks), ['ncells'])

            aggregate_ndblock: NDBlock[np.float32] = cache_dir.cache_im(
                fn=lambda: map_ncell_vector_to_total(ndblock), cid='aggregate_ndblock',
                cache_level=2
            )
            aggregate_features: npt.NDArray[np.float32] = cache_dir.cache_im(
                fn=lambda: aggregate_ndblock.reduce(force_numpy=True), cid='block_cell_count',
                cache_level=2
            )
            lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                    len(self.chunks), ['ncells'], text_color='red')

            heatmap_cache_exists, heatmap_cache_dir = cache_dir.cache(is_dir=True, cid='cell_density_map')
            chunk_size = tuple(ax[0] for ax in self.chunks)
            heatmap_logging(aggregate_ndblock, heatmap_cache_exists, heatmap_cache_dir, viewer_args, chunk_size)

        ndblock = ndblock.select_columns([-1])
        if self.reduce:
            ndblock = ndblock.reduce(force_numpy=False)
        ndblock = ndblock.sum(keepdims=True)
        return ndblock


# --------------------------Convert Ordinal Mask to Cell Count Estimate------------------------


class CountOSBySize(SegProcess):
    """Counting ordinal segmentation contours

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

    def cc_list(self, os: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        contours_np3d = algorithms.npindices_from_os(os)
        ncells = {}
        dc = []
        dc_idx_to_centroid_idx = {}
        idx_max = np.array(tuple(d - 1 for d in os.shape), dtype=np.int64)
        for i in range(len(contours_np3d)):
            contour = contours_np3d[i]
            nvoxel = contour.shape[0]
            if nvoxel <= self.min_size:
                ncells[i] = 0.
            else:
                ncells[i] = 0.

                # if no voxel touch the boundary, we do not want to apply the edge penalty
                on_edge = (contour == 0).astype(np.uint8) + (contour == idx_max[None, :]).astype(np.uint8)
                on_edge = on_edge.sum().item() > 0
                if on_edge:
                    dc_idx_to_centroid_idx[len(dc)] = i
                    dc.append(contour.astype(np.float32).mean(axis=0))
                else:
                    ncells[i] = 1

                if nvoxel > self.size_threshold:
                    ncells[i] += (nvoxel - self.size_threshold) * self.volume_weight
        ps = CountLCEdgePenalized(os.shape, self.border_params)

        if len(dc) == 0:
            dc_centroids = np.zeros((0, os.ndim), dtype=np.float32)
        else:
            dc_centroids = np.array(dc, dtype=np.float32)
        dc_ncells = ps.cc_list(dc_centroids, (0,) * os.ndim)
        for dc_idx in dc_idx_to_centroid_idx:
            i = dc_idx_to_centroid_idx[dc_idx]
            ncells[i] += dc_ncells[dc_idx]
        ncells = np.array([ncells[i] for i in range(len(ncells))], dtype=np.float32)
        return ncells

    def np_features(self,
                    os: npt.NDArray[np.int32],
                    block_info=None) -> npt.NDArray[np.float32]:
        if block_info is None:
            return np.zeros(tuple(), dtype=np.float32)

        # slices = block_info[0]['array-location']
        ps = DirectOSToLC(reduce=True)
        lc = ps.np_features(os, block_info)
        cc_list = self.cc_list(os)
        features = np.concatenate((lc, cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, im: npt.NDArray | da.Array) -> NDBlock[np.float32]:
        return NDBlock.map_ndblocks([NDBlock(im)], self.np_features, out_dtype=np.float32)

    def forward(self,
                im: npt.NDArray[np.int32] | da.Array,
                cid: str = None,
                viewer_args: dict = None
                ) -> npt.NDArray[np.float32]:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        cache_exists, cache_dir = self.tmpdir.cache(is_dir=True, cid=cid)
        ndblock = cache_dir.cache_im(fn=lambda: self.feature_forward(im), cid='os_by_size_features',
                                     cache_level=1)

        if viewer and viewer_args.get('display_points', True):
            features = ndblock.reduce(force_numpy=True)
            features = features[features[:, -1] > 0., :]
            lc_interpretable_napari('bysize_ncells',
                                    features, viewer, im.ndim, ['ncells'])

            aggregate_ndblock: NDBlock[np.float32] = cache_dir.cache_im(
                fn=lambda: map_ncell_vector_to_total(ndblock), cid='aggregate_ndblock',
                cache_level=2
            )
            aggregate_features: npt.NDArray[np.float32] = cache_dir.cache_im(
                fn=lambda: aggregate_ndblock.reduce(force_numpy=True), cid='block_cell_count',
                cache_level=2
            )
            lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                    im.ndim, ['ncells'], text_color='red')

            heatmap_cache_exists, heatmap_cache_dir = cache_dir.cache(is_dir=True, cid='cell_density_map')
            chunk_size = NDBlock(im).get_chunksize()
            heatmap_logging(aggregate_ndblock, heatmap_cache_exists, heatmap_cache_dir, viewer_args, chunk_size)

        ndblock = ndblock.select_columns([-1])

        if self.reduce:
            ndblock = ndblock.reduce(force_numpy=False)
        ndblock = ndblock.sum(keepdims=True)
        return ndblock
