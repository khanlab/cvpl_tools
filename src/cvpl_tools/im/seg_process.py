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
CC - Cell Count Map (3d, np.float64), a cell count number (estimate, can be float) for each block
CD - Cell Density Map (3d, np.float64), this is a down sampled map of the cell density of the brain
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
from typing import Callable, Any, Sequence

import cvpl_tools.im.fs as imfs
import cvpl_tools.im.algorithms as algorithms
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
                                          f'first dimension to be nextra+ndim={nextra + ndim} but got '
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
        pass

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


def block_to_block_forward(
        np_forward: Callable,
        im: npt.NDArray | da.Array | NDBlock,
        cptr: imfs.CachePointer,
        viewer_args=None,
        out_dtype: np.dtype = None, is_label: bool = False,
        compute_chunk_sizes: bool = False):
    if viewer_args is None:
        viewer_args = {}

    def compute():
        nonlocal out_dtype, compute_chunk_sizes
        if isinstance(im, np.ndarray):
            result = np_forward(im)
        elif isinstance(im, da.Array):
            assert im is not None
            if out_dtype is None:
                out_dtype = im.dtype
            result = im.map_blocks(
                np_forward,
                meta=np.array(tuple(), dtype=out_dtype),
                dtype=out_dtype
            )
            if compute_chunk_sizes:
                result.compute_chunk_sizes()
        elif isinstance(im, NDBlock):
            assert im is not None
            if out_dtype is None:
                out_dtype = im.get_dtype()
            result = NDBlock.map_ndblocks([im],
                                          np_forward,
                                          out_dtype=out_dtype,
                                          use_input_index_as_arrloc=0)
            if not result.is_numpy() and compute_chunk_sizes:
                result = result.arr
                result.compute_chunk_sizes()
                result = NDBlock(result)
        else:
            raise TypeError(f'Invalid im type: {type(im)}')
        return result

    patched = viewer_args
    if is_label is not None:
        patched |= dict(is_label=is_label)
    result = cptr.im(fn=compute,
                     cache_level=1,
                     viewer_args=patched)
    return result


class BlockToBlockProcess(SegProcess):
    def __init__(self, out_dtype: np.dtype = None, is_label=False, compute_chunk_sizes=False):
        super().__init__()
        self.out_dtype = out_dtype
        self.is_label = is_label
        self.compute_chunk_sizes = compute_chunk_sizes

    def set_out_dtype(self, out_dtype: np.dtype | None):
        self.out_dtype = out_dtype

    def set_is_label(self, is_label: bool | None):
        self.is_label = is_label

    def forward(self, im: npt.NDArray | da.Array | NDBlock, cptr: imfs.CachePointer, viewer_args=None) \
            -> npt.NDArray | da.Array | NDBlock:
        return block_to_block_forward(
            np_forward=self.np_forward,
            im=im,
            cptr=cptr,
            viewer_args=viewer_args,
            out_dtype=self.out_dtype,
            is_label=self.is_label,
            compute_chunk_sizes=self.compute_chunk_sizes
        )

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
                                          threshold=self.threshold).astype(np.float64)  # N * (ndim + 1) ndarray
            start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float64)
            lc[:, :block.ndim] += start_pos[None, :]
            return lc
        else:
            return block

    def feature_forward(self, im: npt.NDArray[np.float32] | da.Array) -> NDBlock[np.float64]:
        return NDBlock.map_ndblocks([NDBlock(im)], self.np_features, out_dtype=np.float64)

    def forward(self,
                im: npt.NDArray[np.float32] | da.Array,
                cptr: imfs.CachePointer,
                viewer_args: dict = None
                ) -> NDBlock:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)
        ndblock = cptr.im(lambda: self.feature_forward(im), cache_level=1)

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
            -> npt.NDArray[np.float64]:
        if block_info is not None:
            slices = block_info[0]['array-location']
            if spatial_box_width is not None:
                padded_block = algorithms.pad_to_multiple(block, spatial_box_width)
            else:
                padded_block = block
            masked = padded_block * (padded_block > self.min_thres)
            if slices is not None:
                startpoint = np.array([slices[i].start for i in range(len(slices))],
                                      dtype=np.float64)
            else:
                startpoint = np.zeros((block.ndim,), dtype=np.float64)

            if spatial_box_width is not None:
                subblock_shape = (spatial_box_width,) * block.ndim
                masked = algorithms.np_map_block(masked, block_sz=subblock_shape)
                masked: npt.NDArray = masked.sum(axis=tuple(range(block.ndim, block.ndim * 2)))

                features = np.zeros((masked.size, block.ndim + 1), dtype=np.float64)
                inds = np.array(np.indices(masked.shape, dtype=np.float64))
                transpose_axes = tuple(range(1, block.ndim + 1)) + (0,)
                inds = inds.transpose(transpose_axes).reshape(-1, block.ndim)
                features[:, -1] = masked.flatten() * self.scale
                features[:, :-1] = startpoint[None, :] + (inds + .5) * np.array(subblock_shape, dtype=np.float64)
            else:
                features = np.zeros((1, block.ndim + 1), dtype=np.float64)
                features[0, -1] = masked.sum() * self.scale
                features[:, :-1] = startpoint[None, :] + np.array(block.shape, dtype=np.float64) / 2
            return features
        else:
            return block

    def feature_forward(self, im: npt.NDArray[np.float32] | da.Array, spatial_block_width: int = None) \
            -> NDBlock[np.float64]:
        def map_fn(b, block_info):
            return self.np_features(b, block_info, spatial_block_width)

        return NDBlock.map_ndblocks([NDBlock(im)], map_fn, out_dtype=np.float64)

    def forward(self, im: npt.NDArray | da.Array, cptr: imfs.CachePointer, viewer_args: dict = None) \
            -> NDBlock | npt.NDArray:
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)

        cdir = cptr.subdir()
        forwarded = cdir.cache_im(
            fn=lambda: self.feature_forward(im),
            cid='forwarded',
            cache_level=2
        )

        if viewer:
            mask = cdir.cache_im(
                fn=lambda: im > self.min_thres,
                cid='ssi_mask',
                cache_level=2,
                viewer_args=viewer_args | dict(is_label=True))
            if self.spatial_box_width is not None and viewer_args.get('display_points', True):
                ssi = cdir.cache_im(
                    fn=lambda: self.feature_forward(im, spatial_block_width=self.spatial_box_width).reduce(
                        force_numpy=True),
                    cid='ssi',
                    cache_level=2
                )
                lc_interpretable_napari('ssi_block', ssi, viewer, im.ndim, ['ncells'])

            aggregate_ndblock: NDBlock[np.float64] = cdir.cache_im(
                fn=lambda: map_ncell_vector_to_total(forwarded),
                cid='aggregate_ndblock',
                cache_level=2
            )
            heatmap_cptr = cdir.cache(cid='cell_density_map')
            chunk_size = NDBlock(im).get_chunksize()
            heatmap_logging(aggregate_ndblock, heatmap_cptr, viewer_args, chunk_size)

        def fn():
            ndblock = forwarded.select_columns([-1])
            if self.reduce:
                ndblock = ndblock.reduce(force_numpy=False)
            return ndblock

        ssi_result = cdir.cache_im(fn=fn, cid='ssi_result', cache_level=1)

        return ssi_result


# --------------------Convert Binary and Centroid list to Instance Mask------------------------


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
                     lc: npt.NDArray[np.float64],
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
                lc: NDBlock[np.float64],
                cptr: imfs.CachePointer,
                viewer_args: dict = None
                ) -> npt.NDArray[np.int32] | da.Array:
        cdir = cptr.subdir()
        if viewer_args is None:
            viewer_args = {}
        viewer = viewer_args.get('viewer', None)

        if viewer:
            def compute_lbl():
                if isinstance(bs, np.ndarray):
                    lbl_im = instance_label(bs)[0]
                else:
                    lbl_im = bs.map_blocks(
                        lambda block: instance_label(block)[0].astype(np.int32), dtype=np.int32
                    )
                return lbl_im

            lbl_im = cdir.cache_im(fn=compute_lbl,
                                   cid='before_split',
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
                tmp_path = cdir.cache_subpath(cid='blocks').abs_path
                result = ndblock.as_dask_array(tmp_path)
            return result

        result = cdir.cache_im(fn=compute_result,
                               cid='result',
                               cache_level=1,
                               viewer_args=viewer_args | dict(is_label=True))
        return result


def map_ncell_vector_to_total(ndblock: NDBlock[np.float64]) -> NDBlock[np.float64]:
    """Aggregate the counts in ncell vector to get a single ncell estimate

    Args:
        ndblock: Each block contains a ncell vector

    Returns:
        The summed ncell by block; coordinates of each ncell is the center of the block
    """

    def map_fn(block: npt.NDArray[np.float64], block_info):
        slices = block_info[0]['array-location']
        midpoint = np.array(tuple((s.stop - s.start + 1) / 2 + s.start for s in slices), dtype=np.float64)
        ndim = midpoint.shape[0]
        result = np.zeros((1, ndim + 1), dtype=np.float64)
        result[0, :-1] = midpoint
        result[0, -1] = block[:, -1].sum()
        return result

    return NDBlock.map_ndblocks([ndblock], map_fn, out_dtype=np.float64)


def heatmap_logging(aggregate_ndblock: NDBlock[np.float64],
                    cptr: imfs.CachePointer,
                    viewer_args: dict,
                    chunk_size: tuple):
    cdir = cptr.subdir()
    np_arr_path = f'{cdir.abs_path}/density_map.npy'
    if not cdir.exists:
        block = aggregate_ndblock.select_columns([-1])
        ndim = block.get_ndim()

        def map_fn(block: npt.NDArray[np.float64], block_info=None):
            return block.reshape((1,) * ndim)

        block = block.map_ndblocks([block], map_fn, out_dtype=np.float64)
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
