"""
Segmentation and post-processing.

This file is for methods generating (dask) single class segmentation masks of binary or ordinal types, the latter of
which is a 0-N segmentation of N objects of the same class in an image.

Methods in this file should be run quickly and whose performance can be compared against each other to manual
segmentations over a dataset.

The input to these methods are either input 3d single-channel image of type np.float32, or input image paired with
a deep learning segmentation algorithm. The output may be cell count #, binary mask
(np.uint8) or ordinal mask (np.int32).

Conceptually, we define the following input/output types:
IN - Input Image (np.float32) between 0 and 1, this is the brightness dask image as input
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
- input_type() -> ty
- output_type() -> ty
- forward(*args) -> out, this is the forward stage
- interpretable_napari(viewer, *args) -> None, this adds the appropriate human-interpretable debugging-purpose outputs to viewer

About the dask version of the last 2 functions: These are for parallel processing of the images; they need not be
interpretable for large images but the interpretable_napari function should handle small sized dask images for
debugging purpose.
"""

import abc
import enum
import logging
from typing import Callable, Any, Sequence, final, Iterator

import cvpl_tools.fs as fs
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
import cvpl_tools.im.algorithms as algorithms
import cvpl_tools.im.dask_algorithms as dask_algorithms
import dask.array as da
import napari
import numpy as np
import skimage
from scipy.ndimage import (
    gaussian_filter as gaussian_filter,
    label as instance_label,
    find_objects as find_objects
)


# ------------------------------------Helper Functions---------------------------------------


def direct_map_block(forward: Callable, im: da.Array):
    def map_fn(block, block_info=None):
        return forward(block)

    return im.map_blocks(map_fn)


def lc_interpretable_napari(layer_name: str,
                            lc: np.ndarray,
                            viewer: napari.Viewer,
                            ndim: int,
                            extra_features: Sequence):
    """This function is used to display feature points for LC-typed output

    Args:
        layer_name: displayed name of the layer
        lc: The list of features, each row of length (ndim + nextra)
        viewer: Napari viewer to add points to
        ndim: dimension of the image
        extra_features: extra features to be displayed as text
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
        'color': 'green',
        'anchor': 'center',
    }
    viewer.add_points(lc[:, :ndim],
                      size=1.,
                      ndim=ndim,
                      name=layer_name,
                      features=features,
                      text=text_parameters)


def select_feature_columns(features,
                           cols: slice | Sequence[int] | int,
                           reduced: bool) -> Iterator[tuple] | np.ndarray:
    """Performs selection on result returned by map_da_to_rows()"""
    if reduced:
        return features[:, cols]
    else:
        return ((block[:, cols], block_idx, slices) for (block, block_idx, slices) in features)


def sum_feature_columns(features,
                        reduced: bool) -> Iterator[tuple] | np.ndarray:
    """Performs selection on result returned by map_da_to_rows()"""
    if reduced:
        return features.sum(keepdims=True)
    else:
        return ((block.sum(keepdims=True), block_idx, slices)
                for (block, block_idx, slices) in features)


# ---------------------------------------Interfaces------------------------------------------


logger = logging.getLogger('SEG_PROCESSES')


class DatType(enum.Enum):
    IN = 0
    BS = 1
    OS = 2
    LC = 3
    CC = 4
    CD = 5
    ATLAS_MAP = 6
    ATLAS_CD = 7
    OTHER = 8  # tuple of two or more of the above, or etc.


class SegProcess(abc.ABC):
    def __init__(self, input_type: DatType, output_type: DatType):
        self._input_type = input_type
        self._output_type = output_type
        self.tmpdir: fs.TmpDirectory | None = None

    def input_type(self) -> DatType:
        return self._input_type

    def output_type(self) -> DatType:
        return self._output_type

    def input_type_str(self) -> str:
        return self._input_type.name

    def output_type_str(self) -> str:
        return self._output_type.name

    def set_tmpdir(self, tmpdir: fs.TmpDirectory):
        """Use this to cache output results

        For output images that involves interpretation with napari, this is necessary for a smooth display of the
        results.

        Args:
            tmpdir: Any fs.TmpDirectory object that allows many temporary objects to be written as cache in the
                directory
        """
        assert isinstance(tmpdir, fs.TmpDirectory), f'Expected type cvpl_tools.fs.TempDirectory, got {type(tmpdir)}'
        self.tmpdir = tmpdir

    def cache_im(self, im: np.ndarray | da.Array) -> np.ndarray | da.Array:
        if isinstance(im, da.Array):
            assert self.tmpdir is not None, 'Please assign a tmp_prefix before caching Napari image!'
            tmp_prefix = self.tmpdir.assign_tmpdir()
            im = cvpl_ome_zarr_io.cache_image(im, tmp_prefix)
        return im

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("SegProcess base class does not implement forward()!")

    @abc.abstractmethod
    def interpretable_napari(self, *args, **kwargs):
        """By default, we directly try to add the output to Napari

        This is the simplest way to interpret the output, but will not work for all data types. Subclasses should
        disregard this method and write their own if any more complicated interpretation is needed.

        Args:
            viewer: Napari viewer object
            *args: args to be passed to forward()
            **kwargs: keyword args to be passed to forward()
        """
        raise NotImplementedError("SegProcess base class does not implement forward()!")


class BlockToBlockProcess(SegProcess):
    @final
    def forward(self, im: np.ndarray | da.Array) \
            -> np.ndarray | da.Array:
        if isinstance(im, np.ndarray):
            return self.np_forward(im)
        elif isinstance(im, da.Array):
            return self.dask_forward(im)
        else:
            raise TypeError(f'Invalid im type: {type(im)}')

    @abc.abstractmethod
    def np_forward(self, im: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Subclass of BlockToBlockProcess should implement np_forward!')

    def dask_forward(self, im: da.Array) -> da.Array:
        return direct_map_block(self.np_forward, im)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray | da.Array):
        viewer.add_image(self.cache_im(self.forward(im)), name='interpretation')


def forward_sequential_processes(processes: Sequence[SegProcess], arg: Any):
    """Run a sequence of processes as one

    Args:
        processes: The list of SegProcess objects to be chained to one
        arg: The argument to the first SegProcess object

    Returns:
        The result returned after a chain of operations
    """
    cur_out = arg
    for i in range(len(processes)):
        cur_out = processes[i].forward(cur_out)
    return cur_out


def interpretable_napari_sequential_processes(viewer: napari.Viewer,
                                              processes: Sequence[SegProcess],
                                              arg: Any,
                                              add_flags: Sequence[bool] | None = None):
    """Run a sequence of processes, and interpret these by adding them into napari as images

    Args:
        viewer: Napari viewer
        processes: The processes to be interpreted
        arg: The argument to the first process
        add_flags: default to all True; a list of flags indicating which processes in the list should be interpreted
    """
    if add_flags is None:
        add_flags = [True] * len(processes)
    last_true = -1
    for i in range(len(processes)):
        if add_flags[i]:
            last_true = i
    if last_true == -1:
        return

    cur_out = arg
    for i in range(last_true + 1):
        if i != last_true:
            cur_out = processes[i].forward(cur_out)
        processes[i].interpretable_napari(cur_out, viewer=viewer)


# ---------------------------------------Preprocess------------------------------------------


class GaussianBlur(BlockToBlockProcess):
    def __init__(self, sigma: float):
        super().__init__(DatType.IN, DatType.IN)
        self.sigma = sigma

    def np_forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return gaussian_filter(im, sigma=self.sigma)


# -------------------------------------Predict Binary----------------------------------------


class BSPredictor(BlockToBlockProcess):
    def __init__(self, pred_fn: Callable):
        super().__init__(DatType.IN, DatType.BS)
        self.pred_fn = pred_fn

    def np_forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        return self.pred_fn(im)


class SimpleThreshold(BlockToBlockProcess):
    def __init__(self, threshold: float):
        super().__init__(DatType.IN, DatType.BS)
        self.threshold = threshold

    def np_forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        return (im > self.threshold).astype(np.uint8)


# --------------------------------Predict List of Centroids-----------------------------------


class BlobDog(SegProcess):
    def __init__(self, max_sigma=2, threshold: float = 0.1, reduce=False):
        super().__init__(DatType.IN, DatType.LC)
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.reduce = reduce

    def np_features(self, block: np.ndarray, block_idx: tuple | None = None, slices: tuple | None = None):
        lc = skimage.feature.blob_dog(np.array(block * 255, dtype=np.uint8),
                                      max_sigma=self.max_sigma,
                                      threshold=self.threshold).astype(np.float32)  # N * (ndim + 1) ndarray
        if slices is not None:
            start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float32)
            lc[:, :block.ndim] += start_pos[None, :]
        return lc

    def feature_forward(self, im: np.ndarray | da.Array, reduce: bool) \
            -> Iterator[tuple] | np.ndarray:
        return dask_algorithms.map_da_to_rows(im=im,
                                              fn=self.np_features,
                                              return_dim=im.ndim,
                                              return_dtype=np.float32,
                                              return_dask=False,
                                              reduce=reduce)

    def forward(self, im: np.ndarray | da.Array) \
            -> Iterator[tuple] | np.ndarray:
        return select_feature_columns(self.feature_forward(im, reduce=self.reduce), slice(im.ndim), self.reduce)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        blobdog = self.feature_forward(im, reduce=True)
        lc_interpretable_napari('blobdog_centroids', blobdog, viewer, im.ndim, ['sigma'])


# -------------------------------Direct Cell Count Prediction----------------------------------


class ScaledSumIntensity(SegProcess):
    def __init__(self, scale: float = .008, min_thres: float = 0., reduce: bool = True, spatial_box_width: int = 8):
        """Summing up the intensity and scale it to obtain number of cells, directly

        Args:
            scale: Scale the sum of intensity by this to get number of cells
            min_thres: Intensity below this threshold is excluded (set to 0 before summing)
        """
        assert scale >= 0, f'{scale}'
        assert 0. <= min_thres <= 1., f'{min_thres}'
        super().__init__(DatType.IN, DatType.CC)
        self.scale = scale
        self.min_thres = min_thres
        self.reduce = reduce
        self.spatial_box_width = spatial_box_width

    def np_features(self, block: np.ndarray[np.float32],
                    block_index: tuple | None,
                    slices: tuple | None,
                    spatial_box_width: int = None) \
            -> np.ndarray[np.float32]:
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
            masked: np.ndarray = masked.sum(axis=tuple(range(block.ndim, block.ndim * 2)))

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

    def feature_forward(self, im: np.ndarray[np.float32] | da.Array, reduce: bool, spatial_block_width: int = None) \
            -> Iterator[tuple] | np.ndarray[np.float32]:
        def map_fn(b, bidx, slices):
            return self.np_features(b, bidx, slices, spatial_block_width)
        return dask_algorithms.map_da_to_rows(im=im,
                                              fn=map_fn,
                                              return_dim=im.ndim + 1,
                                              return_dtype=np.float32,
                                              return_dask=False,
                                              reduce=reduce)

    def forward(self, im: np.ndarray | da.Array) \
            -> Iterator[tuple] | np.ndarray:
        return select_feature_columns(self.feature_forward(im, reduce=self.reduce), [-1], self.reduce)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32] | da.Array):
        # see what has been completely masked off
        mask = im > self.min_thres
        viewer.add_image(self.cache_im(mask), name='spatial_vis_ncell_sum_intensity')

        # see for each spatial block, how many cells are counted within that block
        ssi = self.feature_forward(im, reduce=True, spatial_block_width=self.spatial_box_width)

        lc_interpretable_napari('block_scaled_sum_intensity', ssi, viewer, im.ndim, ['ncells'])


# ---------------------------Convert Binary Mask to Instance Mask------------------------------


class DirectBSToOS(BlockToBlockProcess):
    def __init__(self):
        super().__init__(DatType.BS, DatType.OS)

    def np_forward(self, bs: np.ndarray[np.uint8]) -> np.ndarray[np.int32]:
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
        super().__init__(DatType.BS, DatType.OS)
        self.size_thres = size_thres
        self.dist_thres = dist_thres
        self.rst = rst
        self.size_thres2 = size_thres2
        self.dist_thres2 = dist_thres2
        self.rst2 = rst2

    def np_forward(self, bs: np.ndarray[np.uint8]) -> np.ndarray[np.int32]:
        lbl_im = algorithms.round_object_detection_3sizes(bs,
                                                          size_thres=self.size_thres,
                                                          dist_thres=self.dist_thres,
                                                          rst=self.rst,
                                                          size_thres2=self.size_thres2,
                                                          dist_thres2=self.dist_thres2,
                                                          rst2=self.rst2)
        return lbl_im
    # TODO: better visualization of this stage


# ---------------------------Convert Binary Mask to Instance Mask------------------------------


class BinaryAndCentroidListToInstance(SegProcess):
    def __init__(self):
        super().__init__(DatType.OTHER, DatType.OS)

    def bacl_forward(self, bs: np.ndarray[np.uint8], lc: np.ndarray[np.float32]) -> np.ndarray[np.int32]:
        assert isinstance(bs, np.ndarray) and isinstance(lc, np.ndarray), \
            f'Error: inputs must be numpy for the forward() of this class, got bs={type(bs)} and lc={type(lc)}'

        # first sort each centroid into contour they belong to
        lc = lc.astype(np.int64)
        lbl_im, max_lbl = instance_label(bs)[0]

        # index 0 is background - centroids fall within this are discarded
        contour_centroids = [[] for _ in range(max_lbl + 1)]

        for centroid in lc:
            c_ord = int(lbl_im[tuple(centroid)])
            contour_centroids[c_ord].append(centroid)

        def map_fn(centroids: list[np.ndarray[np.int64]], indices: list[int], X: tuple[np.ndarray]):
            N = len(centroids)
            assert N >= 2
            assert N == len(indices)
            arr_shape = X[0].shape
            X = np.array(X)
            indices = np.array(indices, dtype=np.int32)

            minD = np.inf  # should be an array with same size as the image, but for simplicity we set this as scalar
            idxD = np.zeros(arr_shape, dtype=np.int32)
            for i in range(N):
                centroid = centroids[i]
                D = X - np.expand_dims(centroid, list(range(1, X.ndim)))
                D = np.sqrt((D * D).astype(np.float32))  # euclidean distance
                new_mask = D < minD
                idxD = new_mask * indices[i] + ~new_mask * idxD
                minD = new_mask * D + ~new_mask * minD
            return idxD

        # now we compute the contours, and brute-force calculate what centroid each pixel is closest to
        object_slices = list(find_objects(lbl_im))
        new_lbl = max_lbl + 1
        for i in range(1, max_lbl + 1):
            slices = object_slices[i]
            if slices is None:
                continue

            # if there are 0 or 1 centroid in the contour, we do nothing
            centroids = contour_centroids[i]  # centroids fall within the current contour
            ncentroid = len(centroids)
            if ncentroid <= 1:
                continue

            # otherwise, divide the contour and map pixels to each
            indices = [i] + [lbl for lbl in range(new_lbl, new_lbl + ncentroid - 1)]
            new_lbl += ncentroid - 1
            mask = lbl_im[slices] == i + 1
            divided = algorithms.coord_map(mask.shape, lambda *X: map_fn(centroids, indices, X))

            lbl_im[slices] = lbl_im[slices] * ~mask + divided
        return lbl_im

    def forward(self,
                bs: np.ndarray[np.uint8] | da.Array,
                lc: np.ndarray[np.float32] | Iterator[tuple]) -> np.ndarray[np.int32] | da.Array:
        lc_map = {
            block_idx: block for (block, block_idx, slices) in lc
        }
        if isinstance(bs, np.ndarray):
            return self.bacl_forward(bs, lc)
        else:
            assert isinstance(bs, da.Array)

            def map_fn(block, block_idx, slices):
                block_lc = lc_map[block_idx]
                block_lc -= np.array([slices[i].start for i in range(len(slices))], dtype=np.float32)
                return self.bacl_forward(block, block_lc)

            return bs.map_blocks(map_fn)

    def interpretable_napari(self,
                             viewer: napari.Viewer,
                             bs: np.ndarray[np.uint8] | da.Array,
                             lc: np.ndarray[np.float32] | Iterator[tuple]):
        viewer.add_image(self.cache_im(self.forward(bs, lc)), name='Inst seg by pixel distance to centroid')


# ---------------------------Ordinal Segmentation to List of Centroids-------------------------


class DirectOSToLC(SegProcess):
    """Convert a 0-N contour mask to a list of N centroids, one for each contour

    The converted list of centroids is in the same order as the original contour order (The contour labeled
    1 will come first and before contour labeled 2, 3 and so on)
    """

    def __init__(self, reduce=False):
        super().__init__(DatType.OS, DatType.LC)
        self.reduce = reduce

    def np_features(self, block: np.ndarray, block_idx: tuple | None = None, slices: tuple | None = None):
        contours_np3d = algorithms.npindices_from_os(block)
        lc = [contour.astype(np.float32).mean(axis=0) for contour in contours_np3d]
        lc = np.array(lc, dtype=np.float32)
        if slices is not None:
            start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float32)
            lc[:, :block.ndim] += start_pos[None, :]
        return lc

    def feature_forward(self, im: np.ndarray | da.Array, reduce: bool) \
            -> list[np.ndarray] | np.ndarray:
        return dask_algorithms.map_da_to_rows(im=im,
                                              fn=self.np_features,
                                              return_dim=im.ndim,
                                              return_dtype=np.float32,
                                              return_dask=False,
                                              reduce=reduce)

    def forward(self, im: np.ndarray | da.Array) \
            -> np.ndarray:
        return self.feature_forward(im, reduce=self.reduce)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        features = self.feature_forward(im, reduce=True)
        lc_interpretable_napari('os_to_lc_centroids', features, viewer, im.ndim, [])


# -----------------------Convert List of Centroids to Cell Count Estimate----------------------


class CountLCEdgePenalized(SegProcess):
    """From a list of cell centroid locations, calculate a cell count estimate

    Each centroid is simply treated as 1 cell when they are sufficiently far from the edge, but as they
    get closer to the edge the divisor becomes >1. and their estimate decreases towards 0, since cells
    near the edge may be double-counted (or triple or more counted if at a corner etc.)
    """

    def __init__(self,
                 im_shape: np.ndarray | tuple[int],
                 border_params: tuple[float, float, float] = (3., -.5, 2.),
                 reduce: bool = False):
        super().__init__(DatType.LC, DatType.CC)
        self.im_shape = np.array(im_shape, dtype=np.float32)
        self.border_params = border_params
        self.reduce = reduce

        intercept, dist_coeff, div_max = border_params
        assert intercept >= 1., f'intercept has to be >= 1. as divisor must be >= 1! (intercept={intercept})'
        assert dist_coeff <= 0., (f'The dist_coeff needs to be non-positive so divisor decreases as cell is further '
                                  f'from the edge')
        assert div_max >= 1., f'The divisor is >= 1, but got div_max < 1! (div_max={div_max})'

    def cc_list(self, lc: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        """Returns a cell count estimate for each contour in the list of centroids

        Args:
            lc: The list of centroids to be given cell estimates for

        Returns:
            A 1-d list, each element is a scalar cell count for the corresponding contour centroid in lc
        """
        midpoint = (self.im_shape * .5)[None, :]

        # compute border distances in each axis direction
        border_dists = np.abs((lc + midpoint) % self.im_shape - (midpoint - .5))

        intercept, dist_coeff, div_max = self.border_params
        mults = 1 / np.clip(intercept - border_dists * dist_coeff, 1., div_max)
        cc_list = np.prod(mults, axis=1)
        return cc_list

    def np_features(self, lc: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        cc_list = self.cc_list(lc)
        features = np.concat((lc, cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, im: np.ndarray | da.Array, reduce: bool) \
            -> list[np.ndarray] | np.ndarray:
        return dask_algorithms.map_da_to_rows(im=im,
                                              fn=self.np_features,
                                              return_dim=im.ndim + 1,
                                              return_dtype=np.float32,
                                              return_dask=False,
                                              reduce=reduce)

    def forward(self, im: np.ndarray | da.Array) \
            -> np.ndarray:
        features = select_feature_columns(self.feature_forward(im, self.reduce), [-1], self.reduce)
        return sum_feature_columns(features, self.reduce)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        features = self.feature_forward(im, reduce=True)
        lc_interpretable_napari('edge_penalized_centroids', features, viewer, im.ndim, ['ncells'])


# --------------------------Convert Ordinal Mask to Cell Count Estimate------------------------


class CountOSNContour(SegProcess):
    """Counting ordinal segmentation contours

    Several features:
    1. A size threshold, below which each contour is counted as a single cell (or part of a single cell,
        in the case it is neighbor to boundary of the image)
    2. Above size threshold, the contour is seen as a cluster of cells an estimate of cell count is given
        based on the volume of the contour
    3. For cells on the boundary location, their estimated ncell is penalized according to the distance
        between the cell centroid and the boundary of the image
    4. A min_size threshold, below (<=) which the contour is simply discarded because it's likely just
        an artifact
    """

    def __init__(self,
                 size_threshold: int | float = 25.,
                 volume_weight: float = 6e-3,
                 border_params: tuple[float, float, float] = (3., -.5, 2.),
                 min_size: int | float = 0,
                 reduce: bool = False):
        super().__init__(DatType.OS, DatType.CC)
        self.size_threshold = size_threshold
        self.volume_weight = volume_weight
        self.border_params = border_params
        self.min_size = min_size
        self.reduce = reduce

    def cc_list(self, os: np.ndarray[np.int32]) -> np.ndarray[np.float32]:
        contours_np3d = algorithms.npindices_from_os(os)
        ncells = {}
        dc = []
        dc_idx_to_centroid_idx = {}
        for i in range(len(contours_np3d)):
            contour = contours_np3d[i]
            nvoxel = contour.shape[0]
            if nvoxel <= self.min_size:
                ncells[i] = 0.
            elif nvoxel <= self.size_threshold:
                dc_idx_to_centroid_idx[len(dc)] = i
                dc.append(contour.astype(np.float32).mean(axis=0))
            else:
                ncells[i] = nvoxel * self.volume_weight
        ps = CountLCEdgePenalized(os.shape, self.border_params)
        dc_centroids = np.array(dc, dtype=np.float32)
        dc_ncells = ps.cc_list(dc_centroids)
        for dc_idx in dc_idx_to_centroid_idx:
            i = dc_idx_to_centroid_idx[dc_idx]
            ncells[i] = dc_ncells[dc_idx]
        ncells = np.array([ncells[i] for i in range(len(ncells))], dtype=np.float32)
        return ncells

    def np_features(self, os: np.ndarray[np.int32]) -> np.ndarray[np.float32]:
        ps = DirectOSToLC()
        lc = ps.forward(os)
        cc_list = self.cc_list(os)
        features = np.concat((lc, cc_list[:, None]), axis=1)
        return features

    def feature_forward(self, im: np.ndarray | da.Array, reduce: bool) \
            -> list[np.ndarray] | np.ndarray:
        return dask_algorithms.map_da_to_rows(im=im,
                                              fn=self.np_features,
                                              return_dim=im.ndim + 1,
                                              return_dtype=np.float32,
                                              return_dask=False,
                                              reduce=reduce)

    def forward(self, im: np.ndarray | da.Array) \
            -> np.ndarray:
        features = select_feature_columns(self.feature_forward(im, self.reduce), [-1], self.reduce)
        return sum_feature_columns(features, self.reduce)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        features = self.feature_forward(im, reduce=True)
        lc_interpretable_napari('edge_penalized_centroids', features, viewer, im.ndim, ['ncells'])
