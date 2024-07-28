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

count_from_lc: LC -> CC
e.g. count_lc_ncentroid, count_edge_penalized_lc_ncentroid

count_from_os: OS -> CC
e.g. count_os_ncontour, count_edge_penalized_os_ncontour, count_os_byvolume

os_to_lc: OS -> LC
e.g. direct_os_to_centroids

count_to_cd: CC -> CD
e.g. count_to_density

cd_to_atlas: (ATLAS_MAP, CD) -> ATLAS_CD
e.g. average_by_region

Each method is an object implementing the SegStage interface that has the following methods:
- input_type() -> ty
- output_type() -> ty
- forward(*args) -> out, this is the forward stage
- interpretable_napari(viewer, *args) -> None, this adds the appropriate human-interpretable debugging-purpose outputs to viewer
"""

import enum
import abc
from typing import Callable, Any, Sequence
import logging

import napari
import numpy as np
from scipy.ndimage import (
    gaussian_filter as gaussian_filter,
    label as instance_label,
    find_objects as find_objects
)
import skimage
import cvpl_tools.im.algorithms as algorithms


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

    def input_type(self) -> DatType:
        return self._input_type

    def output_type(self) -> DatType:
        return self._output_type

    def input_type_str(self) -> str:
        return self._input_type.name

    def output_type_str(self) -> str:
        return self._output_type.name

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("SegProcess base class does not implement forward()!")

    def interpretable_napari(self, *args, **kwargs):
        """By default, we directly try to add the output to Napari

        This is the simplest way to interpret the output, but will not work for all data types. Subclasses should
        disregard this method and write their own if any more complicated interpretation is needed.

        Args:
            viewer: Napari viewer object
            *args: args to be passed to forward()
            **kwargs: keyword args to be passed to forward()
        """
        kwargs['viewer'].add_image(self.forward(*args, **kwargs), name='interpretation')


def forward_sequential_processes(processes: Sequence[SegProcess], arg: Any):
    cur_out = arg
    for i in range(len(processes)):
        cur_out = processes[i].forward(cur_out)
    return cur_out


def interpretable_napari_sequential_processes(viewer: napari.Viewer,
                                       processes: Sequence[SegProcess],
                                       arg: Any,
                                       add_flags: Sequence[bool] | None = None):
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


class GaussianBlur(SegProcess):
    def __init__(self, sigma: float):
        super().__init__(DatType.IN, DatType.IN)
        self.sigma = sigma

    def forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return gaussian_filter(im, sigma=self.sigma)


# -------------------------------------Predict Binary----------------------------------------


class BSPredictor(SegProcess):
    def __init__(self, pred_fn: Callable):
        super().__init__(DatType.IN, DatType.BS)
        self.pred_fn = pred_fn

    def forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        return self.pred_fn(im)


class SimpleThreshold(SegProcess):
    def __init__(self, threshold: float):
        super().__init__(DatType.IN, DatType.BS)
        self.threshold = threshold

    def forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        return (im > self.threshold).astype(np.uint8)


# --------------------------------Predict List of Centroids-----------------------------------


class BlobDog(SegProcess):
    def __init__(self, max_sigma=2, threshold: float = 0.1):
        super().__init__(DatType.IN, DatType.LC)
        self.max_sigma = max_sigma
        self.threshold = threshold

    def forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        blob_dog = skimage.feature.blob_dog(np.array(im * 255, dtype=np.uint8),
                                             max_sigma=self.max_sigma,
                                             threshold=self.threshold)  # N * 3 ndarray
        return blob_dog[:, :3].astype(np.float32)

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        lc = self.forward(im)
        features = {
            'sigma': lc[:, im.ndim]
        }

        text_parameters = {
            'string': 'sigma=\n{sigma:.2f}',
            'size': 9,
            'color': 'green',
            'anchor': 'center',
        }
        viewer.add_points(lc[:, :im.ndim],
                          size=1.,
                          ndim=im.ndim,
                          name='blobdog_centroids',
                          features=features,
                          text=text_parameters)


# -------------------------------Direct Cell Count Prediction----------------------------------


class ScaledSumIntensity(SegProcess):
    def __init__(self, scale: float = .008, min_thres: float = 0., spatial_box_width: int = 8):
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
        self.spatial_box_width = spatial_box_width

    def forward(self, im: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        mask = im > self.min_thres
        ncells = (im * mask).sum() * self.scale
        return ncells

    def interpretable_napari(self, viewer: napari.Viewer, im: np.ndarray[np.float32], *args, **kwargs):
        # see what has been completely masked off
        mask = im > self.min_thres
        viewer.add_image(mask, name='spatial_vis_ncell_sum_intensity')

        # see for each spatial block, how many cells are counted within that block
        block_masked_im = algorithms.np_map_block(im * mask, (self.spatial_box_width, ) * im.ndim)
        summed_axes = tuple(range(im.ndim, im.ndim * 2))

        block_ncells = block_masked_im.sum(axis=summed_axes) * self.scale
        coords = np.array(np.indices(block_ncells.shape), dtype=np.float32)
        coords = (coords + .5) * self.spatial_box_width - .5
        coords = coords.transpose(tuple(range(1, im.ndim + 1)) + (0, ))
        coords = coords.reshape((-1, im.ndim))  # now N * im.ndim coordinates
        ncells = block_ncells.flatten()

        # reference: https://napari.org/stable/gallery/add_points_with_features.html
        features = {
            'ncells': ncells
        }

        text_parameters = {
            'string': 'Ncell=\n{ncells:.2f}',
            'size': 9,
            'color': 'green',
            'anchor': 'center',
            # 'translation': [-3, 0],
        }
        viewer.add_points(coords,
                          ndim=im.ndim,
                          size=1.,
                          features=features,
                          text=text_parameters)


# ---------------------------Convert Binary Mask to Instance Mask------------------------------


class DirectBSToOS(SegProcess):
    def __init__(self):
        super().__init__(DatType.BS, DatType.OS)

    def forward(self, bs: np.ndarray[np.uint8]) -> np.ndarray[np.int32]:
        lbl_im, nlbl = instance_label(bs)
        return lbl_im


class Watershed3SizesBSToOS(SegProcess):
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

    def forward(self, bs: np.ndarray[np.uint8]) -> np.ndarray[np.int32]:
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

    def forward(self, bs: np.ndarray[np.uint8], lc: np.ndarray[np.float32]) -> np.ndarray[np.int32]:
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
        object_slices: list = find_objects(lbl_im)
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


# -----------------------Convert List of Centroids to Cell Count Estimate----------------------


class CountLCEdgePenalized(SegProcess):
    """From a list of cell centroid locations, calculate a cell count estimate

    Each centroid is simply treated as 1 cell when they are sufficiently far from the edge, but as they
    get closer to the edge the divisor becomes >1. and their estimate decreases towards 0, since cells
    near the edge may be double-counted (or triple or more counted if at a corner etc.)
    """

    def __init__(self,
                 im_shape: np.ndarray | tuple[int],
                 border_params: tuple[float, float, float] = (3., -.5, 2.)):
        super().__init__(DatType.LC, DatType.CC)
        self.im_shape = np.array(im_shape, dtype=np.float32)
        self.border_params = border_params

        intercept, dist_coeff, div_max = border_params
        assert intercept >= 1., f'intercept has to be >= 1. as divisor must be >= 1! (intercept={intercept})'
        assert dist_coeff <= 0., (f'The dist_coeff needs to be non-positive so divisor decreases as cell is further '
                                  f'from the edge')
        assert div_max >= 1., f'The divisor is >= 1, but got div_max < 1! (div_max={div_max})'

    def forward(self, lc: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        midpoint = (self.im_shape * .5)[None, :]

        # compute border distances in each axis direction
        border_dists = np.abs((lc + midpoint) % self.im_shape - (midpoint - .5))

        intercept, dist_coeff, div_max = self.border_params
        mults = 1 / np.clip(intercept - border_dists * dist_coeff, 1., div_max)
        count = np.prod(mults, axis=1).sum()
        return count


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
                 min_size: int | float = 0):
        super().__init__(DatType.OS, DatType.CC)
        self.size_threshold = size_threshold
        self.volume_weight = volume_weight
        self.border_params = border_params
        self.min_size = min_size

    def forward(self, os: np.ndarray[np.int32]) -> np.ndarray[np.float32]:
        contours_np3d = algorithms.find_np3d_from_os(os)
        ncell = np.array(0., dtype=np.float32)
        centroids = []
        for contour_np3d in contours_np3d:
            nvoxel = contour_np3d.shape[0]
            if nvoxel <= self.min_size:
                continue
            elif nvoxel <= self.size_threshold:
                centroids.append(contour_np3d.astype(np.float32).mean(axis=0))
            else:
                ncell += nvoxel * self.volume_weight
        ps = CountLCEdgePenalized(os.shape, self.border_params)
        centroids = np.array(centroids, dtype=np.float32)
        ncell += ps.forward(centroids)
        return ncell
