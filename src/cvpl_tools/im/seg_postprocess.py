"""
Segmentation and post-processing.

This file is for methods generating (dask) single class segmentation masks of binary or ordinal types, the latter of
which is a 0-N segmentation of N objects of the same class in an image.

Methods in this file should be run quickly and whose performance can be compared against each other to manual
segmentations over a dataset.

The input to these methods are either input 3d single-channel image of type np.float32, or input image paired with
a deep learning segmentation algorithm. The output may be cell count #, binary mask
(np.uint8) or ordinal mask (np.uint16).

Conceptually, we define the following input/output types:
IN - Input Image (np.float32) between 0 and 1, this is the brightness dask image as input
BS - Binary Segmentation (3d, np.uint8), this is the binary mask single class segmentation
OS - Ordinal Segmentation (3d, np.uint16), this is the 0-N where contour 1-N each denotes an object; also single class
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

count_from_os: OS -> CC
e.g. count_os_ncontour, count_edge_penalized_os_ncontour, count_os_byvolume

count_from_lc: LC -> CC
e.g. count_lc_ncentroid, count_edge_penalized_lc_ncentroid

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
- add_to_viewer(viewer, *args) -> None, this adds the appropriate human-interpretable debugging-purpose outputs to viewer
"""

import enum
import abc
from typing import Callable, Any, Sequence

import napari
import numpy as np
from scipy.ndimage import gaussian_filter
import skimage
import algorithms


# ---------------------------------------Interfaces------------------------------------------


class DatType(enum.Enum):
    IN = 0
    BS = 1
    OS = 2
    LC = 3
    CC = 4
    CD = 5
    ATLAS_MAP = 6
    ATLAS_CD = 7


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

    def add_to_viewer(self, *args, **kwargs):
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


def add_to_viewer_sequential_processes(viewer: napari.Viewer,
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
        processes[i].add_to_viewer(cur_out, viewer=viewer)


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
        blobs_dog = skimage.feature.blob_dog(np.array(im * 255, dtype=np.uint8),
                                             max_sigma=self.max_sigma,
                                             threshold=self.threshold)  # N * 3 ndarray
        return blobs_dog

    def add_to_viewer(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        lc = self.forward(im)
        viewer.add_points(lc, ndim=3, name='blobdog_centroids')


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

    def add_to_viewer(self, viewer: napari.Viewer, im: np.ndarray[np.float32]):
        # see what has been completely masked off
        mask = im > self.min_thres
        viewer.add_image(mask, name='spatial_vis_ncell_sum_intensity')

        # see for each spatial block, how many cells are counted within that block
        block_masked_im = algorithms.np_map_block(im * mask, (self.spatial_box_width, ) * im.ndim)
        block_ncells = block_masked_im.sum(axis=list(range(im.ndim, im.ndim * 2))) * self.scale
        coords = (np.array(np.indices(block_ncells), dtype=np.float32) + .5) * self.spatial_box_width
        coords = coords.transpose(list(range(1, im.ndim)) + [0]).reshape(-1, im.ndim)  # now N * im.ndim coordinates
        ncells = block_ncells.flatten()

        # reference: https://napari.org/stable/gallery/add_points_with_features.html
        features = {
            'ncells': ncells
        }

        text_parameters = {
            'string': 'ncell: {ncells:.2f}',
            'size': 12,
            'color': 'green',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }
        viewer.add_points(coords,
                          ndim=im.ndim,
                          features=features,
                          text=text_parameters)


# ---------------------------Convert Binary Mask to Instance Mask------------------------------




