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
import copy
import logging
from typing import Callable, Any, Sequence

import cvpl_tools.tools.fs as tlfs
from cvpl_tools.fsspec import ensure_rdir_filesystem, RDirFileSystem
import cvpl_tools.im.algorithms as algorithms
from cvpl_tools.im.ndblock import NDBlock
import dask.array as da
from dask.distributed import print as dprint
import numpy as np
import numpy.typing as npt

try:
    from napari import Viewer
except OSError:
    Viewer = None  # if no graphical display is needed and napari backend is missing, import will fail

import skimage
from scipy.ndimage import (
    label as instance_label,
    find_objects as find_objects
)


# ------------------------------------Helper Functions---------------------------------------


def lc_interpretable_napari(layer_name: str,
                            lc: npt.NDArray,
                            viewer: Viewer,
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
    viewer.add_points(lc[:, :ndim],
                      size=1.,
                      ndim=ndim,
                      name=layer_name,
                      features=features,
                      text=text_parameters,
                      visible=False)


# ---------------------------------------Interfaces------------------------------------------


logger = logging.getLogger('SEG_PROCESSES')


async def block_to_block_forward(
        np_forward: Callable,
        im: npt.NDArray | da.Array | NDBlock,
        context_args: None | dict = None,
        out_dtype: np.dtype = None,
        compute_chunk_sizes: bool = False):
    """Call np_forward() on im, and optionally cache the result locally

    Args:
        np_forward: Chunk-wise process function
        im: The image to process
        context_args: dictionary of contextual arguments, see docstring for cvpl_tools.tools.fs.cache_im for more info
        out_dtype (np.dtype): Output data type
        compute_chunk_sizes (bool): If True, compute chunk sizes before caching the loaded image

    Returns:
        Returns the loaded image from the cached image
    """
    if context_args is None:
        context_args = {}

    async def compute():
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
            result = await NDBlock.map_ndblocks([im],
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

    result = await tlfs.cache_im(fn=compute, context_args=context_args)
    return result


# -------------------------------------Predict Binary----------------------------------------


async def in_to_bs_custom(pred_fn, im, context_args: dict = None):
    """Process the array and return an np.uint8 label array of the same size

    Args:
        pred_fn: A chunk-wise function that takes in a numpy image array and returns a label mask of the same size
            i.e. the input and output are npt.NDArray -> npt.NDArray[np.uint8]
        im: The array to predict on
        context_args: Dictionary of contextual arguments, see docstring of cvpl_tools.tools.fs.cache_im for more info

    Returns:
        Processed binary segmentation array
    """

    def np_forward(im: npt.NDArray[np.float32], block_info=None) -> npt.NDArray[np.uint8]:
        return pred_fn(im)

    if context_args is None:
        context_args = dict()
    else:
        context_args = copy.copy(context_args)
    context_args['viewer_args'] = context_args.get('viewer_args', {}) | dict(is_label=True)
    return await block_to_block_forward(
        np_forward=np_forward,
        im=im,
        context_args=context_args,
        out_dtype=np.uint8
    )


async def in_to_bs_simple_threshold(threshold: int | float, im, context_args: dict = None):
    """Returns im > threshold"""
    return await in_to_bs_custom(lambda im: (im > threshold).astype(np.uint8), im, context_args)


# --------------------------------Predict List of Centroids-----------------------------------


def in_to_lc_blobdog_np_features(block: npt.NDArray[np.float32], min_sigma=1, max_sigma=2,
                                 threshold: float = 0.1, block_info=None) -> npt.NDArray[np.float32]:
    if block_info is not None:
        slices = block_info[0]['array-location']
        lc = skimage.feature.blob_dog(np.array(block * 255, dtype=np.uint8),
                                      min_sigma=min_sigma,
                                      max_sigma=max_sigma,
                                      threshold=threshold).astype(np.float64)  # N * (ndim + 1) ndarray
        start_pos = np.array([slices[i].start for i in range(len(slices))], dtype=np.float64)
        lc[:, :block.ndim] += start_pos[None, :]
        return lc
    else:
        return block


async def in_to_lc_blobdog_forward(im: npt.NDArray[np.float32] | da.Array,
                                   min_sigma=1,
                                   max_sigma=2,
                                   threshold: float = 0.1,
                                   reduce: bool = False,
                                   context_args: dict = None
                                   ) -> NDBlock:
    if context_args is None:
        context_args = {}
    viewer_args = context_args.get('viewer_args')
    viewer = viewer_args.get('viewer', None)

    coroutine = NDBlock.map_ndblocks([NDBlock(im)],
                                     fn=in_to_lc_blobdog_np_features,
                                     out_dtype=np.float64,
                                     fn_args=dict(min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold))
    ndblock = await tlfs.cache_im(coroutine, context_args=context_args)

    if viewer and viewer_args.get('display_points', True):
        blobdog = await ndblock.reduce(force_numpy=True)
        lc_interpretable_napari('blobdog_centroids', blobdog, viewer, im.ndim, ['sigma'])

    ndblock = ndblock.select_columns(slice(im.ndim))
    if reduce:
        ndblock = await ndblock.reduce(force_numpy=False)
    return ndblock


# -------------------------------Direct Cell Count Prediction----------------------------------


def in_to_cc_sum_scaled_intensity_np_features(block: npt.NDArray[np.float32],
                                              scale: float = .008,
                                              min_thres: float = 0.,
                                              spatial_box_width: None | int = None,
                                              block_info=None) -> npt.NDArray[np.float64]:
    if block_info is not None:
        slices = block_info[0]['array-location']
        if spatial_box_width is not None:
            padded_block = algorithms.pad_to_multiple(block, spatial_box_width)
        else:
            padded_block = block
        masked = padded_block * (padded_block > min_thres)
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
            features[:, -1] = masked.flatten() * scale
            features[:, :-1] = startpoint[None, :] + (inds + .5) * np.array(subblock_shape, dtype=np.float64)
        else:
            features = np.zeros((1, block.ndim + 1), dtype=np.float64)
            features[0, -1] = masked.sum() * scale
            features[:, :-1] = startpoint[None, :] + np.array(block.shape, dtype=np.float64) / 2
        return features
    else:
        return block


async def in_to_cc_sum_scaled_intensity(im,
                                        scale: float = .008,
                                        min_thres: float = 0.,
                                        spatial_box_width: None | int = None,
                                        reduce: bool = True,
                                        context_args: dict = None
                                        ):
    """Summing up the intensity and scale it to obtain number of cells, directly

    Args:
        im: The image to perform sum on
        scale: Scale the sum of intensity by this to get number of cells
        min_thres: Intensity below this threshold is excluded (set to 0 before summing)
        spatial_box_width: If not None, will use this as the box width for adding points to Napari
        reduce: If True, reduce the result to a single number in numpynumpy array
        context_args: Contextual arguments \
            - viewer_args (dict, optional): specifies the viewer arguments related to napari display of intermediate \
                and end results \
            - cache_url (str | RDirFileSystem, optional): Points to the directory under which cache will be saved

    Returns:
        An array in which number represents the estimated object count in chunk after summing intensity
    """
    if context_args is None:
        context_args = {}
    storage_options = context_args.get('storage_options', None)
    viewer_args = context_args.get('viewer_args', {})
    cache_url = context_args.get('cache_url', None)
    viewer = viewer_args.get('viewer', None)
    if cache_url is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(cache_url)

    async def feature_forward(im: npt.NDArray[np.float32] | da.Array, **kwargs) -> NDBlock[np.float64]:
        return await NDBlock.map_ndblocks([NDBlock(im)],
                                          in_to_cc_sum_scaled_intensity_np_features,
                                          out_dtype=np.float64,
                                          fn_args=kwargs)

    forwarded = await tlfs.cache_im(
        fn=feature_forward(im, scale=scale, min_thres=min_thres, spatial_box_width=None),
        context_args=dict(cache_url=None if fs is None else fs['forward_pass'])
    )

    if viewer:
        mask = await tlfs.cache_im(
            fn=lambda: im > min_thres,
            context_args=dict(cache_url=None if fs is None else fs['viewer_pass_mask'],
                              viewer_args=viewer_args | dict(is_label=True),
                              storage_options=storage_options))
        if spatial_box_width is not None and viewer_args.get('display_points', True):
            feature_forwarded = await feature_forward(im,
                                                      scale=scale,
                                                      min_thres=min_thres,
                                                      spatial_box_width=spatial_box_width)
            ssi = await tlfs.cache_im(
                fn=feature_forwarded.reduce(force_numpy=True),
                context_args=dict(cache_url=None if fs is None else fs['ssi'])
            )
            lc_interpretable_napari('ssi_block', ssi, viewer, im.ndim, ['ncells'])

        aggregate_ndblock: NDBlock[np.float64] = await tlfs.cache_im(
            fn=map_ncell_vector_to_total(forwarded),
            context_args=dict(cache_url=None if fs is None else fs['aggregate_ndblock'])
        )
        chunk_size = NDBlock(im).get_chunksize()
        await heatmap_logging(aggregate_ndblock, None if fs is None else fs['cell_density_map'], viewer_args,
                              chunk_size)

    async def fn():
        ndblock = forwarded.select_columns([-1])
        if reduce:
            ndblock = await ndblock.reduce(force_numpy=False)
        return ndblock

    ssi_result = await tlfs.cache_im(fn=fn(),
                                     context_args=dict(cache_url=None if fs is None else fs['ssi_result']))

    return ssi_result


# --------------------Convert Binary and Centroid list to Instance Mask------------------------


def split_ndarray_by_centroid(
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


def bs_lc_to_os_np_forward(bs: npt.NDArray[np.uint8],
                           lc: npt.NDArray[np.float64],
                           max_split: int = 10,
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
        if ncentroid <= 1 or ncentroid > max_split:
            continue

        # otherwise, divide the contour and map pixels to each
        indices = [i] + [lbl for lbl in range(new_lbl, new_lbl + ncentroid - 1)]
        new_lbl += ncentroid - 1
        mask = lbl_im[slices] == i
        stpt = np.array(tuple(s.start for s in slices), dtype=np.int64)
        centroids = [centroid - stpt for centroid in centroids]
        divided = algorithms.coord_map(mask.shape,
                                       lambda *X: split_ndarray_by_centroid(centroids, indices, X))

        lbl_im[slices] = lbl_im[slices] * ~mask + divided * mask

    return lbl_im


async def bs_lc_to_os_forward(bs: npt.NDArray[np.uint8] | da.Array,
                              lc: NDBlock[np.float64],
                              max_split: int = 10,
                              context_args: dict = None
                              ) -> npt.NDArray[np.int32] | da.Array:
    if context_args is None:
        context_args = {}
    cache_url = context_args.get('cache_url', None)
    if cache_url is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(cache_url)
    viewer_args = context_args.get('viewer_args', {})
    viewer = viewer_args.get('viewer', None)
    storage_options = context_args.get('storage_options', None)

    if viewer:
        def compute_lbl():
            if isinstance(bs, np.ndarray):
                lbl_im = instance_label(bs)[0]
            else:
                lbl_im = bs.map_blocks(
                    lambda block: instance_label(block)[0].astype(np.int32), dtype=np.int32
                )
            return lbl_im

        lbl_im = await tlfs.cache_im(fn=compute_lbl,
                                     context_args=dict(
                                         cache_url=None if fs is None else fs['before_split'],
                                         viewer_args=viewer_args | dict(is_label=True),
                                         storage_options=storage_options
                                     ))

    async def compute_result():
        nonlocal bs, lc
        bs = NDBlock(bs)
        is_numpy = bs.is_numpy() or lc.is_numpy()
        if is_numpy:
            # if one is numpy but not both numpy, force both of them to be numpy
            if not lc.is_numpy():
                lc = await lc.reduce(force_numpy=True)
            elif not bs.is_numpy():
                bs = await bs.as_numpy()

        ndblock = await NDBlock.map_ndblocks([bs, lc], bs_lc_to_os_np_forward,
                                             out_dtype=np.int32, use_input_index_as_arrloc=0,
                                             fn_args=dict(max_split=max_split))
        if is_numpy:
            result = await ndblock.as_numpy()
        else:
            result = ndblock.as_dask_array(storage_options=storage_options)
        return result

    result = await tlfs.cache_im(fn=compute_result,
                                 context_args=dict(
                                     cache_url=None if fs is None else fs['result'],
                                     viewer_args=viewer_args | dict(is_label=True),
                                     storage_options=storage_options
                                 ))
    return result


async def map_ncell_vector_to_total(ndblock: NDBlock[np.float64]) -> NDBlock[np.float64]:
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

    return await NDBlock.map_ndblocks([ndblock], map_fn, out_dtype=np.float64)


async def heatmap_logging(aggregate_ndblock: NDBlock[np.float64],
                          fs: RDirFileSystem,
                          viewer_args: dict,
                          chunk_size: tuple):
    if fs is not None:
        query = tlfs.cdir_init(fs)
    else:
        query = None

    if query is None or query.commit is None:
        block = aggregate_ndblock.select_columns([-1])
        ndim = block.get_ndim()

        def map_fn(block: npt.NDArray[np.float64], block_info=None):
            return block.reshape((1,) * ndim)

        block = await block.map_ndblocks([block], map_fn, out_dtype=np.float64)
        block = await block.as_numpy()

        if query is not None:
            with fs.open('density_map.npy', mode='wb') as fd:
                np.save(fd, block)
            tlfs.cdir_commit(fs)

    if viewer_args is None:
        viewer_args = {}
    viewer: Viewer = viewer_args.get('viewer', None)

    if query is not None:
        with fs.open('density_map.npy', mode='rb') as fd:
            block = np.load(fd)
    if viewer:
        block = np.log2(block + 1.)
        viewer.add_image(block, name='cell_density_map', scale=chunk_size, blending='additive', colormap='red',
                         translate=tuple(sz / 2 for sz in chunk_size))
