import copy

import napari
from mpmath.libmp import BACKEND

import cvpl_tools.ome_zarr.io as ome_zarr_io
import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
import cvpl_tools.ome_zarr.napari.add as nozadd
import os
import numpy as np
import dask.array as da
import asyncio


def ensure_downsample_tuple(ndownsample_level: int | tuple) -> tuple:
    if isinstance(ndownsample_level, int):
        ndownsample_level = (ndownsample_level,) * 3
    return ndownsample_level


def _apply_bias(im: da.Array, bias: da.Array) -> da.Array:
    """Guarantees result is dask array with elements of type np.uint16"""
    return (im / bias).clip(min=0, max=2 ** 16 - 1).astype(np.uint16)


def apply_bias(im, im_ndownsample_level: int | tuple, bias, bias_ndownsample_level: int | tuple, write_loc: str = None):
    im_ndownsample_level = ensure_downsample_tuple(im_ndownsample_level)
    bias_ndownsample_level = ensure_downsample_tuple(bias_ndownsample_level)

    corr_exists = write_loc is not None and os.path.exists(write_loc)

    if not corr_exists:
        upsampling_factor = tuple(2 ** (bias_ndownsample_level[i] - im_ndownsample_level[i])
                                  for i in range(len(im_ndownsample_level)))
        print(f'bias_shape:{bias.shape}, im.shape:{im.shape}, upsampling factor: {upsampling_factor}')
        bias = dask_ndinterp.scale_nearest(bias.rechunk((upsampling_factor[0], 4096, 4096)),
                                           scale=upsampling_factor,
                                           output_shape=im.shape, output_chunks=(4, 4096, 4096)).persist()
        im = _apply_bias(im, bias)
        if write_loc is not None:
            asyncio.run(ome_zarr_io.write_ome_zarr_image(write_loc, da_arr=im, MAX_LAYER=2))

    if write_loc is not None:
        im = ome_zarr_io.load_dask_array_from_path(write_loc, mode='r', level=0)

    return im


def get_optional_zip_path(path) -> None | str:
    """Give the path to a folder or its zipped file, return its path"""
    eff_zip_path = None
    if os.path.exists(path):
        eff_zip_path = path
    elif os.path.exists(f'{path}.zip'):
        eff_zip_path = f'{path}.zip'
    return eff_zip_path


def downsample(im,
               reduce_fn,
               ndownsample_level: int | tuple,
               ba_channel=None,
               write_loc: None | str = None,
               viewer_args: dict = None):
    """Downsample network image

    If write_loc is None, the downsampled image will not be saved

    Args:
        im (str | dask.array.Array): The original image's path, the image should be of shape (C, Z, Y, X)
        reduce_fn (Callable): Function of reduction used in measure_block_reduce
        ba_channel (int): The channel in original to take for down-sampling
        ndownsample_level (int | tuple): The number of downsamples in each axis
        write_loc (None | str): Location to write if provided
        viewer_args (dict):
    """
    if isinstance(ndownsample_level, int):
        ndownsample_level = (ndownsample_level,) * 3
    horizontal_min = min(ndownsample_level[1:])

    # create a downsample of the original image (can be on network)
    if not os.path.exists(write_loc):
        if isinstance(im, str):
            print(f'ome_io.load_dask_array_from_path from path {im}')
            group = ome_zarr_io.load_zarr_group_from_path(im, mode='r')
            if str(horizontal_min) not in group:
                horizontal_min = ome_zarr_io.get_highest_downsample_level(group)
            im = da.from_array(group[str(horizontal_min)])
            further_downsample = tuple(l - horizontal_min for l in ndownsample_level[1:])
        else:
            further_downsample = ndownsample_level[1:]
        if ba_channel is not None:
            im = im[ba_channel]

        print(f'Initial local image is not found, downsample from the network, network image is of size {im.shape}')
        downsample_factor_vertical = 2 ** ndownsample_level[0]
        im = dask_ndinterp.measure_block_reduce(im,
                                                block_size=(downsample_factor_vertical,
                                                            2 ** further_downsample[0],
                                                            2 ** further_downsample[1]),
                                                reduce_fn=reduce_fn,
                                                input_chunks=(downsample_factor_vertical, 4096, 4096)).rechunk((4, 4096, 4096))
        if write_loc is None:
            return im
        print(f'Downsampled image is of size {im.shape}, writing...')
        asyncio.run(ome_zarr_io.write_ome_zarr_image(write_loc, da_arr=im, MAX_LAYER=2))
    im = ome_zarr_io.load_dask_array_from_path(write_loc, mode='r', level=0)

    if viewer_args is None:
        viewer_args = {}
    else:
        viewer_args = copy.copy(viewer_args)
    viewer = viewer_args.pop('viewer', None)
    if viewer is not None:
        nozadd.group_from_path(viewer, write_loc, kwargs=viewer_args)

    return im


def calc_translate(voxel_scale: tuple, display_shape: tuple) -> tuple:
    return tuple((-display_shape[i] + voxel_scale[i]) / 2 for i in range(len(voxel_scale)))


def calc_tr_sc_args(voxel_scale: tuple, display_shape: tuple):
    translate = calc_translate(voxel_scale, display_shape)
    return dict(
        translate=translate,
        scale=voxel_scale,
    )

