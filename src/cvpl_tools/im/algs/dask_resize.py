from typing import Sequence

import dask.array as da
import numpy as np

from cvpl_tools.im.fs import CachePointer

# ----------------------------------- Part 1: rescale, pad and crop -------------------------------------------
"""
The main use case this section of code is: The user tries to resize and fit a smaller 
or larger mask array to another image array by matching the shape of the two images. This consists of two 
steps:
1. Upsample or downsample the mask to match image
2. When the upsample or downsample factor is not a whole integer, add or crop off some of the border pixels
"""


def pad_crop(array: da.Array,
             change_width: Sequence | int,
             mode='constant',
             **kwargs):
    """Similar to dask.array.pad, but allows negative change_width"""
    NDIM = array.ndim
    pad_tups = np.zeros((NDIM, 2), dtype=np.int64)
    if isinstance(change_width, int):
        pad_tups[:] = change_width
    else:
        change_width = tuple(change_width)
        if isinstance(change_width[0], int):
            # shallow change_width, wrap it in a tuple
            change_width = (change_width,)
        for i in range(array.ndim):
            if len(change_width) > i:
                pad_tup = change_width[i]
            else:
                pad_tup = change_width[-1]
            pad_tups[i, 0] = pad_tup[0]
            pad_tups[i, 1] = pad_tup[1]

    to_pad = pad_tups * (pad_tups > 0)
    to_crop = -pad_tups * (pad_tups < 0)

    padded = da.pad(array, to_pad.tolist(), mode, **kwargs)
    padded_shape = padded.shape
    crop_slices = tuple(slice(to_crop[i, 0], padded_shape[i] - to_crop[i, 1]) for i in range(NDIM))

    return padded[crop_slices]


def approx_upsample_scale(src_shape: tuple[int, ...],
                          tgt_shape: tuple[int, ...]
                          ) -> tuple[list[int, ...], list[int, ...]]:
    scale = []
    residue = []

    for i in range(len(src_shape)):
        ss, ts = src_shape[i], tgt_shape[i]
        s = max(round(ts / ss), 1)
        r = ts - ss * s
        scale.append(s)
        residue.append(r)
    return scale, residue


def approx_downpsample_scale(src_shape: tuple[int, ...],
                             tgt_shape: tuple[int, ...]
                             ) -> tuple[list[int, ...], list[int, ...]]:
    """The residue is calculated assuming scale is supplied as factor to the any_to_any.DownsamplingByIntFactor class"""
    scale = []
    residue = []

    for i in range(len(src_shape)):
        ss, ts = src_shape[i], tgt_shape[i]
        s = max(round(ss / ts), 1)
        r = ts - ((ss - 1) // s + 1)
        scale.append(s)
        residue.append(r)
    return scale, residue


def upsample_pad_crop_fit(src_arr: da.Array, tgt_arr: da.Array, cptr: CachePointer,
                          viewer_args: dict, mode='constant', **kwargs):
    import cvpl_tools.im.process.any_to_any as sp_any_to_any
    src_shape = src_arr.shape
    tgt_shape = tgt_arr.shape
    scale, residue = approx_upsample_scale(src_shape, tgt_shape)
    upsampler = sp_any_to_any.UpsamplingByIntFactor(factor=tuple(scale), order=0)
    arr = await upsampler.forward(src_arr, cptr=cptr, viewer_args=viewer_args)

    change_width = tuple((0, -r) for r in residue)
    arr = pad_crop(arr, change_width=change_width, mode=mode, **kwargs)
    return arr


if __name__ == '__main__':
    test_arr = da.zeros((4, 4), dtype=np.float32)
    test_arr[1, 1] = 1
    test_arr2 = pad_crop(test_arr, change_width=-1, mode='constant')
    from numpy.testing import assert_almost_equal

    assert_almost_equal(test_arr2,
                        np.array(
                            ((1., 0.),
                             (0., 0.)),
                            dtype=np.float32
                        ))
