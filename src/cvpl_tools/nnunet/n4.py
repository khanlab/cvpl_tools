import os.path
from os import write

import ants
import numpy as np
import numpy.typing as npt
import cvpl_tools.ome_zarr.io as ome_io
from cvpl_tools.fsspec import RDirFileSystem
import dask.array as da


def bias_correct(im: npt.NDArray, spline_param, shrink_factor, return_bias_field: bool = False) -> npt.NDArray:
    """If return_bias_field is True, return the bias field instead of the corrected image

    See ants.n4_bias_field_correction documentation

    Examples
        - bias_correct(im, spline_param=(16, ) * 3, shrink_factor=8, return_bias_field=return_bias_field)
    """
    orig_shape = im.shape
    im = ants.from_numpy(im)
    imn4 = ants.n4_bias_field_correction(im,
                                         spline_param=spline_param,
                                         shrink_factor=shrink_factor,
                                         return_bias_field=return_bias_field)
    imn4 = imn4.numpy()
    # imn4 = np.transpose(imn4, (2, 0, 1))
    n4_shape = imn4.shape
    assert orig_shape == n4_shape, f'n4 image got shape={n4_shape}, but got original shape={orig_shape}!'
    return imn4


async def obtain_bias(im: npt.NDArray | da.Array, write_loc=None, asynchronous: bool = False) -> npt.NDArray:
    """Returns a bias field numpy array that is of the same size and shape as the input

    Corrected image can be obtained by computing im / obtain_bias(im)
    """
    import asyncio
    import dask.array as da

    if write_loc is None or not RDirFileSystem(write_loc).exists(''):
        if isinstance(im, da.Array):
            im = im.compute()
        bias = bias_correct(im, spline_param=(8,) * 3, shrink_factor=4, return_bias_field=True)
        assert isinstance(bias, np.ndarray), f'{bias}'
        if write_loc is not None:
            await ome_io.write_ome_zarr_image(write_loc,
                                              da_arr=da.from_array(bias),
                                              MAX_LAYER=1,
                                              asynchronous=asynchronous)
    if write_loc is not None:
        bias = ome_io.load_dask_array_from_path(write_loc, mode='r', level=0).compute()

    return bias
