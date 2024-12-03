import ants
import numpy.typing as npt
import cvpl_tools.ome_zarr.io as ome_io


def bias_correct(im: npt.NDArray, spline_param, shrink_factor, return_bias_field: bool = False) -> npt.NDArray:
    """If return_bias_field is True, return the bias field instead of the corrected image

    See ants.n4_bias_field_correction documentation

    Examples
        - bias_correct(im, spline_param=(16, ) * 3, shrink_factor=8, return_bias_field=return_bias_field)
    """
    im = ants.from_numpy(im)
    imn4 = ants.n4_bias_field_correction(im,
                                         spline_param=spline_param,
                                         shrink_factor=shrink_factor,
                                         return_bias_field=return_bias_field)
    return imn4.numpy()


def obtain_bias(im: npt.NDArray, write_loc=None) -> npt.NDArray:
    """Returns a bias field numpy array that is of the same size and shape as the input

    Corrected image can be obtained by computing im / obtain_bias(im)
    """
    bias = bias_correct(im, spline_param=(16,) * 3, shrink_factor=8, return_bias_field=True)
    if write_loc is not None:
        ome_io.write_ome_zarr_image(write_loc, f'{write_loc}.tmp', da_arr=bias, MAX_LAYER=1)
    return bias
