"""
Modified from reference: https://github.com/dask/dask-image/blob/main/dask_image/ndinterp/__init__.py

Copyright (c) 2017-2018, dask-image Developers (see AUTHORS.rst for details)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import functools
import math
from itertools import product

import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import scipy
from scipy.ndimage import affine_transform as ndimage_affine_transform

from dask_image.dispatch._dispatch_ndinterp import (
    dispatch_affine_transform,
    dispatch_asarray,
    dispatch_spline_filter,
)
from dask_image.ndfilters._utils import _get_depth_boundary


# ----------------------------------- affine transform -----------------------------------------


def scale_nearest(image: da.Array, scale: float | tuple[float, ...], output_shape: tuple[int, ...],
                  output_chunks: tuple[int, ...] = None, **kwargs) -> da.Array:
    """Scaling the image without interpolation (order=0)

    Note this function may cause a noticeable positional shift when scaling down

    Args:
        image: The image to be scaled by the given factor
        scale: The scale to be applied to each axis; if float
        output_shape: Shape of the output array
        output_chunks: Shape of the chunks of the output array
        **kwargs: arguments to be passed to affine_transform_nearest

    Returns:
        Scaled dask array
    """
    if isinstance(scale, (int, float)):
        scale = (scale,) * image.ndim + (1,)
    else:
        assert len(scale) == image.ndim, f'Scale vector should be a vector of same length as image number of dims'
        scale = np.concatenate((scale, (1,)), axis=0)
    matrix = np.diag(1 / np.array(scale, dtype=np.float32))
    return affine_transform_nearest(image,
                                    matrix,
                                    output_shape=output_shape,
                                    output_chunks=output_chunks,
                                    **kwargs)


def affine_transform_nearest(
        image: da.Array,
        matrix,
        output_shape: tuple[int, ...] = None,
        output_chunks: tuple[int, ...] = None,
        **kwargs
) -> da.Array:
    """Apply an affine transform using Dask. For every
    output chunk, only the slice containing the relevant part
    of the image is processed. Chunkwise processing is performed
    either using `ndimage.affine_transform` or
    `cupyx.scipy.ndimage.affine_transform`, depending on the input type.

    Position of each voxel in input and output is assumed to be the
    center of the cube e.g. for 3d this is (x + .5, y + .5, z + .5),
    note that this is different from scipy's affine_transform where
    the position is (x, y, z)

    Notes
    -----
        Differences to `ndimage.affine_transformation`:
        - modes 'reflect', 'mirror' and 'wrap' are not supported

        Arguments equal to `ndimage.affine_transformation`,
        except for `output_chunks`.

    Parameters
    ----------
    image : array_like (Numpy Array, Cupy Array, Dask Array...)
        The image array.
    matrix : array (ndim, ndim+1) or (ndim+1, ndim+1)
        Transformation matrix.
    output_shape : tuple of ints, optional
        The shape of the array to be returned.
    output_chunks : tuple of ints, optional
        The shape of the chunks of the output Dask Array.

    Returns
    -------
    affine_transform : Dask Array
        A dask array representing the transformed output

    """
    assert not np.isnan(matrix).any(), f'affine matrix is not supposed to contain nan, however, found matrix={matrix}'
    assert kwargs.get('order', 0) == 0, f'affine_transform_nearest does not take order parameter!'
    NDIM = image.ndim
    assert matrix.shape[0] == NDIM or matrix.shape[0] == NDIM + 1, (f'affine matrix must be of height same as '
                                                                    f'image.ndim or image.ndim + 1 '
                                                                    f'(bottom row stripped), got shape {matrix.shape}')
    assert matrix.shape[1] == NDIM + 1, (f'affine matrix must be of width same as image.ndim + 1, '
                                         f'got shape {matrix.shape}')
    offset = matrix[:NDIM, NDIM]
    matrix = matrix[:NDIM, :NDIM]

    if not isinstance(image, da.core.Array):
        image = da.from_array(image)

    if output_shape is None:
        output_shape = image.shape

    if output_chunks is None:
        output_chunks = image.shape

    # Perform test run to ensure parameter validity.
    ndimage_affine_transform(np.zeros([0] * NDIM),
                             matrix,
                             0)

    matrix = np.asarray(matrix)

    cval = kwargs.pop('cval', 0)
    mode = kwargs.pop('mode', 'nearest')
    prefilter = kwargs.pop('prefilter', False)

    supported_modes = ['constant', 'nearest']
    if scipy.__version__ > np.lib.NumpyVersion('1.6.0'):
        supported_modes += ['grid-constant']
    if mode in ['wrap', 'reflect', 'mirror', 'grid-mirror', 'grid-wrap']:
        raise NotImplementedError(
            f"Mode {mode} is not currently supported. It must be one of "
            f"{supported_modes}.")

    image_shape = image.shape

    # calculate output array properties
    normalized_chunks = da.core.normalize_chunks(output_chunks,
                                                 tuple(output_shape))
    block_indices = product(*(range(len(bds)) for bds in normalized_chunks))
    block_offsets = [np.cumsum((0,) + bds[:-1]) for bds in normalized_chunks]

    # use dispatching mechanism to determine backend
    affine_transform_method = dispatch_affine_transform(image)
    asarray_method = dispatch_asarray(image)

    # construct dask graph for output array
    # using unique and deterministic identifier
    output_name = 'affine_transform-' + tokenize(image, matrix,
                                                 output_shape, output_chunks,
                                                 kwargs)
    output_layer = {}
    rel_images = []
    for ib, block_ind in enumerate(block_indices):

        out_chunk_shape = [normalized_chunks[dim][block_ind[dim]]
                           for dim in range(NDIM)]
        out_chunk_offset = [block_offsets[dim][block_ind[dim]]
                            for dim in range(NDIM)]

        out_chunk_edges = np.array([i for i in np.ndindex((2,) * NDIM)]) \
                          * np.array(out_chunk_shape) + np.array(out_chunk_offset)

        rel_image_edges = np.dot(matrix, out_chunk_edges.T).T + offset

        rel_image_i = np.min(rel_image_edges, 0)
        rel_image_f = np.max(rel_image_edges, 0)

        # Calculate edge coordinates required for the footprint of the
        # spline kernel according to
        # https://github.com/scipy/scipy/blob/9c0d08d7d11fc33311a96d2ac3ad73c8f6e3df00/scipy/ndimage/src/ni_interpolation.c#L412-L419 # noqa: E501
        # Also see this discussion:
        # https://github.com/dask/dask-image/issues/24#issuecomment-706165593 # noqa: E501
        for dim, s in zip(range(NDIM), image_shape):
            rel_image_i[dim] = np.floor(rel_image_i[dim] + .5)
            rel_image_f[dim] = np.floor(rel_image_f[dim] + .5)

            rel_image_i[dim] = np.clip(rel_image_i[dim], 0, s - 1)
            rel_image_f[dim] = np.clip(rel_image_f[dim], 1, s)

        rel_image_slice = []
        for dim in range(NDIM):
            imin = int(rel_image_i[dim])
            imax = max(imin + 1, int(rel_image_f[dim]))
            rel_image_slice.append(slice(imin, imax))
        rel_image_slice = tuple(rel_image_slice)

        rel_image = image[rel_image_slice]

        """Block comment for future developers explaining how `offset` is
        transformed into `offset_prime` for each output chunk.
        Modify offset to point into cropped image.
        y = Mx + o
        Coordinate substitution:
        y' = y - y0(min_coord_px)
        x' = x - x0(chunk_offset)
        Then:
        y' = Mx' + o + Mx0 - y0
        M' = M
        o' = o + Mx0 - y0
        """

        offset_prime = offset + np.dot(matrix, out_chunk_offset) - rel_image_i - .5

        output_layer[(output_name,) + block_ind] = (
            affine_transform_method,
            (da.core.concatenate3, rel_image.__dask_keys__()),
            asarray_method(matrix),
            offset_prime,
            tuple(out_chunk_shape),
            None,  # out
            0,
            mode,
            cval,
            False  # prefilter
        )

        rel_images.append(rel_image)

    graph = HighLevelGraph.from_collections(output_name, output_layer,
                                            dependencies=[image] + rel_images)

    meta = dispatch_asarray(image)([0]).astype(image.dtype)

    transformed = da.Array(graph,
                           output_name,
                           shape=tuple(output_shape),
                           # chunks=output_chunks,
                           chunks=normalized_chunks,
                           meta=meta)

    return transformed


# magnitude of the maximum filter pole for each order
# (obtained from scipy/ndimage/src/ni_splines.c)
_maximum_pole = {
    2: 0.171572875253809902396622551580603843,
    3: 0.267949192431122706472553658494127633,
    4: 0.361341225900220177092212841325675255,
    5: 0.430575347099973791851434783493520110,
}


def _get_default_depth(order, tol=1e-8):
    """Determine the approximate depth needed for a given tolerance.

    Here depth is chosen as the smallest integer such that ``|p| ** n < tol``
    where `|p|` is the magnitude of the largest pole in the IIR filter.
    """
    return math.ceil(np.log(tol) / np.log(_maximum_pole[order]))


def spline_filter(
        image,
        order=3,
        output=np.float64,
        mode='mirror',
        output_chunks=None,
        *,
        depth=None,
        **kwargs
):
    if not isinstance(image, da.core.Array):
        image = da.from_array(image)

    # use dispatching mechanism to determine backend
    spline_filter_method = dispatch_spline_filter(image)

    try:
        dtype = np.dtype(output)
    except TypeError:  # pragma: no cover
        raise TypeError(  # pragma: no cover
            "Could not coerce the provided output to a dtype. "
            "Passing array to output is not currently supported."
        )

    if depth is None:
        depth = _get_default_depth(order)

    if mode == 'wrap':
        raise NotImplementedError(
            "mode='wrap' is unsupported. It is recommended to use 'grid-wrap' "
            "instead."
        )

    # Note: depths of 12 and 24 give results matching SciPy to approximately
    #       single and double precision accuracy, respectively.
    boundary = "periodic" if mode == 'grid-wrap' else "none"
    depth, boundary = _get_depth_boundary(image.ndim, depth, boundary)

    # cannot pass a func kwarg named "output" to map_overlap
    spline_filter_method = functools.partial(spline_filter_method,
                                             output=dtype)

    result = image.map_overlap(
        spline_filter_method,
        depth=depth,
        boundary=boundary,
        dtype=dtype,
        meta=image._meta,
        # spline_filter kwargs
        order=order,
        mode=mode,
    )

    return result


# ----------------------------------- block reduce -----------------------------------------


def measure_block_reduce(image: da.Array, block_size: int | tuple[int, ...],
                         reduce_fn, cval: int | float = 0, input_chunks: tuple[int, ...] = None) -> da.Array:
    """Extension of skimage block reduce to dask"""
    from skimage.measure import block_reduce

    ndim = image.ndim
    if isinstance(block_size, int):
        block_size = (block_size,) * ndim

    def process_block(block, block_info=None):
        im = block_reduce(block, block_size=block_size, func=reduce_fn, cval=cval)
        return im

    if input_chunks is None:
        IDEAL_SIZE = 1000000  # a block size to aim for
        nexpand = max(int(np.power((IDEAL_SIZE / np.prod(block_size)), 1/ndim).item()), 1)
        input_chunks = tuple(nexpand * s for s in block_size)
    image.rechunk(input_chunks)

    result = image.map_blocks(process_block, meta=np.array(tuple(), dtype=image.dtype)).persist()

    result.compute_chunk_sizes()
    return result


if __name__ == '__main__':
    import dask.array as da
    from numpy.testing import assert_equal

    im = da.from_array(np.zeros((3, 2), dtype=np.int32))
    im[1, 1] = 2
    im[2, 0] = 1
    im = measure_block_reduce(im, block_size=(2, 3), reduce_fn=np.max)

    # chunk would be chosen to fit IDEAL_SIZE which covers the entire image in this case
    im.compute_chunk_sizes()
    assert im.chunksize == (2, 1), im.chunksize

    assert_equal(im.compute(), np.array(((2,), (1,)), dtype=np.int32))

    im = da.from_array(np.ones((3, 3, 6), dtype=np.float32))
    im = measure_block_reduce(im, block_size=(2, 3, 3), reduce_fn=np.min, cval=-1)
    assert_equal(im.compute(), np.array((((1, 1),), ((-1, -1),)), dtype=np.float32))
