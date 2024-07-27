"""
This file is for cv algorithms
"""


from typing import Callable
import numpy as np


def coord_map(im_shape: tuple, map_fn: Callable) -> np.ndarray:
    """
    Take a function mapping coordinates to pixel values and generate the specified image; np.indices
    is used in the underlying implementation.
    Args:
        im_shape: The shape of the image
        map_fn: The function that maps (numpy arrays of) indices to pixel values
    Returns:
        The image whose pixel values are specified by the function

    Example:
        >>> coord_map((2, 3), lambda Y, X: X - Y)
        array([[ 0,  1,  2],
               [-1,  0,  1]])
    """
    coords = np.indices(im_shape)
    im = map_fn(*coords)
    return im


def np_map_block(im: np.ndarray, block_sz) -> np.ndarray:
    """map_block(), but for numpy arrays

    Makes image from shape (Z, Y, X...) into (Zb, Yb, Xb..., Zv, Yv, Xv...) where b are block indices within space and
    v are voxel spatial indices within the block.
    Image size should be divisible by block size.

    Args:
        im: The numpy array of n dimensions to be mapped
        block_sz: the shape of each block

    Returns:
        Expanded array with 2n dimensions in total, first n are block indices and last n are voxel indices
    """
    assert im.ndim == len(block_sz), (f'Got block shape {block_sz} of ndim={len(block_sz)} different from im.ndim='
                                      f'{im.ndim}!')
    expanded_shape = []
    for i in range(im.ndim):
        block_axlen = im.shape[i] // block_sz[i]
        voxel_axlen = block_sz[i]
        assert voxel_axlen * block_axlen == im.shape[i], (f'Got indivisible image shape {im.shape[i]} by block size '
                                                          f'{block_sz[i]} on axis {i}')
        expanded_shape.extend((block_axlen, voxel_axlen))

    ax_order = [2 * i for i in range(im.ndim)] + [1 + 2 * i for i in range(im.ndim)]
    expanded_im = im.reshape(expanded_shape)
    return expanded_im.transpose(ax_order)
