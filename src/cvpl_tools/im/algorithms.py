"""
This file is for algorithms manipulating numpy images
"""


from typing import Callable
import numpy as np


def coord_map(im_shape: tuple, func: Callable) -> np.ndarray:
    """
    Take a function mapping coordinates to pixel values and generate the specified image; np.indices
    is used in the underlying implementation.
    Args:
        im_shape: The shape of the image
        func: The function that maps (numpy arrays of) indices to pixel values
    Returns:
        The image whose pixel values are specified by the function

    Example:
        >>> coord_map((2, 3), lambda Y, X: X - Y)
        array([[ 0,  1,  2],
               [-1,  0,  1]])
    """
    coords = np.indices(im_shape)
    im = func(*coords)
    return im
