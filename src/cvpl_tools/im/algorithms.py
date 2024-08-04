"""
This file is for cv algorithms
"""


from typing import Callable
import numpy as np
import skimage
from scipy.ndimage import (
    distance_transform_edt as distance_transform_edt,
    label as instance_label,
    find_objects as find_objects
)


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


def pad_to_multiple(arr: np.ndarray, n: int) -> np.ndarray:
    """Numpy, pad an array on each axis to a multiple of n.

    This function ensures no operation is done and original array is returned if the shape is already
    a multiple for each dimension. Other-wise a new array will be created with minimum shape matching
    the requirement and it will be returned.

    Args:
        arr: The array to be padded
        n: Each axis should be padded to a multiple of this number

    Returns:
        The padded array
    """
    pad_width = tuple((0, (n - dim % n) % n) for dim in arr.shape)
    for tup in pad_width:
        if tup != (0, 0):
            break
        return arr
    return np.pad(array=arr,
                  pad_width=pad_width,
                  mode='constant')


# ----------------------------Specific Image Processing Algorithms-----------------------------


def find_np3d_from_bs(mask: np.ndarray[np.uint8]) -> list[np.ndarray[np.int64]]:
    """Find sparse representation of the contour locations for a contour mask.

    "bs"=binary segmentation mask

    The input lbl_im is an image with pixel values 0-N, typically returned from scipy.ndimage.label() or
    skimage.morphology.label().

    Args:
        mask: The binary mask to be segmented into contours represented in argwhere format

    Returns:
        A list of contours, each represented in a Mi * d ndarray of type np.int64. Each row is a location
        vector indicating the pixel location, and there are Mi pixels making up the ith contour.

        If some number < N in lbl_im corresponds to no pixels in the array, the corresponding entry will
        be None. This function will not label lbl_im == 0 (which is assumed to be the background class)
    """
    lbl_im = instance_label(mask)[0]
    return npindices_from_os(lbl_im)


def npindices_from_os(lbl_im: np.ndarray[np.int32]) -> list[np.ndarray[np.int64]]:
    """Find sparse representation of the contour locations for a contour mask.

    "os"=ordinal segmentation mask

    The input lbl_im is an image with pixel values 0-N, typically returned from scipy.ndimage.label() or
    skimage.morphology.label().

    Args:
        lbl_im: The image with pixel values 0-N, each number 1-N correspond to a separate object.

    Returns:
        A list of contours, each represented in a Mi * d ndarray of type np.int64. Each row is a location
        vector indicating the pixel location, and there are Mi pixels making up the ith contour.

        If some number < N in lbl_im corresponds to no pixels in the array, the corresponding entry will
        be None. This function will not label lbl_im == 0 (which is assumed to be the background class)
    """
    object_slices = find_objects(lbl_im)
    result = []
    for slices in object_slices:
        if slices is None:
            result.append(None)
            continue

        i = len(result) + 1
        mask_np3d = lbl_im[slices] == i
        result.append(np.argwhere(mask_np3d))
    return result


# ------------------------------------------Watershed------------------------------------------


def watershed(seg_bin, dist_thres=1., remove_smaller_than=None):
    """Run Watershed algorithm to perform instance segmentation.

    The result is an index labeled int64 mask

    Args:
        seg_bin: The binary [0, 1] 3d mask to run watershed on to separate blobs into instances.
        dist_thres: Only pixels this much into the contours are retained; pixels on contours surface are removed
        remove_smaller_than: Contours smaller than this value are added as part of neighboring contours (once)

    Returns:
        The instance segmented int64 mask from 0 to N, where N is number of objects
    """
    # reference: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    # fp_width = 2
    # fp = [(np.ones((fp_width, 1, 1)), 1), (np.ones((1, fp_width, 1)), 1), (np.ones((1, 1, fp_width)), 1)]
    # sure_bg = morph.binary_dilation(seg_bin, fp)
    sure_bg = seg_bin
    # sure_fg = morph.binary_erosion(seg_bin, fp)
    dist_transform = distance_transform_edt(seg_bin)
    sure_fg = dist_transform >= dist_thres
    unknown = sure_bg ^ sure_fg
    lbl_im = skimage.morphology.label(sure_fg, connectivity=1)

    # so that sure_bg is 1 and unknown region is 0
    lbl_im += 1
    lbl_im[unknown == 1] = 0
    result = skimage.segmentation.watershed(-dist_transform, lbl_im, connectivity=1)
    result -= result > 0  # we don't need to mark sure_bg as 1, make all marker id smaller by 1

    if remove_smaller_than is not None:
        # watershed again over the small object regions
        small_mask, big_mask = split_labeled_objects(result, remove_smaller_than, connectivity=1)
        lbl_im[small_mask] = 0
        result = skimage.segmentation.watershed(-dist_transform, lbl_im, connectivity=1)

        # at this point, there may be unfilled space, which are rare cases where lots of small objects make up a
        # larger connected part; these space should be cells but the individual objects do not meet size requirement
        unfilled = skimage.morphology.label(result == 0, connectivity=1)
        result += (unfilled != 0) * (unfilled + result.max())

        result -= result > 0  # we don't need to mark sure_bg as 1, make all marker id smaller by 1

    return result


def split_labeled_objects(lbl_im, size_thres, connectivity=1):
    component_sizes = np.bincount(lbl_im.ravel())
    small_inds = component_sizes < size_thres
    small_inds[0] = False
    small_mask = small_inds[lbl_im]
    big_inds = component_sizes >= size_thres
    big_inds[0] = False
    big_mask = big_inds[lbl_im]
    return small_mask, big_mask


def split_objects(seg, size_thres, connectivity=1):
    lbl_im = skimage.morphology.label(seg, connectivity=connectivity)
    return split_labeled_objects(lbl_im, size_thres, connectivity)


def round_object_detection_3sizes(seg, size_thres, dist_thres, rst, size_thres2, dist_thres2, rst2):
    """detect round objects in a binary image

    Args:
        seg: the binary mask where we want to detect on
        size_thres: the size threshold below which size, contours are all kept as is
        dist_thres: threshold for seeding in the watershed algorithm
        rst:
        size_thres2:
        dist_thres2:
        rst2:

    Returns:
        Same shape image labeled 0, 1, ..., n, where 0 is background and 1...n are detected objects
    """
    # objects too small cannot be connected. We use this property to first find small objects that must be single cells
    small_mask, big_mask = split_objects(seg, size_thres, connectivity=1)
    big_mask, big_mask2 = split_objects(big_mask, size_thres2, connectivity=1)

    # big objects may be a single large cell or overlapping small cells, run watershed to separate overlapping cells
    lbl_im = watershed(big_mask, dist_thres=dist_thres, remove_smaller_than=rst)
    lbl_im2 = watershed(big_mask2, dist_thres=dist_thres2, remove_smaller_than=rst2)

    # finally, combine the two set of cells
    small_labeled = skimage.morphology.label(small_mask, connectivity=1)
    lbl_im += (small_labeled != 0) * (small_labeled + lbl_im.max())
    lbl_im += (lbl_im2 != 0) * (lbl_im2 + lbl_im.max())
    # lbl_im = (lbl_im2 > 0) * 1 + (lbl_im > 0) * 2 + small_mask * 3

    return lbl_im
