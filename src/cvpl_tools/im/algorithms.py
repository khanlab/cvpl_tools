"""
This file is for cv algorithms
"""
import enum
from typing import Callable, Type
import numpy as np
import numpy.typing as npt
import skimage
from scipy.ndimage import (
    distance_transform_edt as distance_transform_edt,
    label as instance_label,
    find_objects as find_objects
)


def coord_map(im_shape: tuple, map_fn: Callable) -> npt.NDArray:
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


def np_map_block(im: npt.NDArray, block_sz) -> npt.NDArray:
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


def np_unique(lbl_im: npt.NDArray[np.int32], return_index: bool = False,
              return_inverse: bool = False, return_counts: bool = False,
              axis=None, *,
              equal_nan: bool = False):
    """Before 2.0.0, numpy will return an array of 1d regardless of input shape
    
    This function fixes this by forcing the function signature working the same as version 2
    """
    # this function fixes this by forcing the function signature working the same as version 2
    ret = np.unique(lbl_im, return_index, return_inverse, return_counts, axis, equal_nan=equal_nan)
    if return_inverse and np.__version__ < '2.0.0':
        im_shape = lbl_im.shape
        ret = ret[:-1] + (ret[-1].reshape(im_shape),)
    return ret


def pad_to_multiple(arr: npt.NDArray, n: int) -> npt.NDArray:
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


def setdiff2d(A, B):
    """An extension of np.setdiff1d to 2d array

    copied from
    https://stackoverflow.com/questions/64414944/hot-to-get-the-set-difference-of-two-2d-numpy-arrays-or-equivalent-of-np-setdif
    """
    nrows, ncols = A.shape
    dtype = dict(
        names=['f{}'.format(i) for i in range(ncols)],
        formats=ncols * [A.dtype]
    )
    C = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
    return C


# ----------------------------Specific Image Processing Algorithms-----------------------------


def find_np3d_from_bs(mask: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.int64]]:
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
    return npindices_from_os(lbl_im)[0]


def npindices_from_os(
        lbl_im: npt.NDArray[np.int32],
        return_object_slices: bool = False,
        is_sparse: bool = False,
) -> tuple:
    """Find sparse representation of the contour locations for a contour mask.

    "os"=ordinal segmentation mask

    The input lbl_im is an image with pixel values 0-N, typically returned from scipy.ndimage.label() or
    skimage.morphology.label().

    Args:
        lbl_im: The image with pixel values 0-N, each number 1-N correspond to a separate object.
        return_object_slices: If True, return the boundbox used to find the corresponding mask image
        is_sparse: If True, input index labels may be large int but only a few indices are present; also
            return the corresponding indices after the other returning data

    Returns:
        A list of contours, each represented in a Mi * d ndarray of type np.int64. Each row is a location
        vector indicating the pixel location, and there are Mi pixels making up the ith contour.

        If some number in lbl_im corresponds to no pixels in the array, the corresponding entry will
        be None. This function will not label lbl_im == 0 (which is assumed to be the background class)

        The returned result list's index 0 correspond to the first non-background instance; if slices are
        returned, the ith index of result list will correspond to the ith index of slices list
    """
    lbl_ndim = lbl_im.ndim
    if is_sparse:
        unique, unique_inverse = np_unique(lbl_im, return_inverse=True)
        unique = unique[1:]  # remove 0 = background class
        lbl_im = unique_inverse

    object_slices = find_objects(lbl_im)
    result = []
    result_slices = []
    for slices in object_slices:
        if slices is None:
            result.append(None)
            continue

        i = len(result) + 1
        mask_argwhere = np.argwhere(lbl_im[slices] == i)
        mask_argwhere += np.array(tuple(s.start for s in slices), dtype=np.int64)
        result.append(mask_argwhere)
        result_slices.append(slices)
        assert mask_argwhere.shape[1] == lbl_ndim, (f'Expected row length to match label ndim, got shape '
                                                    f'{mask_argwhere.shape} and ndim={lbl_ndim}')

    ret = (result,)
    if return_object_slices:
        ret += (result_slices,)
    if is_sparse:
        assert len(result) == len(unique), f'Expected equal length, got {len(result)}, {len(unique)}'
        ret += (unique,)
    return ret


# ------------------------------------------Watershed------------------------------------------


def watershed(seg_bin, dist_thres=1., remove_smaller_than=None):
    """Run Watershed algorithm to perform instance segmentation.

    The result is an index labeled int64 mask

    Args:
        seg_bin: The binary [0, 1] 3d mask to run watershed on to separate blobs into instances.
        dist_thres: Only pixels this much into the contours are retained; pixels on contours surface are removed
        remove_smaller_than: Contours smaller than this value are added as part of neighboring contours (once)

    Returns:
        The instance segmented int64 mask from 0 to N, where N is higher than the number of objects. Note
        this algorithm does not guarantee there are N objects found, since some of the contours between 1
        and N will be removed as a filter step
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


def round_object_detection_3sizes(seg, size_thres, dist_thres, rst, size_thres2, dist_thres2, rst2,
                                  remap_indices: bool = True):
    """detect round objects in a binary image

    Args:
        seg: the binary mask where we want to detect on
        size_thres: the size threshold below which size, contours are all kept as is
        dist_thres: threshold for seeding in the watershed algorithm
        rst:
        size_thres2:
        dist_thres2:
        rst2:
        remap_indices: If True, make sure `lbl_im.max()` is the number of objects; if this is False,
            `lbl_max()` may be larger than the actual number of objects detected

    Returns:
        Same shape image labeled 0, 1, ..., n, where 0 is background and 1...n are detected objects; note if
        remap_indices is False, there may not be n objects as some of the objects between 1 and n are removed
        from the list during filter step.
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

    if remap_indices:
        _, lbl_im = np_unique(lbl_im, return_inverse=True)

    return lbl_im


def voronoi_ndarray(im_shape: tuple, centroids: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Given k centroids in a voxel ndarray, color the array by voronoi algorithm

    Args:
        im_shape: Shape of the array to be colored
        centroids: The centroids in the space

    Returns:
        The indices of type np.int32 according to centroid order
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Algorithm provided by chatgpt-4o

    indices = np.indices(im_shape)
    grid_points = np.stack([idx.ravel() for idx in indices], axis=-1)

    # Use cKDTree to find the nearest centroid for each voxel
    tree = cKDTree(centroids)
    _, labels = tree.query(grid_points)
    return labels.astype(np.int32).reshape(im_shape)


# --------------------------------Statistical Analysis-------------------------------------


def Stats_MAE(counted, gt):
    """
    Params
        counted (list) - the counted cells in each image
        gt (Iterable[float]) - the ground truth number of cells in each image
    Returns
        the mean absolute difference between counted and gt
    """
    return np.abs(np.array(counted, dtype=np.float64) - np.array(gt, dtype=np.float64)).mean().item()


def Stats_ShowScatterPairComparisons(counted: np.array, gt, enumType: Type[enum.Enum]) -> None:
    """Show comparison between different algorithms counting results to the gt

    Args:
        counted: (a NCounter * NImages np.ndarray) The counted cells in each image, by each counter
        gt: (Iterable[float]) The ground truth number of cells in each image
        enumType: The enum class defined for counting
    """
    import matplotlib.pyplot as plt
    counting_method_inverse_dict = {item.value: item.name for item in enumType}
    nrow = int((len(counting_method_inverse_dict) - 1) / 6) + 1
    fig, axes = plt.subplots(nrow, 6, figsize=(24, nrow * 4), sharex=True, sharey=True)

    gt_arr = np.array(gt, dtype=np.float64)
    for i in range(counted.shape[0]):
        X, Y = gt_arr, counted[i]

        iax, jax = i // 6, i % 6
        ax: plt.Axes = axes[iax, jax]
        ax.set_box_aspect(1)
        ax.set_title(counting_method_inverse_dict[i])
        ax.scatter(X, Y)

    plt.show()


if __name__ == '__main__':
    centroids = ((2, 2), (0, 1), (0, 0))
    colored = voronoi_ndarray((3, 3), np.array(centroids, dtype=np.int32))
    assert colored.dtype == np.int32, f'got colored.dtype={colored.dtype}'
    colored[2, 0] = 0  # ambiguous as to whether this is 0 or 2, for testing purpose we don't mind this
    reference = np.array(
        ((2, 1, 1),
         (2, 1, 0),
         (0, 0, 0)), dtype=np.int32
    )
    assert (colored == reference).sum().item() == colored.size, f'got array {colored}'

    # lbl_im = np.array(
    #     (((1, 3),
    #       (0, 0)),
    #      ((0, 0),
    #       (0, 2))),
    #     dtype=np.int32
    # )
    # _contours_np3d, _ids = npindices_from_os(lbl_im, is_sparse=True)
    # for cnt in _contours_np3d:
    #     print('contour')
    #     print(cnt, cnt.shape)
    # print(_ids)
