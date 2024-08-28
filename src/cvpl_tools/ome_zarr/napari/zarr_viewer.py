"""
This file provides visualization utilities for ome-zarr file, similar to napari-ome-zarr,
but includes features like displaying zip ome zarr files
"""

import napari
import zarr
import dask.array as da
from cvpl_tools.ome_zarr.io import load_zarr_group_from_path


# -------------Part 1: convenience functions, for adding ome zarr images using paths--------------


def add_ome_zarr_group_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None,
                                 kwargs=None, lbl_kwargs=None):
    """Add an ome zarr group to napari viewer from given group path.

    This is a combination of load_zarr_group_from_path() and add_ome_zarr_group() functions.
    """
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_group(viewer, zarr_group, kwargs, lbl_kwargs)


def add_ome_zarr_array_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None, kwargs=None):
    """Add an ome zarr array to napari viewer from given array path.

    This is a combination of load_zarr_array_from_path() and add_ome_zarr_group() functions.
    """
    if kwargs is None:
        kwargs = {}
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_array(viewer, zarr_group, **kwargs)


# ------------------------Part 2:adding ome zarr files using zarr group---------------------------


def add_ome_zarr_group(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group,
                       kwargs: dict = None, lbl_kwargs: dict = None):
    """Add an ome zarr image (if exists) along with its labels (if exist) to viewer.

    Args:
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
        kwargs: dictionary, keyword arguments to be passed to viewer.add_image for root image
        lbl_kwargs: dictionary, keyword arguments to be passed to viewer.add_image for label images
    """
    if kwargs is None:
        kwargs = {}
    if '0' in zarr_group:
        add_ome_zarr_array(viewer, zarr_group, **kwargs)
    if 'labels' in zarr_group:
        lbls_group = zarr_group['labels']
        for group_key in lbls_group.group_keys():
            lbl_group = lbls_group[group_key]
            add_ome_zarr_array(viewer, lbl_group, name=group_key, **lbl_kwargs)


def add_ome_zarr_array(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group, start_level: int = 0,
                       is_label=False, **kwargs):
    """Add a multiscale ome zarr image or label to viewer.

    Args:
        viewer (napari.Viewer): Napari viewer object to attach image to.
        zarr_group (zarr.hierarchy.Group): The zarr group that contains the ome zarr file.
        start_level (int): The lowest level (highest resolution) to be added, default to 0
        is_label (bool): If True, display the image as label; this is suitable for instance segmentation
            masks where the results need a distinct color for each number
        ``**kwargs``: Keyword arguments to be passed to viewer.add_image for root image.
    """
    multiscale = []
    while True:
        i = len(multiscale) + start_level
        i_str = str(i)
        if i_str in zarr_group:  # by ome zarr standard, image pyramid starts from 0 to NLEVEL - 1
            multiscale.append(da.from_zarr(zarr_group[i_str]))
        else:
            break
    if is_label:
        viewer.add_labels(multiscale, multiscale=True, **kwargs)
    else:
        viewer.add_image(multiscale, multiscale=True, **kwargs)
