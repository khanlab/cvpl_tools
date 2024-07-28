"""
This file provides visualization utilities for ome-zarr file, similar to napari-ome-zarr,
but includes features like displaying zip ome zarr files
"""

import napari
import zarr
import dask.array as da
from cvpl_tools.ome_zarr.io import load_zarr_group_from_path


# -------------Part 1: convenience functions, for adding ome zarr images using paths--------------


def add_ome_zarr_group_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None, kwargs=None):
    """Add an ome zarr group to napari viewer from given group path.

    This is a combination of load_zarr_group_from_path() and add_ome_zarr_group() functions.
    """
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_group(viewer, zarr_group, kwargs)


def add_ome_zarr_array_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None, kwargs=None):
    """Add an ome zarr array to napari viewer from given array path.

    This is a combination of load_zarr_array_from_path() and add_ome_zarr_group() functions.
    """
    if kwargs is None:
        kwargs = {}
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_array(viewer, zarr_group, **kwargs)


# ------------------------Part 2:adding ome zarr files using zarr group---------------------------


def add_ome_zarr_group(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group, kwargs=None):
    """Add an ome zarr image (if exists) along with its labels (if exist) to viewer.

    Args
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
        kwargs: dictionary, keyword arguments to be passed to viewer.add_image for root image
    """
    if kwargs is None:
        kwargs = {}
    if '0' in zarr_group:
        add_ome_zarr_array(viewer, zarr_group, **kwargs)
    if 'labels' in zarr_group:
        lbls_group = zarr_group['labels']
        for group_key in lbls_group.group_ordered_keys():
            lbl_group = lbls_group[group_key]
            add_ome_zarr_array(viewer, lbl_group, name=group_key)


def add_ome_zarr_array(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group, **kwargs):
    """Add a multiscale ome zarr image or label to viewer.

    Args:
        viewer (napari.Viewer): Napari viewer object to attach image to.
        zarr_group (zarr.hierarchy.Group): The zarr group that contains the ome zarr file.
        ``**kwargs``: Keyword arguments to be passed to viewer.add_image for root image.
    """
    multiscale = []
    while True:
        i = len(multiscale)
        i_str = str(i)
        if i_str in zarr_group:  # by ome zarr standard, image pyramid starts from 0 to NLEVEL - 1
            multiscale.append(da.from_zarr(zarr_group[i_str]))
        else:
            break
    viewer.add_image(multiscale, multiscale=True, **kwargs)
