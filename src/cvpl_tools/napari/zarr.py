"""
This file provides visualization utilities for ome-zarr file, similar to napari-ome-zarr,
but includes features like displaying zip ome zarr files
"""


import napari
import zarr
import dask.array as da


def load_zarr_group_from_path(path: str, mode=None, use_zip: bool | None=None):
    """Loads either a zarr folder or zarr zip file into a zarr group.

    Args:
        path: path to the zarr folder or zip to be opened
        mode: file open mode e.g. 'r', only pass this if the file is a zip file
        use_zip: if True, treat path as a zip; if False, treat path as a folder; if None,
            use path to determine file type
    Returns:
        the opened zarr group
    """
    if use_zip is None:
        use_zip = path.endswith('.zip')
    if use_zip:
        store = zarr.ZipStore(path, mode=mode)
        zarr_group = zarr.open(store, mode=mode)
    else:
        store = zarr.DirectoryStore(path)
        zarr_group = zarr.open(store, mode=mode)
    return zarr_group


# -------------Part 1: convenience functions, for adding ome zarr images using paths--------------


def add_ome_zarr_group_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None=None):
    """Add an ome zarr group to napari viewer from given group path.

    A combination of load_zarr_group_from_path() and add_ome_zarr_group() functions.
    """
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_group(viewer, zarr_group)


def add_ome_zarr_array_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None=None):
    """Add an ome zarr array to napari viewer from given array path.

    A combination of load_zarr_array_from_path() and add_ome_zarr_group() functions.
    """
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_array(viewer, zarr_group)


# ------------------------Part 2:adding ome zarr files using zarr group---------------------------


def add_ome_zarr_group(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group):  # TODO: add kwargs
    """Add an ome zarr image (if exists) along with its labels (if exist) to viewer.

    Args
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
    """
    if '0' in zarr_group:
        add_ome_zarr_array(viewer, zarr_group)
    if 'labels' in zarr_group:
        lbls_group = zarr_group['labels']
        for array_key in lbls_group.array_keys():
            lbl_group = lbls_group[array_key]
            add_ome_zarr_array(viewer, lbl_group)


def add_ome_zarr_array(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group, **kwargs):
    """Add a multi-scale ome zarr image or label to viewer.

    Args
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
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
