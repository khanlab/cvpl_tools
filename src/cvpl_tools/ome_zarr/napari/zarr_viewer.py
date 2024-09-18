"""
This file provides visualization utilities for ome-zarr file, similar to napari-ome-zarr,
but includes features like displaying zip ome zarr files
"""
from typing import Callable

import napari
import zarr
import dask.array as da
from cvpl_tools.ome_zarr.io import load_zarr_group_from_path
import numpy as np
import numpy.dtypes as np_dtypes


# -------------Part 1: convenience functions, for adding ome zarr images using paths--------------


def add_ome_zarr_group_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None,
                                 merge_channels=False, kwargs=None, lbl_kwargs=None):
    """Add an ome zarr group to napari viewer from given group path.

    This is a combination of load_zarr_group_from_path() and add_ome_zarr_group() functions.
    """
    assert isinstance(merge_channels, bool)
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_group(viewer, zarr_group,
                       merge_channels=merge_channels,
                       kwargs=kwargs,
                       lbl_kwargs=lbl_kwargs)


def add_ome_zarr_array_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None,
                                 merge_channels=False, kwargs=None):
    """Add an ome zarr array to napari viewer from given array path.

    This is a combination of load_zarr_array_from_path() and add_ome_zarr_group() functions.
    """
    assert isinstance(merge_channels, bool)
    if kwargs is None:
        kwargs = {}
    zarr_group = load_zarr_group_from_path(path, 'r', use_zip)
    add_ome_zarr_array(viewer, zarr_group, merge_channels=merge_channels, **kwargs)


# ------------------------Part 2:adding ome zarr files using zarr group---------------------------


def add_ome_zarr_group(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group,
                       merge_channels=False,
                       kwargs: dict = None, lbl_kwargs: dict = None):
    """Add an ome zarr image (if exists) along with its labels (if exist) to viewer.

    Args:
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
        merge_channels: If True, display the image as one layers instead of a layer per channel
        kwargs: dictionary, keyword arguments to be passed to viewer.add_image for root image
        lbl_kwargs: dictionary, keyword arguments to be passed to viewer.add_image for label images
    """
    assert isinstance(merge_channels, bool)
    if kwargs is None:
        kwargs = {}
    if '0' in zarr_group:
        add_ome_zarr_array(viewer, zarr_group,
                           merge_channels=merge_channels,
                           **kwargs)
    if 'labels' in zarr_group:
        if lbl_kwargs is None:
            lbl_kwargs = {}
        lbls_group = zarr_group['labels']
        for group_key in lbls_group.group_keys():
            lbl_group = lbls_group[group_key]
            add_ome_zarr_array(viewer, lbl_group, name=group_key, **lbl_kwargs)


def _add_ch(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group,
            arr_from_group: Callable[[zarr.hierarchy.Group], da.Array],
            start_level: int = 0,
            is_label=False, **kwargs):
    """Adds a channel or a set of channels to viewer"""
    multiscale = []
    while True:
        i = len(multiscale) + start_level
        i_str = str(i)
        if i_str in zarr_group:  # by ome zarr standard, image pyramid starts from 0 to NLEVEL - 1
            multiscale.append(arr_from_group(zarr_group[i_str]))
            ty = np.dtype(multiscale[-1].dtype)
            if isinstance(ty, np_dtypes.VoidDType):
                if list(ty.fields.items())[0][1][0] == np.int32:
                    new_ty = np.int32
                else:
                    print(f'got {ty} of type {type(ty)}')
                multiscale[-1] = multiscale[-1].astype(np.int32)
        else:
            break
    if is_label:
        viewer.add_labels(multiscale, multiscale=True, **kwargs)
    else:
        viewer.add_image(multiscale, multiscale=True, **kwargs)


def add_ome_zarr_array(viewer: napari.Viewer, zarr_group: zarr.hierarchy.Group,
                       merge_channels=False,
                       start_level: int = 0, is_label=False, **kwargs):
    """Add a multiscale ome zarr image or label to viewer.

    The first channel is assumed to be the channel dimension. This is relevant only if merge_channels=False

    Args:
        viewer (napari.Viewer): Napari viewer object to attach image to.
        zarr_group (zarr.hierarchy.Group): The zarr group that contains the ome zarr file.
        merge_channels: If True, display the image as one layers instead of a layer per channel
        start_level (int): The lowest level (highest resolution) to be added, default to 0
        is_label (bool): If True, display the image as label; this is suitable for instance segmentation
            masks where the results need a distinct color for each number
        ``**kwargs``: Keyword arguments to be passed to viewer.add_image for root image.
    """
    assert isinstance(merge_channels, bool)
    arr_shape = da.from_zarr(zarr_group[str(start_level)]).shape
    ndim = len(arr_shape)
    if ndim <= 2 or merge_channels:
        _add_ch(viewer, zarr_group, lambda g: da.from_zarr(g), start_level, is_label, **kwargs)
    else:
        nchan = arr_shape[0]
        assert nchan <= 20, (f'More than 20 channels (nchan={nchan}) found for add_ome_zarr_array, are you sure '
                             f'this is not a mistake? The function takes in a merge_channels option for images '
                             f'that are not multi-channel; by default the first axis of the image will be '
                             f'treated as channel dimension!')
        name = kwargs.get('name', 'ome_zarr')
        for i in range(nchan):
            ch_name = f'{name}_ch{i}'
            kwargs['name'] = ch_name
            _add_ch(viewer, zarr_group, lambda g: da.take(da.from_zarr(g), indices=i, axis=0),
                    start_level, is_label, **kwargs)
