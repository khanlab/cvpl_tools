"""
This file provides visualization utilities for ome-zarr file, similar to napari-ome-zarr,
but includes features like displaying zip ome zarr files

In particular, this file provides interfaces to add ome zarr multiscale arrays to napari
viewer by wrapping the add_image or add_labels functions of napari viewer.
"""
from typing import Callable

import napari
try:
    from napari.layers import Layer
except OSError:
    Layer = None  # if no graphical display is needed and napari backend is missing, import will fail
import zarr
import dask.array as da
import cvpl_tools.ome_zarr.io as ome_io
import copy


# -------------Part 1: convenience functions, for adding ome zarr images using paths--------------


def preprocess_path_and_array_slices(path: str, kwargs: dict | None) -> tuple[str, dict]:
    """Preprocess array slices from path and update kwargs depending on what is found"""
    path, slices = ome_io.split_query_string(path)
    if kwargs is None:
        kwargs = dict()
    else:
        kwargs = copy.copy(kwargs)
        if slices is not None:
            to_append = kwargs.pop('array_slices', tuple())
            kwargs['array_slices'] = slices + to_append
    return path, kwargs


def group_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None,
                    merge_channels=True, kwargs=None, lbl_kwargs=None
                    ) -> Layer | list[Layer, ...]:
    """Add an ome zarr group to napari viewer from given group path.

    This is a combination of load_zarr_group_from_path() and add_ome_zarr_group() functions.
    """
    assert isinstance(merge_channels, bool)
    path, kwargs = preprocess_path_and_array_slices(path, kwargs)

    zarr_group = ome_io.load_zarr_group_from_path(path, 'r', use_zip)
    return group(viewer,
                 zarr_group,
                 merge_channels=merge_channels,
                 kwargs=kwargs,
                 lbl_kwargs=lbl_kwargs)


def subarray_from_path(viewer: napari.Viewer, path: str, use_zip: bool | None = None,
                       merge_channels=True, kwargs=None
                       ) -> Layer | list[Layer, ...]:
    """Add an ome zarr array to napari viewer from given array path.

    This is a combination of load_zarr_array_from_path() and add_ome_zarr_group() functions.
    """
    assert isinstance(merge_channels, bool)
    path, kwargs = preprocess_path_and_array_slices(path, kwargs)

    zarr_group = ome_io.load_zarr_group_from_path(path, 'r', use_zip)
    return subarray(viewer, zarr_group, merge_channels=merge_channels, **kwargs)


# ------------------------Part 2:adding ome zarr files using zarr group---------------------------


def group(viewer: napari.Viewer,
          zarr_group: zarr.hierarchy.Group,
          merge_channels=True,
          kwargs: dict = None,
          lbl_kwargs: dict = None
          ) -> Layer | list[Layer, ...]:
    """Add an ome zarr image (if exists) along with its labels (if exist) to viewer.

    Args:
        viewer: Napari viewer object to attach image to
        zarr_group: The zarr group that contains the ome zarr file
        merge_channels: If True, display the image as one layers instead of a layer per channel
        kwargs: dictionary, keyword arguments to be passed to viewer.add_image for root image
        lbl_kwargs: dictionary, keyword arguments to be passed to viewer.add_image for label images
    """
    assert isinstance(merge_channels, bool)
    assert isinstance(zarr_group, zarr.hierarchy.Group), (f'Expected zarr_group of type zarr.hierarchy.Group, got '
                                                          f'type(zarr_group)={type(zarr_group)}')

    if kwargs is None:
        kwargs = {}
    layer = None
    if '0' in zarr_group:
        layer = subarray(viewer,
                         zarr_group,
                         merge_channels=merge_channels,
                         **kwargs)
    if 'labels' in zarr_group:
        if lbl_kwargs is None:
            lbl_kwargs = {}
        lbls_group = zarr_group['labels']
        for group_key in lbls_group.group_keys():
            lbl_group = lbls_group[group_key]
            cur_layer = subarray(viewer, lbl_group, name=group_key, **lbl_kwargs)
            if layer is None:
                layer = cur_layer
    return layer


def _add_ch(viewer: napari.Viewer,
            zarr_group: zarr.hierarchy.Group,
            arr_from_group: Callable[[zarr.hierarchy.Group], da.Array],
            start_level: int = 0,
            is_label: object = False, **kwargs: object
            ) -> Layer:
    """Adds a channel or a set of channels to viewer"""
    multiscale = []
    while True:
        i = len(multiscale) + start_level
        i_str = str(i)
        if i_str in zarr_group:  # by ome zarr standard, image pyramid starts from 0 to NLEVEL - 1
            multiscale.append(arr_from_group(zarr_group[i_str]))
        else:
            break

    multiscale_flag = len(multiscale) > 1
    if not multiscale_flag:
        # newer version of napari seem to treat this as a single image with a width 1 axis in front, so unpack it
        multiscale = multiscale[0]
    if is_label:
        layer = viewer.add_labels(multiscale, multiscale=multiscale_flag, **kwargs)
    else:
        layer = viewer.add_image(multiscale, multiscale=multiscale_flag, **kwargs)
    return layer


def subarray(viewer: napari.Viewer,
             zarr_group: zarr.hierarchy.Group,
             merge_channels=True,
             start_level: int = 0,
             is_label=False,
             **kwargs
             ) -> Layer | list[Layer, ...]:
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
    assert isinstance(start_level, int)

    # catch bugs like accidentally used subarray() instead of subarray_from_path()
    assert isinstance(zarr_group, zarr.hierarchy.Group), (f'Expected zarr_group of type zarr.hierarchy.Group, got '
                                                          f'type(zarr_group)={type(zarr_group)}')

    kwargs = copy.copy(kwargs)
    str_ind = str(start_level)
    zarr_subgroup = zarr_group[str_ind]
    arr_shape = da.from_zarr(zarr_subgroup).shape
    ndim = len(arr_shape)

    if kwargs.get('array_slices') is not None:
        slices = kwargs.pop('array_slices')

        def arr_from_group(g):
            return da.from_array(g)[slices]
    else:
        def arr_from_group(g):
            return da.from_array(g)

    if ndim <= 2 or merge_channels:
        layer = _add_ch(viewer, zarr_group, arr_from_group, start_level, is_label, **kwargs)
    else:
        nchan = arr_shape[0]
        assert nchan <= 20, (f'More than 20 channels (nchan={nchan}) found for add_ome_zarr_array, are you sure '
                             f'this is not a mistake? The function takes in a merge_channels option for images '
                             f'that are not multi-channel; by default the first axis of the image will be '
                             f'treated as channel dimension!')
        name = kwargs.get('name', 'ome_zarr')
        layer = []
        for i in range(nchan):
            ch_name = f'{name}_ch{i}'
            kwargs['name'] = ch_name
            cur_layer = _add_ch(viewer, zarr_group, lambda g: da.take(arr_from_group(g), indices=i, axis=0),
                                start_level, is_label, **kwargs)
            layer.append(cur_layer)
    return layer
