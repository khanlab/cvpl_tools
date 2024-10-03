"""
Add support for writing dask array to ome zarr zip file, and avoid repeated computation as
discussed in https://github.com/ome/ome-zarr-py/issues/392
"""
from typing import Sequence

import ome_zarr
import ome_zarr.io
import cvpl_tools.ome_zarr.ome_zarr_writer_patched as writer
import os
import shutil
import zarr
import dask.array as da
import numpy as np
import urllib.parse
import numcodecs


# ----------------------------------Part 1: utilities---------------------------------------


def encode_path_as_filename(path):
    encoded_path = urllib.parse.quote(path, safe='')
    return encoded_path


def load_zarr_group_from_path(path: str,
                              mode: str | None = None,
                              use_zip: bool | None = None,
                              level: int | None = None) -> zarr.Group:
    """Loads either a zarr folder or zarr zip file into a zarr group.

    Args:
        path: path to the zarr folder or zip to be opened
        mode: file open mode e.g. 'r', only pass this if the file is a zip file
        use_zip: if True, treat path as a zip; if False, treat path as a folder; if None, use path to determine
            file type
        level: If None (default), load the entire ome zarr; if an int is provided, load the corresponding level
            in the ome zarr array
    Returns:
        the opened zarr group
    """
    if use_zip is None:
        use_zip = path.endswith('.zip')
    if use_zip:
        store = zarr.ZipStore(path, mode=mode)
    else:
        store = zarr.DirectoryStore(path)
    zarr_group = zarr.open(store, mode=mode)
    if level is not None:
        assert isinstance(level, int)
        zarr_group = zarr_group[str(level)]

    return zarr_group


def split_query_string(path: str) -> tuple[str, tuple[slice, ...] | None]:
    """Split a url to path and an optional query dict

    Examples:
        split_query_string('path/to/file') -> ('path/to/file', None)
        split_query_string('path/to/file?slices=[3, :2]') -> ('path/to/file', tuple(3, slice(None, 2)))

    Args:
        path: url to be splitted

    Returns:
        splitted path and query string; if path does not contain a query string, then the second returned arg is None.
    """
    if '?' in path:
        # parse the query string at the end
        path, query = path.split('?')
        query = urllib.parse.parse_qs(query)

        slices = query['slices'][0]
        assert slices[0] == '[' and slices[-1] == ']', f'Slices must start with [ and end with ], got slices={slices}'
        slices = slices[1:-1].split(',')
        result = []
        for comp in slices:
            if ':' in comp:
                result.append(slice(*[int(x) if x else None for x in comp.split(':')]))
            else:
                result.append(int(comp))
        slices = tuple(result)
    else:
        slices = None
    return path, slices


def load_dask_array_from_path(path: str,
                              mode: str | None = None,
                              use_zip: bool | None = None,
                              level: int | None = None) -> da.Array:
    """Loads either a zarr folder or zarr zip file into a zarr group.

    Compared to load_zarr_group_from_path, this function allows specifying which slice and channel to read using
    a query string syntax (idea thanks to Davis Bennett in the thread
    https://forum.image.sc/t/loading-only-one-channel-from-an-ome-zarr/97798)

    Examples:
        load_dask_array_from_path('C://path/to/image.ome.zarr?slices=[1]')  # use channel 1
        load_dask_array_from_path('/path/to/image.ome.zarr?slices=[:, :100]')  # all channels, 0-99 on the second axis

    Args:
        path: path to the zarr folder or zip to be opened
        mode: file open mode e.g. 'r', only pass this if the file is a zip file
        use_zip: if True, treat path as a zip; if False, treat path as a folder; if None, use path to determine
            file type
        level: If None (default), load the entire ome zarr; if an int is provided, load the corresponding level
            in the ome zarr array
    Returns:
        the opened zarr group
    """
    path, slices = split_query_string(path)

    zarr_group = load_zarr_group_from_path(path, mode, use_zip, level)
    arr = da.from_array(zarr_group)

    if slices is not None:
        arr = arr[slices]

    return arr


# --------------------------------Part 2: write image-------------------------------------


def _get_coord_transform_yx_for_write(ndim, MAX_LAYER) -> list:  # down sampling by 2 each layer
    assert ndim >= 2, 'Can not 2d down sample an image of dimension less than 2d!'
    coordinate_transformations = []
    for layer in range(MAX_LAYER + 1):
        coordinate_transformations.append(
            [{'scale': [1.] * (ndim - 2) + [(2 ** layer) * 1., (2 ** layer) * 1.],  # image-pyramids in XY only
              'type': 'scale'}])
    return coordinate_transformations


def _get_axes_for_write(ndim: int) -> list:
    axes = ['x', 'y', 'z', 'c']  # usually ['x', 'y', 'z', 'c', 't'] but we don't need that for our use
    return list(reversed(axes[:ndim]))


def write_ome_zarr_image_direct(zarr_group: zarr.Group,
                                da_arr: da.Array | None = None,
                                lbl_arr: da.Array | None = None,
                                lbl_name: str | None = None,
                                MAX_LAYER: int = 3,
                                storage_options: dict = None,
                                lbl_storage_options: dict = None):
    """Direct write of dask array to target ome zarr group (can not be a zip)

    Args:
        zarr_group: The output zarr group which we will write an ome zarr image to
        da_arr: (dask array) The content of the image to write
        lbl_arr: If provided, this is the array to write at zarr_group['labels'][lbl_name]
        lbl_name: name of the label array subgroup
        MAX_LAYER: The maximum layer of down sampling; starting at layer=0
        storage_options: options for storing the image
        lbl_storage_options: options for storing the labels
    """
    if storage_options is None:
        compressor = numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
        storage_options = dict(
            dimension_separator='/',
            compressor=compressor
        )
    if da_arr is not None:
        # assert the group is empty, since we are writing a new group
        for mem in zarr_group:
            raise ValueError('ZARR group is Non-empty, please remove the original ZARR before running the program to '
                             f'create synthetic data. ZARR group: {zarr_group}')

    scaler = ome_zarr.scale.Scaler(max_layer=MAX_LAYER, method='nearest')
    if da_arr is not None:
        writer.write_image(image=da_arr,
                           group=zarr_group,
                           scaler=scaler,
                           coordinate_transformations=_get_coord_transform_yx_for_write(da_arr.ndim, MAX_LAYER),
                           storage_options=storage_options,
                           axes=_get_axes_for_write(da_arr.ndim))

    if lbl_arr is not None:
        assert lbl_name is not None, ('ERROR: Please provide lbl_name along when writing labels '
                                      '(lbl_arr is not None)')

        # we could just use ['c', 'z', 'y', 'x'], however, napari ome zarr can't handle channel types but only space
        # type axes. So we need to fall back to manual definition, avoid 'c' which defaults to a channel type
        lbl_axes = [{'name': ch, 'type': 'space'} for ch in _get_axes_for_write(lbl_arr.ndim)]

        if lbl_storage_options is None:
            compressor = numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
            lbl_storage_options = dict(compressor=compressor)
        writer.write_labels(labels=lbl_arr,
                            group=zarr_group,
                            scaler=scaler,
                            name=lbl_name,
                            coordinate_transformations=_get_coord_transform_yx_for_write(lbl_arr.ndim, MAX_LAYER),
                            storage_options=lbl_storage_options,
                            axes=lbl_axes)
        # ome_zarr.writer.write_label_metadata(group=g,
        #                      name=f'/labels/{lbl_name}',
        #                      properties=properties)


def write_ome_zarr_image(ome_zarr_path: str,
                         tmp_path: str = None,  # provide this if making a zip file
                         da_arr: da.Array | None = None,
                         lbl_arr: da.Array | None = None,
                         lbl_name: str | None = None,
                         make_zip: bool | None = None,
                         MAX_LAYER: int = 0,
                         logging=False,
                         storage_options: dict = None,
                         lbl_storage_options: dict = None):
    """Write dask array as an ome zarr

    For writing to zip file: due to dask does not directly support write to zip file, we instead create a temp ome zarr
    output and copy it into a zip after done. This is why tmp_path is required if make_zip=True

    Args:
        ome_zarr_path: The path to target ome zarr folder, or ome zarr zip folder if make_zip is True
        tmp_path: temporary files will be stored under this,
        da_arr: If provided, this is the array to write at {ome_zarr_path}
        lbl_arr: If provided, this is the array to write at {ome_zarr_path}/labels/{lbl_name}
        lbl_name: name of the folder of the label array
        make_zip: bool, if True the output is a zip; if False a folder; if None, then determine based on file suffix
        MAX_LAYER: The maximum layer of down sampling; starting at layer=0
        logging: If true, print message when job starts and ends
        storage_options: options for storing the image
        lbl_storage_options: options for storing the labels
    """
    if tmp_path is not None:
        os.makedirs(tmp_path, exist_ok=True)

    if make_zip is None:
        make_zip = ome_zarr_path.endswith('.zip')
    if logging:
        print('Writing Folder.')
    if make_zip:
        encode_name = encode_path_as_filename(ome_zarr_path)
        folder_ome_zarr_path = f'{tmp_path}/{encode_name}.temp'
        if os.path.exists(folder_ome_zarr_path):
            shutil.rmtree(folder_ome_zarr_path)
    else:
        folder_ome_zarr_path = ome_zarr_path

    store = ome_zarr.io.parse_url(folder_ome_zarr_path, mode='w').store
    g = zarr.group(store)
    write_ome_zarr_image_direct(g, da_arr, lbl_arr, lbl_name, MAX_LAYER=MAX_LAYER,
                                storage_options=storage_options,
                                lbl_storage_options=lbl_storage_options)
    if logging:
        print('Folder is written.')
    store.close()

    if make_zip:
        if logging:
            print('Writing zip.')
        store = ome_zarr.io.parse_url(folder_ome_zarr_path,
                                      mode='r').store  # same folder but this time we open it in read mode
        g = zarr.group(store)
        with zarr.ZipStore(ome_zarr_path, mode='w') as target_store:
            target_g = zarr.group(target_store)
            zarr.copy_all(g, target_g)

        if logging:
            print('Zip is written.')
        # remove the temp folder
        shutil.rmtree(folder_ome_zarr_path)


# ------------------------------Part 3: synthetic dataset-----------------------------------


def generate_synthetic_dataset(ome_zarr_path: str,
                               tmp_path: str = None,
                               arr_sz: tuple = (2, 224, 1600, 2048),
                               chunks: tuple = (1, 1, 1024, 1024),
                               make_zip: bool | None = None,
                               MAX_LAYER=0) -> zarr.Group:
    """Generate a 4d synthetic test ome zarr image physically stored in ome_zarr_path.

    Args:
        ome_zarr_path: Where to store the generated test image
        tmp_path: The temporary path for write if making a zip file image
        arr_sz: The size of the synthetic image
        chunks: The chunk size of the image
        make_zip: If True, make a physical zip storage of instead of a directory of ome zarr image
        MAX_LAYER: The maximum down sampling layer
    Returns:
        An opened zarr group of the newly generated synthetic image
    """
    arr: da.Array = da.zeros(arr_sz, dtype=np.uint16, chunks=chunks)

    def process_block(block, block_info=None):
        if block_info is not None:
            # calculate (global) indices array for each pixel
            block_slice = block_info[0]['array-location']
            indices = np.indices(block.shape)
            for dim in range(indices.shape[0]):
                indices[dim] += block_slice[dim][0]
        else:
            return np.zeros(block.shape, dtype=np.uint16)
        # now, create balls in the block
        sq = np.zeros(block.shape)  # distance squared
        for dim in range(1, indices.shape[0]):  # every dim except channel dim which does not have distance
            sq += np.power(indices[dim], 2.) * .0002
        for dim in range(1, indices.shape[0]):  # every dim except channel dim which does not have distance
            indices[dim] %= 32
            sq += np.power(indices[dim] - 15.5, 2.)
        im = np.array(np.clip(1200. - sq * 15., 0., 1200.), dtype=np.uint16)
        return im

    arr = arr.map_blocks(process_block, dtype=np.uint16)
    write_ome_zarr_image(ome_zarr_path, tmp_path, da_arr=arr, make_zip=make_zip, MAX_LAYER=MAX_LAYER)
    group = load_zarr_group_from_path(ome_zarr_path, mode='r', use_zip=make_zip)
    return group


def dask_checkerboard(chunks: Sequence[Sequence[int]]) -> da.Array:
    """Create a synthetic checkerboard pattern dask image

    Args:
        chunks: The chunks of the checkerboard, from which shape and result array chunk sizes will
            be calculated from

    Returns:
        A dask.Array of checkerboard patterns (0=black and 1=white) of type np.uint8; top-left is
            black
    """
    shape = tuple(sum(c) for c in chunks)
    checkerboard: da.Array = da.zeros(shape, chunks=chunks, dtype=np.uint8)

    def map_fn(block, block_info=None):
        block_index = block_info[0]['chunk-location']
        color = sum(block_index) % 2
        return np.ones_like(block) * color

    checkerboard = checkerboard.map_blocks(map_fn, meta=np.zeros(tuple(), dtype=np.uint8), dtype=np.uint8)
    return checkerboard

