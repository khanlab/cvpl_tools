"""
Add support for writing dask array to ome zarr zip file, and avoid repeated computation as
discussed in https://github.com/ome/ome-zarr-py/issues/392
"""

import ome_zarr
import cvpl_tools.ome_zarr.ome_zarr_writer_patched as writer
import os
import shutil
import zarr
import dask.array as da
import numpy as np
from ome_zarr.io import parse_url
import urllib.parse


# ----------------------------------Part 1: utilities---------------------------------------


def encode_path_as_filename(path):
    encoded_path = urllib.parse.quote(path, safe='')
    return encoded_path


def load_zarr_group_from_path(path: str, mode=None, use_zip: bool | None = None) -> zarr.Group:
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
    else:
        store = zarr.DirectoryStore(path)
    zarr_group = zarr.open(store, mode=mode)
    return zarr_group


def cache_image(arr: da.Array, location: str):
    """cache a dask image on disk, and return a new 'r' mode opened dask array of it

    Args:
        arr: the dask array to save
        location: the location to save the dask array (new folder to be created)

    Returns:
        A dask.Array object read from the saved location; array with the same content as the input
    """
    store = parse_url(location, mode='w').store
    g = zarr.group(store)
    write_ome_zarr_image_direct(g, arr, MAX_LAYER=0)
    zarr_group = zarr.open(location, mode='r')
    zarr_subgroup = zarr_group['0']
    arr_read = da.from_zarr(zarr_subgroup)
    return arr_read


# --------------------------------Part 2: write image-------------------------------------


def write_ome_zarr_image_direct(zarr_group: zarr.Group,
                                da_arr: da.Array | None = None,
                                lbl_arr: da.Array | None = None,
                                lbl_name: str | None = None,
                                MAX_LAYER: int = 3):
    """Direct write of dask array to target ome zarr group."""
    if da_arr is not None:
        # assert the group is empty, since we are writing a new group
        for mem in zarr_group:
            raise ValueError('ZARR group is Non-empty, please remove the original ZARR before running the program to '
                             f'create synthetic data. ZARR group: {zarr_group}')

    scaler = ome_zarr.scale.Scaler(max_layer=MAX_LAYER, method='nearest')
    coordinate_transformations = []
    for layer in range(MAX_LAYER + 1):
        coordinate_transformations.append(
            [{'scale': [1., 1., (2 ** layer) * 1., (2 ** layer) * 1.],  # image-pyramids in XY only
              'type': 'scale'}])
    if da_arr is not None:
        writer.write_image(image=da_arr,
                           group=zarr_group,
                           scaler=scaler,
                           coordinate_transformations=coordinate_transformations,
                           storage_options={'dimension_separator': '/'},
                           axes=['c', 'z', 'y', 'x'])

    # we could just use ['c', 'z', 'y', 'x'], however, napari ome zarr can't handle channel types but only space
    # type axes. So we need to fall back to manual definition, avoid 'c' which defaults to a channel type
    lbl_axes = [{'name': ch, 'type': 'space'} for ch in ['c', 'z', 'y', 'x']]
    if lbl_arr is not None:
        assert lbl_name is not None, ('ERROR: Please provide lbl_name along when writing labels '
                                      '(lbl_arr is not None)')
        import numcodecs
        compressor = numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
        writer.write_labels(labels=lbl_arr,
                            group=zarr_group,
                            scaler=scaler,
                            name=lbl_name,
                            coordinate_transformations=coordinate_transformations,
                            storage_options=dict(compressor=compressor),
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
                         MAX_LAYER=3,
                         logging=False):
    """Write dask array as an ome zarr

    Args
        ome_zarr_path - The path to target ome zarr folder, or ome zarr zip folder if make_zip is True
        tmp_path - temporary files will be stored under this,
        da_arr - If provided, this is the array to write at {ome_zarr_path}
        lbl_arr - If provided, this is the array to write at {ome_zarr_path}/labels/{lbl_name}
        lbl_name - name of the folder of the label array
        make_zip - bool, if True the output is a zip; if False a folder; if None, then determine based on file suffix
        MAX_LAYER - The maximum layer of down sampling; starting at layer=0
        logging - If true, print message when job starts and ends
    """
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
    write_ome_zarr_image_direct(g, da_arr, lbl_arr, lbl_name, MAX_LAYER=MAX_LAYER)
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
                               tmp_path: str,
                               arr_sz: tuple = (2, 224, 1600, 2048),
                               chunks: tuple = (1, 1, 1024, 1024),
                               make_zip: bool | None = None,
                               MAX_LAYER=0):
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
    os.makedirs(tmp_path, exist_ok=True)
    write_ome_zarr_image(ome_zarr_path, tmp_path, da_arr=arr, make_zip=make_zip, MAX_LAYER=MAX_LAYER)
    group = load_zarr_group_from_path(f'{ome_zarr_path}/0', mode='r', use_zip=make_zip)
    return group
