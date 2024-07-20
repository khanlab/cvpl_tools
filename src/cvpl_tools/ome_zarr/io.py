"""
Add support for writing dask array to ome zarr zip file, and avoid repeated computation as
discussed in https://github.com/ome/ome-zarr-py/issues/392
"""


import ome_zarr
import os
import shutil
import zarr


def write_da_as_ome_zarr_direct(zarr_group: zarr.Group, da_arr=None, lbl_arr=None, lbl_name=None, MAX_LAYER=3):
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
        ome_zarr.writer.write_image(image=da_arr,
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
        ome_zarr.writer.write_labels(labels=lbl_arr,
                     group=zarr_group,
                     scaler=scaler,
                     name=lbl_name,
                     coordinate_transformations=coordinate_transformations,
                     storage_options=dict(compressor=compressor),
                     axes=lbl_axes)
        # ome_zarr.writer.write_label_metadata(group=g,
        #                      name=f'/labels/{lbl_name}',
        #                      properties=properties)


def write_da_as_ome_zarr(ome_zarr_path, tmp_path, da_arr=None,
                         lbl_arr=None, lbl_name=None, make_zip=False, MAX_LAYER=3):
    """Write dask array as an ome zarr

    Args
        ome_zarr_path - The path to target ome zarr folder, or ome zarr zip folder if make_zip is True
    """
    print('Writing Folder.')
    if make_zip:
        folder_ome_zarr_path = f'{tmp_path}/{lbl_name}_tmp0'
        if os.path.exists(folder_ome_zarr_path):
            shutil.rmtree(folder_ome_zarr_path)
    else:
        folder_ome_zarr_path = ome_zarr_path
    store = ome_zarr.io.parse_url(folder_ome_zarr_path, mode='w').store
    g = zarr.group(store)
    if da_arr is not None:
        path1 = f'{tmp_path}/{lbl_name}_tmp1'
        if os.path.exists(path1):
            shutil.rmtree(path1)
        da_arr = cache_image(da_arr, path1)
    if lbl_arr is not None:
        path2 = f'{tmp_path}/{lbl_name}_tmp2'
        if os.path.exists(path2):
            shutil.rmtree(path2)
        lbl_arr = cache_image(lbl_arr, path2)
    write_da_as_ome_zarr_direct(g, da_arr, lbl_arr, lbl_name, MAX_LAYER=MAX_LAYER)
    print('Folder is written.')
    store.close()
    if make_zip:
        print('Writing zip.')
        store = ome_zarr.io.parse_url(folder_ome_zarr_path, mode='r').store  # same folder but this time we open it in read mode
        g = zarr.group(store)
        target_store = zarr.ZipStore(ome_zarr_path, mode='w')
        target_g = zarr.group(target_store)
        zarr.copy_all(g, target_g)
        store.close()
        print('Zip is written.')
