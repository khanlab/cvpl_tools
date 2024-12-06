def main():
    import cvpl_tools.ome_zarr.napari.add as nozadd
    import napari
    import cvpl_tools.ome_zarr.io as ome_io

    viewer = napari.Viewer(ndisplay=2)
    groups = (('gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/GLOBAL_LABEL/os/global_os/dask_im', 'os', True),
        ('gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/input_im/dask_im', 'im', False))
    for group, name, is_label in groups:
        shape = ome_io.load_dask_array_from_path(group, level=0).shape
        print(shape)
    exit(0)
    for group, name, is_label in groups:
        nozadd.group_from_path(viewer,
                               'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/GLOBAL_LABEL/os/global_os/dask_im',
                               kwargs=dict(name=name, is_label=is_label))
    # viewer.add_image(bias, name='bias')
    # viewer.add_image(arr, name='im_mini')
    # viewer.add_image(arr / bias, name='im_mini_corr')
    # nozadd.group_from_path(viewer,
    #                        'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/bias.ome.zarr',
    #                        kwargs=dict(name='bias2'))
    # nozadd.group_from_path(viewer,
    #                        'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/im_mini.ome.zarr',
    #                        kwargs=dict(name='im2'))
    # nozadd.group_from_path(viewer, 'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/im_corrected.ome.zarr', kwargs=dict(name='im_corr'))
    viewer.show(block=True)


if __name__ == '__main__':
    main()

