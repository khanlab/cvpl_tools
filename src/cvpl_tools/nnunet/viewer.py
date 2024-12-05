from cvpl_tools.nnunet.n4 import bias_correct


def main():
    import cvpl_tools.ome_zarr.napari.add as nozadd
    import napari
    import cvpl_tools.ome_zarr.io as ome_io
    import cvpl_tools.nnunet.n4 as n4

    arr = ome_io.load_dask_array_from_path('C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/im_mini.ome.zarr',
                                           mode='r', level=0).compute()
    bias = n4.obtain_bias(arr)
    print(arr.shape, bias.shape)

    viewer = napari.Viewer(ndisplay=2)
    # nozadd.group_from_path(viewer,
    #                        'gcs://khanlab-scratch/tmp/CacheDirectoryBlaze_F1A1Te4Blaze/GLOBAL_LABEL/os/global_os/dask_im',
    #                        kwargs=dict(name='os', is_label=True))
    # nozadd.group_from_path(viewer,
    #                        'gcs://khanlab-scratch/tmp/CacheDirectoryBlaze_F1A1Te4Blaze/input_im/dask_im',
    #                        kwargs=dict(name='im'))
    viewer.add_image(bias, name='bias')
    viewer.add_image(arr, name='im_mini')
    viewer.add_image(arr / bias, name='im_mini_corr')
    nozadd.group_from_path(viewer,
                           'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/bias.ome.zarr',
                           kwargs=dict(name='bias2'))
    nozadd.group_from_path(viewer,
                           'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/im_mini.ome.zarr',
                           kwargs=dict(name='im2'))
    nozadd.group_from_path(viewer, 'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_F4A1Te3Blaze/im_corrected.ome.zarr', kwargs=dict(name='im_corr'))
    viewer.show(block=True)


if __name__ == '__main__':
    main()

