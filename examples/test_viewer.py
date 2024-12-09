import napari
import cvpl_tools.tools.fs as tlfs
import cvpl_tools.ome_zarr.napari.add as nozadd

if __name__ == '__main__':
    viewer = napari.Viewer(ndisplay=2)
    nozadd.group_from_path(viewer, 'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/input_im/dask_im',
                           kwargs=dict(
                               name='im',
                               visible=False
                           ))
    nozadd.group_from_path(viewer, 'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/per_pixel_multiplier/dask_im',
                           kwargs=dict(
                               name='ppm',
                               is_label=False,
                               visible=False
                           ))
    nozadd.group_from_path(viewer,
                           'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/GLOBAL_LABEL/os/global_os/dask_im',
                           kwargs=dict(
                               name='os',
                               is_label=True,
                               visible=False
                           ))
    # nozadd.group_from_path(viewer, 'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/GLOBAL_LABEL/os/global_os/dask_im',
    #                        kwargs=dict(
    #                            name='os'
    #                        ))
    viewer.show(block=True)
