import napari
import cvpl_tools.tools.fs as tlfs
import cvpl_tools.ome_zarr.napari.add as nozadd


if __name__ == '__main__':
    viewer = napari.Viewer(ndisplay=2)
    nozadd.group_from_path(viewer, 'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-F4A1Te3/micr/sub-F4A1Te3_sample-brain_acq-blaze4x_SPIM.ome.zarr', kwargs=dict(
        name='im'
    ))
    # nozadd.group_from_path(viewer, 'gcs://khanlab-scratch/tmp/CacheDirectory_F4A1Te3Blaze/GLOBAL_LABEL/os/global_os/dask_im',
    #                        kwargs=dict(
    #                            name='os'
    #                        ))
    viewer.show(block=True)


