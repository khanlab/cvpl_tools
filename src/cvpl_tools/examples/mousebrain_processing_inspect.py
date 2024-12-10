import cvpl_tools.examples.mousebrain_processing as mp
import napari
import tifffile
import cvpl_tools.ome_zarr.io as ome_io
import numpy as np


def inspect_negmask(SUBJECT_ID):
    # second downsample negmask vs. second downsample original image; inspect local image
    import cvpl_tools.ome_zarr.napari.add as nozadd

    subject = mp.get_subject(SUBJECT_ID)

    viewer = napari.Viewer(ndisplay=2)
    nozadd.group_from_path(viewer, subject.SECOND_DOWNSAMPLE_PATH, kwargs=dict(
        name='im',
        visible=False
    ))
    nozadd.group_from_path(viewer, subject.SECOND_DOWNSAMPLE_CORR_PATH, kwargs=dict(
        name='corr',
    ))
    neg_mask = tifffile.imread(subject.NNUNET_OUTPUT_TIFF_PATH)
    viewer.add_labels(neg_mask, name='neg_mask')
    viewer.show(block=True)


def inspect_corrected(SUBJECT_ID):
    import magicgui

    subject = mp.get_subject(SUBJECT_ID)

    viewer = napari.Viewer(ndisplay=2)

    im = ome_io.load_dask_array_from_path(subject.SECOND_DOWNSAMPLE_PATH, mode='r',
                                                     level=0).compute()
    corr = ome_io.load_dask_array_from_path(subject.SECOND_DOWNSAMPLE_CORR_PATH, mode='r',
                                                     level=0).compute()
    viewer.add_image(corr, name='corr')
    viewer.add_image(im, name='im', visible=False)

    @magicgui.magicgui(value={'max': 100000})
    def image_arithmetic(
            layerA: 'napari.types.ImageData',
            value: float
    ) -> 'napari.types.ImageData':
        """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
        if layerA is not None:
            arr = np.zeros(layerA.shape, dtype=np.uint8)
            arr[:] = layerA > value
            viewer.add_labels(arr, name='result')

    viewer.window.add_dock_widget(image_arithmetic)

    viewer.show(block=True)


def inspect_os(SUBJECT_ID):
    import cvpl_tools.ome_zarr.napari.add as nozadd
    import cvpl_tools.nnunet.current_im as ci

    subject = mp.get_subject(SUBJECT_ID)

    display_shape = ome_io.load_dask_array_from_path(f'{subject.COILED_CACHE_DIR_PATH}/input_im/dask_im', mode='r', level=0).shape

    viewer = napari.Viewer(ndisplay=2)
    nozadd.group_from_path(viewer, f'{subject.COILED_CACHE_DIR_PATH}/input_im/dask_im',
                           kwargs=dict(
                               name='im',
                               visible=False,
                               **ci.calc_tr_sc_args(voxel_scale=(1,) * 3, display_shape=display_shape)
                           ))
    nozadd.group_from_path(viewer,
                           f'{subject.COILED_CACHE_DIR_PATH}/per_pixel_multiplier/dask_im',
                           kwargs=dict(
                               name='ppm',
                               is_label=False,
                               visible=False,
                               **ci.calc_tr_sc_args(voxel_scale=(4, 8, 8), display_shape=display_shape)
                           ))
    nozadd.group_from_path(viewer,
                           f'{subject.COILED_CACHE_DIR_PATH}/GLOBAL_LABEL/os/global_os/dask_im',
                           kwargs=dict(
                               name='os',
                               is_label=True,
                               visible=False,
                               **ci.calc_tr_sc_args(voxel_scale=(1,) * 3, display_shape=display_shape)
                           ))
    viewer.show(block=True)


def annotate_neg_mask(SUBJECT_ID):
    import cvpl_tools.nnunet.annotate as annotate

    subject = mp.get_subject(SUBJECT_ID)

    im_annotate = ome_io.load_dask_array_from_path(subject.FIRST_DOWNSAMPLE_PATH, mode='r', level=0).compute()
    viewer = napari.Viewer(ndisplay=2)
    annotate.annotate(viewer,
                      im_annotate,
                      annotation_folder=subject.SUBJECT_FOLDER,
                      canvas_path=subject.NNUNET_OUTPUT_TIFF_PATH,
                      SUBJECT_ID=subject.SUBJECT_ID)
    viewer.show(block=True)


if __name__ == '__main__':
    inspect_corrected('M4A2Te3Blaze')
    annotate_neg_mask('M4A2Te3Blaze')
    inspect_os('M4A2Te3Blaze')


