import cvpl_tools.examples.mousebrain_processing as mp
import napari
import tifffile
import cvpl_tools.ome_zarr.io as ome_io
import numpy as np

from cvpl_tools.fsspec import RDirFileSystem


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
    import cvpl_tools.ome_zarr.napari.add as nozadd

    subject = mp.get_subject(SUBJECT_ID)

    viewer = napari.Viewer(ndisplay=2)

    print(subject.SECOND_DOWNSAMPLE_PATH)
    print(subject.SECOND_DOWNSAMPLE_CORR_PATH)
    im = ome_io.load_dask_array_from_path(subject.SECOND_DOWNSAMPLE_PATH, mode='r',
                                                     level=0).compute()
    corr = ome_io.load_dask_array_from_path(subject.SECOND_DOWNSAMPLE_CORR_PATH, mode='r',
                                                     level=0).compute()
    viewer.add_image(corr, name='corr', contrast_limits=[0., subject.MAX_THRESHOLD])
    nozadd.group_from_path(viewer, subject.THIRD_DOWNSAMPLE_BIAS_PATH, kwargs=dict(
        name='bias',
    ))
    viewer.add_image(im, name='im', visible=False, contrast_limits=[0., subject.MAX_THRESHOLD])

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
    import cvpl_tools.nnunet.lightsheet_preprocess as ci

    subject = mp.get_subject(SUBJECT_ID)

    display_shape = ome_io.load_dask_array_from_path(f'{subject.COILED_CACHE_DIR_PATH}/input_im/dask_im', mode='r', level=0).shape

    viewer = napari.Viewer(ndisplay=2)
    nozadd.group_from_path(viewer, f'{subject.COILED_CACHE_DIR_PATH}/input_im/dask_im',
                           kwargs=dict(
                               name='im',
                               visible=False,
                               **ci.calc_tr_sc_args(voxel_scale=(1,) * 3, display_shape=display_shape),
                               contrast_limits=[0, 1],
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
    import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
    import cvpl_tools.nnunet.lightsheet_preprocess as lightsheet_preprocess
    import asyncio

    subject = mp.get_subject(SUBJECT_ID)

    if not RDirFileSystem(subject.FIRST_DOWNSAMPLE_CORR_PATH).exists(''):
        first_downsample = ome_io.load_dask_array_from_path(subject.FIRST_DOWNSAMPLE_PATH, mode='r', level=0)
        third_downsample_bias = ome_io.load_dask_array_from_path(subject.THIRD_DOWNSAMPLE_BIAS_PATH, mode='r', level=0)
        first_downsample_bias = dask_ndinterp.scale_nearest(third_downsample_bias, scale=(4, 4, 4),
                                                             output_shape=first_downsample.shape,
                                                             output_chunks=(4, 4096, 4096)).persist()

        first_downsample_corr = lightsheet_preprocess.apply_bias(first_downsample, (1,) * 3, first_downsample_bias, (1,) * 3)
        asyncio.run(ome_io.write_ome_zarr_image(subject.FIRST_DOWNSAMPLE_CORR_PATH, da_arr=first_downsample_corr, MAX_LAYER=2))
    first_downsample_corr = ome_io.load_dask_array_from_path(subject.FIRST_DOWNSAMPLE_CORR_PATH, mode='r', level=0).compute()
    print('second downsample corrected image done')

    viewer = napari.Viewer(ndisplay=2)
    canvas_shape = ome_io.load_dask_array_from_path(subject.SECOND_DOWNSAMPLE_PATH, mode='r', level=0).shape
    annotate.annotate(viewer,
                      first_downsample_corr,
                      canvas_path=subject.NNUNET_OUTPUT_TIFF_PATH,
                      ndownsample_level=(2,) * 3)
    viewer.show(block=True)


if __name__ == '__main__':
    for ID in ('M7A1Te4Blaze',):
        if ID in ('M4A2Te3Blaze', 'o22', 'o23'):
            continue
        print(f'Starting inspection on subject {ID}')

        inspect_corrected(ID)
        # annotate_neg_mask(ID)
        # inspect_os(ID)


