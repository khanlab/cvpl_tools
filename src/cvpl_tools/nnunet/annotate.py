import os.path

import monai_wrapper.tools.global_vars as global_vars
import monai_wrapper.tools.current_im as current_im
import napari
import numpy as np
import tifffile
import cvpl_tools.im.algs.dask_ndinterp as ndinterp
import cvpl_tools.ome_zarr.io as ome_io
import cvpl_tools.ome_zarr.napari.add as nozadd
import dask.array as da
import asyncio

from monai_wrapper.nnunet.triplanar import dice_on_volume_pair

ANNOTATION_FOLDER = 'C:/Users/than83/Documents/progtools/datasets/annotated'
CANVAS_PATH = f'{ANNOTATION_FOLDER}/canvas_{global_vars.SUBJECT_ID}.tiff'
CANVAS_REF_PATH = f'{ANNOTATION_FOLDER}/canvas_{global_vars.SUBJECT_ID}_ref.tiff'

TARGET_FOLDER = f'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_{global_vars.SUBJECT_ID}'


def get_canvas():
    if os.path.exists(CANVAS_PATH):
        READ_PATH = CANVAS_PATH
    elif os.path.exists(CANVAS_REF_PATH):
        READ_PATH = CANVAS_REF_PATH
    else:
        READ_PATH = None

    if READ_PATH is None:
        canvas = np.zeros(current_im.CANVAS_SHAPE, dtype=np.uint8)
    else:
        canvas = tifffile.imread(READ_PATH).astype(np.uint8)
        assert np.dtype(canvas.dtype) == np.uint8, f'{canvas.dtype}'
    return canvas


def annotate():
    """
    usage:
    import monai_wrapper.tools.annotate as ann
    ann.annotate()
    """
    import magicgui

    viewer = napari.Viewer(ndisplay=2)
    im_layer = viewer.add_image(current_im.im_annotate, name='im', **current_im.calc_tr_sc_args())

    canvas = get_canvas()
    canvas_layer = viewer.add_labels(canvas, name='canvas', **current_im.calc_tr_sc_args(voxel_scale=current_im.CANVAS_VOXEL_SCALE))

    for path in tuple(
            # 'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_o22_ref.tiff',
            # 'Cache_500epoch_Run20241108/dir_cache_predict/0.npy',
            # 'Cache_500epoch_triplanar_Run20241113/dir_cache_predict/0_yx.tiff',
            # 'Cache_500epoch_triplanar_Run20241113/dir_cache_predict/0_xz.tiff',
            # 'Cache_500epoch_triplanar_Run20241113/dir_cache_predict/0_zy.tiff',
            # 'Cache_500epoch_triplanar_Run20241113/dir_cache_predict/0.tiff',
            # 'Cache_250epoch_tri5f_Run20241114/dir_cache_predict/0_o23.tiff',
            # 'Cache_250epoch_tri5f_Run20241114/dir_cache_predict/0_5folds.tiff',
            # 'Cache_250epoch_Run20241120/dir_cache_predict/0_o23.tiff',
            # 'Cache_250epoch_tri5f_Run20241114/dir_cache_predict/0_o24.tiff',
            # 'Cache_250epoch_Run20241120/dir_cache_predict/0_o24.tiff',
            # 'Cache_250epoch_Run20241120/dir_cache_predict/0_o24oldBlaze.tiff',
    ):
        if path.startswith('C:'):
            PRED_PATH = path
            pred = tifffile.imread(PRED_PATH)
        else:
            PRED_PATH = f'C:/Users/than83/Documents/progtools/datasets/nnunet/{path}'
            if path.endswith('.npy'):
                pred = np.load(PRED_PATH) > 0.
            else:
                pred = tifffile.imread(PRED_PATH)
        print(pred.shape, canvas.shape)
        print (f'dice for {path}:', dice_on_volume_pair(canvas, pred))
        pred = da.from_array(pred)
        viewer.add_labels(pred, name=path, **current_im.calc_tr_sc_args(voxel_scale=current_im.CANVAS_VOXEL_SCALE))

    @viewer.bind_key('ctrl+shift+s')
    def save_canvas(_: napari.Viewer):
        nonlocal canvas
        canvas = canvas_layer.data
        tifffile.imwrite(CANVAS_PATH, canvas)

    @magicgui.magicgui(value={'max': 100000})
    def image_arithmetic(
            layerA: 'napari.types.ImageData',
            value: float
    ) -> 'napari.types.ImageData':
        """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
        if layerA is not None:
            arr = np.zeros(layerA.shape, dtype=np.uint8)
            arr[:] = layerA > value
            viewer.add_labels(arr, name='result',
                              **current_im.calc_tr_sc_args())
            # return layerA > value
        return None

    bias_scale = tuple(s * 2 for s in current_im.CANVAS_VOXEL_SCALE)
    nozadd.group_from_path(viewer, f'{global_vars.BIAS_PATH}.zip', kwargs=current_im.calc_tr_sc_args(voxel_scale=bias_scale))
    nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/im.ome.zarr', kwargs=current_im.calc_tr_sc_args(voxel_scale=current_im.CANVAS_VOXEL_SCALE))

    viewer.window.add_dock_widget(image_arithmetic)
    viewer.show(block=True)


def post_annotate_downsample(show_result=False):
    """can run without the corresponding annotated label image"""
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    im_orig = ome_io.load_dask_array_from_path(global_vars.TARGET_PATH, mode='r', level=0)
    im = ndinterp.measure_block_reduce(im_orig, (2,) * 3, reduce_fn=global_vars.REDUCE_FN, input_chunks=(2, 4096, 4096)).rechunk((4, 4096, 4096))
    asyncio.run(ome_io.write_ome_zarr_image(f'{TARGET_FOLDER}/im.ome.zarr', da_arr=im, MAX_LAYER=1))
    im = ndinterp.measure_block_reduce(im_orig, (4,) * 3, reduce_fn=global_vars.REDUCE_FN, input_chunks=(4, 4096, 4096)).rechunk((4, 4096, 4096))
    asyncio.run(ome_io.write_ome_zarr_image(f'{TARGET_FOLDER}/im_mini.ome.zarr', da_arr=im, MAX_LAYER=0))

    if show_result:
        viewer = napari.Viewer(ndisplay=2)
        nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/im.ome.zarr', kwargs=dict(name='im',
                                                                                   **current_im.calc_tr_sc_args(voxel_scale=(2,) * 3)))
        nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/im_mini.ome.zarr', kwargs=dict(name='im_mini',
                                                                                        **current_im.calc_tr_sc_args(voxel_scale=(4,) * 3)))
        viewer.show(block=True)


def apply_bias_correction(show_result=False):
    im = ome_io.load_dask_array_from_path(f'{TARGET_FOLDER}/im.ome.zarr', level=0, mode='r')
    bias = ome_io.load_dask_array_from_path(f'{TARGET_FOLDER}/bias.ome.zarr.zip', level=0, mode='r')
    print(f'im.shape={im.shape}, bias.shape={bias.shape}; applying bias over image to obtain corrected image...')
    bias = ndinterp.scale_nearest(bias, scale=(2, 2, 2), output_shape=im.shape, output_chunks=(4, 4096, 4096)).persist()
    im_corr = current_im.apply_bias(im, bias).rechunk((4, 4096, 4096))
    asyncio.run(ome_io.write_ome_zarr_image(f'{TARGET_FOLDER}/im_corrected.ome.zarr', da_arr=im_corr, MAX_LAYER=0))
    if show_result:
        viewer = napari.Viewer(ndisplay=2)
        nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/im.ome.zarr', kwargs=dict(name='im',
                                                                                   **current_im.calc_tr_sc_args()))
        nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/bias.ome.zarr.zip', kwargs=dict(name='bias',
                                                                                         **current_im.calc_tr_sc_args(voxel_scale=(2,) * 3)))
        nozadd.group_from_path(viewer, f'{TARGET_FOLDER}/im_corrected.ome.zarr', kwargs=dict(name='im_corrected',
                                                                                   **current_im.calc_tr_sc_args()))
        viewer.show(block=True)

