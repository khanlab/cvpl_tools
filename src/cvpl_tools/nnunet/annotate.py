import os.path

import cvpl_tools.nnunet.current_im as current_im
import napari
import numpy as np
import tifffile
import dask.array as da
from cvpl_tools.nnunet.triplanar import dice_on_volume_pair


def get_canvas(canvas_path, canvas_ref_path, canvas_shape):
    if os.path.exists(canvas_path):
        READ_PATH = canvas_path
    elif canvas_ref_path is not None and os.path.exists(canvas_ref_path):
        READ_PATH = canvas_ref_path
    else:
        READ_PATH = None

    if READ_PATH is None:
        canvas = np.zeros(canvas_shape, dtype=np.uint8)
    else:
        canvas = tifffile.imread(READ_PATH).astype(np.uint8)
        assert np.dtype(canvas.dtype) == np.uint8, f'{canvas.dtype}'
    return canvas


def annotate(viewer, im_annotate, annotation_folder, canvas_path, SUBJECT_ID: str):
    """
    usage:
    import cvpl_tools.nnunet.annotate as ann
    ann.annotate()
    """
    import magicgui
    import cvpl_tools.nnunet.current_im as current_im_py

    im_layer = viewer.add_image(im_annotate, name='im', **current_im_py.calc_tr_sc_args(voxel_scale=(1,) * 3, display_shape=im_annotate.shape))

    canvas = get_canvas(canvas_path, None, im_annotate.shape)
    canvas_layer = viewer.add_labels(canvas, name='canvas', **current_im.calc_tr_sc_args(voxel_scale=(2,) * 3, display_shape=im_annotate.shape))

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
        print(f'dice for {path}:', dice_on_volume_pair(canvas, pred))
        pred = da.from_array(pred)
        viewer.add_labels(pred, name=path, **current_im.calc_tr_sc_args(voxel_scale=current_im.CANVAS_VOXEL_SCALE, display_shape=im_annotate.shape))

    @viewer.bind_key('ctrl+shift+s')
    def save_canvas(_: napari.Viewer):
        nonlocal canvas
        canvas = canvas_layer.data
        tifffile.imwrite(canvas_path, canvas)

    @magicgui.magicgui(value={'max': 100000})
    def image_arithmetic(
            layerA: 'napari.types.ImageData',
            value: float
    ) -> 'napari.types.ImageData':
        """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
        if layerA is not None:
            arr = np.zeros(layerA.shape, dtype=np.uint8)
            arr[:] = layerA > value
            viewer.add_labels(arr,
                              name='result',
                              **current_im.calc_tr_sc_args(voxel_scale=(1,) * 3, display_shape=im_annotate.shape))

    viewer.window.add_dock_widget(image_arithmetic)
