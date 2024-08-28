from typing import Any

import napari
from napari.layers import Layer
import numpy as np
import numpy.typing as npt
import dask.array as da
import json
from cvpl_tools.ome_zarr.napari.annotation_record_manager import AnnotationRecordManager
from napari.utils.events import Event
import cvpl_tools.im.algorithms as np_algorithms
import scipy


# def annotation_event_dispatcher(viewer: napari.Viewer, e):
#     for callback in viewer.mouse_drag_callbacks:
#         callback(viewer, e)
#     pass
#
#
# def set_mme(viewer: napari.Viewer, enable: bool):
#     """Enable/Disable default mouse move event (which moves and zooms on images when you drags the mouse)
#
#     reference: https://forum.image.sc/t/how-to-programmatically-disable-dragging-layers/43060
#     by Talley Lambert on how to programmatically disable drag events
#
#     Args:
#         viewer: napari Viewer to set mme on
#         enable: if True, enable; else disable
#     """
#     canvas_widget = viewer.window.qt_viewer.canvas.native
#     if not hasattr(canvas_widget, '_native_mme'):
#         if not enable:
#             canvas_widget._native_mme = canvas_widget.mouseMoveEvent
#             canvas_widget.mouseMoveEvent = lambda e: None
#     else:
#         if enable:
#             canvas_widget.mouseMoveEvent = canvas_widget._native_mme
#             del canvas_widget._native_mme


class UserInputBuffer:
    def __init__(self, viewer, record_manager: AnnotationRecordManager, masks):
        self.uin_arr: npt.NDArray[np.uint8] = np.zeros(masks[0]['arr'].shape, dtype=np.uint8)
        self.im_arr: npt.NDArray = masks[0]['arr']
        self.im_layer: Layer = masks[0]['layer']
        self.mask_layer: Layer = masks[0]['layer']
        self.completed_layer: Layer = masks[1]['layer']
        self.uin_layer: Layer = viewer.add_labels(self.uin_arr, name='user_input', scale=(1., 1., 1.))
        self.uin_points_layer: Layer = viewer.add_points(ndim=3, name='user_input_shape')

        self.threshold = 0.4
        thres_arr = self.im_arr > self.threshold
        self.lbl_im, self.nlbl = scipy.ndimage.label(thres_arr)[0]
        self.lbl_idx_to_np3d, self.cnt_slices = np_algorithms.npindices_from_os(self.lbl_im, return_object_slices=True)
        self.last_pts: npt.NDArray[np.int32] = None

        self.pt_to_cnt_idx = {}
        self.cnt_idx_to_pt = {i + 1: [] for i in range(self.nlbl)}

    def insert_pt(self, pt: npt.NDArray[np.int32]) -> int:
        pt_tup = tuple(pt)
        cnt_idx = self.lbl_im[pt_tup]
        self.pt_to_cnt_idx[pt_tup] = cnt_idx
        self.cnt_idx_to_pt[cnt_idx].append(pt_tup)
        return cnt_idx

    def remove_pt(self, pt: npt.NDArray[np.int32]) -> int:
        pt_tup = tuple(pt)
        cnt_idx = self.pt_to_cnt_idx.pop(pt_tup)
        self.cnt_idx_to_pt[cnt_idx].remove(pt_tup)
        return cnt_idx

    def update_uin_layer(self, pts: npt.NDArray[np.int32]):
        if self.last_pts is not None:
            diff_pts1 = np.setdiff1d(self.last_pts, pts, axis=0)
            diff_pts2 = np.setdiff1d(pts, self.lastpts, axis=0)
            diff_pts = np.concatenate((diff_pts1, diff_pts2), axis=0)
        else:
            diff_pts1 = np.zeros((0, pts.shape[1]), dtype=pts.dtype)
            diff_pts2 = pts
            diff_pts = pts
        self.last_pts = pts

        changed_cnts = set()
        for pt in diff_pts1:
            changed_cnts.add(self.remove_pt(pt))
        for pt in diff_pts2:
            changed_cnts.add(self.insert_pt(pt))


def setup_annotation_callbacks(viewer: napari.Viewer,
                               record_manager: AnnotationRecordManager,
                               masks: list[dict[str, Any]]):
    # stores the user drawn mask to be written onto mask
    uin_arr: npt.NDArray[np.uint8] = np.zeros(masks[0]['arr'].shape, dtype=np.uint8)

    im_arr: npt.NDArray = masks[0]['arr']
    im_layer: Layer = masks[0]['layer']
    mask_layer: Layer = masks[0]['layer']
    completed_layer: Layer = masks[1]['layer']
    uin_layer: Layer = viewer.add_labels(uin_arr, name='user_input', scale=(1., 1., 1.))
    uin_points_layer: Layer = viewer.add_points(ndim=3, name='user_input_shape')

    # @viewer.layers.events.connect
    # def handler(arg):
    #     # reference: https://forum.image.sc/t/registering-callback-functions-for-events-in-napari/32210/3
    #     ty = arg.type
    #     # if ty == 'labels_update':
    #     #     layer = viewer.layers[arg.index]
    #     #     if layer == mask_layer:
    #     #         print('on mask layer')
    #     #         print(arg.__dict__['_kwargs'])
    #     if ty not in ('set_data', 'thumbnail'):
    #         print("type", arg.type)

    THRESHOLD = 0.4
    thres_arr = im_arr > THRESHOLD
    thres_arr = np_algorithms.find_np3d_from_bs(thres_arr)
    def update_uin_layer(pts: npt.NDArray[np.int32], range_slices: tuple):
        thres_arr

    @uin_points_layer.events.data.connect
    def on_points_changed(event):
        pts = event.source.data
        if len(pts) > 0:
            pts = [im_layer.world_to_data(pt) for pt in pts]
            pts = np.array(pts, dtype=np.float32)
            pts = pts.round().astype(np.int32)

            # filter off points that are outside and force points onto grid
            im_shape = im_arr.shape
            off_screen = np.zeros((pts.shape[0],), dtype=np.bool)
            for i in range(len(im_shape)):
                off_screen = off_screen | (pts[:, i] < 0) | (pts[:, i] >= im_shape[i])
            pts = pts[~off_screen]

            update_uin_layer(pts)

    @viewer.bind_key('a')
    def hello(viewer):
        # on press
        print('hello world!')

        yield

        # on release
        print('goodbye world :(')

