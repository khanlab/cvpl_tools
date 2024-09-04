from dataclasses import dataclass
from typing import Any, Callable

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
import uuid


def are_instersect_slices(slicesA: tuple, slicesB: tuple):
    for i in range(len(slicesA)):
        sA, sB = slicesA[i], slicesB[i]
        if sA.start >= sB.stop or sB.start >= sA.stop:
            return False
    return True


def take_intersect(slicesA: tuple, slicesB: tuple):
    new_s = tuple(slice(max(slicesA[k].start, slicesB[k].start), min(slicesA[k].stop, slicesB[k].stop))
                  for k in range(len(slicesA)))
    return new_s


def are_in_slices(pts: npt.NDArray[np.int32], slices: tuple) -> npt.NDArray[bool]:
    in_slices = np.ones(pts.shape[0], dtype=bool)
    for i in range(len(slices)):
        s = slices[i]
        X = pts[:, i]
        in_slices = in_slices & (X >= s.start) & (X < s.stop)
    return in_slices


def merge_intersecting_slices(slices, ids=None) -> tuple[list, list]:
    """Returns the merged slices and their corresponding ids"""
    slices = list(slices)
    if ids is None:
        ids = [[i] for i in range(len(slices))]
    else:
        ids = [[i] for i in ids]
    assert len(slices) == len(ids), 'Each slice tuple corresponds to an id'
    if len(slices) <= 1:
        return slices, ids

    i = 0
    while i < len(slices) - 1:
        si = slices[i]
        intersect_flag = False
        j = i + 1
        while j < len(slices):
            sj = slices[j]
            if are_instersect_slices(si, sj):
                intersect_flag = True
                break
            j += 1
        if intersect_flag:
            sj = slices.pop(j)
            slices.pop(i)
            new_s = tuple(slice(min(si[k].start, sj[k].start), max(si[k].stop, sj[k].stop)) for k in range(len(si)))
            slices.append(new_s)

            idj = ids.pop(j)
            idi = ids.pop(i)
            ids.append(idi + idj)
        else:
            i += 1
    return slices, ids


def construct_slices(pts: npt.NDArray[np.int32] | list[tuple, ...], width: int):
    if isinstance(pts, np.ndarray):
        pts = pts.tolist()
    ndim = len(pts[0])
    slices = [tuple(slice(pt[i] - width, pt[i] + width + 1) for i in range(ndim)) for pt in pts]
    slices, result_pts = merge_intersecting_slices(slices)
    for i in range(len(result_pts)):
        pts_ids = result_pts[i]
        result_pts[i] = np.array([pts[i] for i in pts_ids], dtype=np.int32)
    return slices, result_pts


@dataclass
class LblInfo:
    thres_arr: npt.NDArray[np.uint8]
    lbl_im: npt.NDArray[np.int32]
    nlbl: int
    cnt_slices: list
    next_pt_id: int

    def __init__(self, thres_arr, next_pt_id: int):
        self.thres_arr = thres_arr
        self.lbl_im, self.nlbl = scipy.ndimage.label(thres_arr)
        self.cnt_slices = list(scipy.ndimage.find_objects(self.lbl_im))
        self.next_pt_id = next_pt_id

    def assign_id(self):
        pt_id = self.next_pt_id
        self.next_pt_id += 1
        return pt_id


class UserInputBuffer:
    """Manages the last few drawn contours"""

    def __init__(self,
                 viewer: napari.Viewer,
                 record_manager: AnnotationRecordManager,
                 masks,
                 name='user_input',
                 bind_key: str | None = None):
        self.im_arr = masks[0]['arr']
        self.im_layer = masks[0]['layer']
        self.ndim = self.im_arr.ndim
        uin_arr: npt.NDArray[np.uint8] = np.zeros(self.im_arr.shape, dtype=np.uint8)
        self.uin_layer: Layer = viewer.add_labels(uin_arr, name=name, scale=self.im_layer.scale)
        self.uin_points_layer: Layer = viewer.add_points(ndim=3, name=f'{name}_shape', size=1,
                                                         scale=self.im_layer.scale)
        self.uin_points_layer.mode = 'add'  # default to add points
        self.record_manager = record_manager
        self.width = 40

        self.last_pts: npt.NDArray[np.int32] = None
        self.lblinfo: LblInfo = None
        self.pt_to_paintid = None
        self.pt_to_cnt_idx = None
        self.cnt_idx_to_pt = None
        self.cnt_idx_to_pt_history = None

        self.on_points_changed = self.uin_points_layer.events.data.connect(self.on_points_changed)
        self.on_push = None

        if bind_key is not None:
            @viewer.bind_key(bind_key)
            def when_push(e):
                if self.on_push is not None:
                    self.on_push(self.uin_points_layer.data)
                self.clear_data()

    def connect(self, on_push):
        self.on_push = on_push

    def set_pts(self, pts: npt.NDArray[np.int32]):
        for pt in pts:
            self.insert_pt(pt)

    def insert_pt(self, pt_tup: tuple) -> int:
        cnt_idx = self.lblinfo.lbl_im[pt_tup].item()
        self.pt_to_cnt_idx[pt_tup] = cnt_idx
        self.cnt_idx_to_pt[cnt_idx].append(pt_tup)
        self.cnt_idx_to_pt_history[cnt_idx].add(pt_tup)
        return cnt_idx

    def remove_pt(self, pt_tup: tuple) -> int:
        cnt_idx = self.pt_to_cnt_idx.pop(pt_tup)
        self.cnt_idx_to_pt[cnt_idx].remove(pt_tup)
        return cnt_idx

    def clear_data(self):
        self.last_pts: npt.NDArray[np.int32] = None
        self.pt_to_paintid = {}
        self.pt_to_cnt_idx = {}
        self.cnt_idx_to_pt = {i: [] for i in range(self.lblinfo.nlbl + 1)}
        self.cnt_idx_to_pt_history = {i: set() for i in range(self.lblinfo.nlbl + 1)}
        self.uin_points_layer.data = np.zeros((0, self.ndim), dtype=np.int32)
        self.uin_layer.data = np.zeros_like(self.uin_layer.data)
        self.uin_layer.refresh()

    def update_lblinfo(self, lblinfo: LblInfo):
        # TODO: allow regional update
        saved_pts = self.last_pts
        self.lblinfo = lblinfo
        self.clear_data()
        if saved_pts is not None:
            self.set_pts(saved_pts)  # this is to avoid using the previous threshold for new object

    def on_points_changed(self, event):
        pts = event.source.data
        if len(pts) > 0:
            pts = [self.im_layer.world_to_data(self.uin_points_layer.data_to_world(pt)) for pt in pts]
            pts = np.array(pts, dtype=np.float32)
            pts = pts.round().astype(np.int32)

            # filter off points that are outside and force points onto grid
            im_shape = self.im_arr.shape
            off_screen = np.zeros((pts.shape[0],), dtype=np.bool)
            for i in range(len(im_shape)):
                off_screen = off_screen | (pts[:, i] < 0) | (pts[:, i] >= im_shape[i])
            pts = pts[~off_screen]

            self.update_uin_layer(pts)

    def update_uin_layer(self, pts: npt.NDArray[np.int32]):
        lblinfo = self.lblinfo
        pts = np.unique(pts, axis=0)

        if self.last_pts is not None:
            diff_pts1 = np_algorithms.setdiff2d(self.last_pts, pts)
            diff_pts2 = np_algorithms.setdiff2d(pts, self.last_pts)
        else:
            diff_pts1 = np.zeros((0, pts.shape[1]), dtype=pts.dtype)
            diff_pts2 = pts
        # all_changed_pts_map = {}
        self.last_pts = pts

        changed_cnts = set()
        for pt in diff_pts1:
            pt_tup = tuple(pt)
            changed_cnt = self.remove_pt(pt_tup)
            changed_cnts.add(changed_cnt)
            self.pt_to_paintid.pop(pt_tup)
        for pt in diff_pts2:
            pt_tup = tuple(pt)
            changed_cnt = self.insert_pt(pt_tup)
            changed_cnts.add(changed_cnt)
            self.pt_to_paintid[pt_tup] = lblinfo.assign_id()
        if 0 in changed_cnts:
            changed_cnts.remove(0)  # we don't need to split background
        if len(changed_cnts) == 0:
            return

        # now we have a list of changed contours, we want to modify the original image regions
        # where these changed contours overlap
        uin_im = self.uin_layer.data
        for cnt_idx in changed_cnts:
            print(cnt_idx, 'changed')
            contour_slices = lblinfo.cnt_slices[cnt_idx - 1]
            contour_pts: list = self.cnt_idx_to_pt[cnt_idx]
            all_contour_pts = np.array(contour_pts, dtype=np.int32)
            all_pts_history = list(self.cnt_idx_to_pt_history[cnt_idx])
            all_changed_pts = np.array(all_pts_history, dtype=np.int32)
            changed_pts_slices, _ = construct_slices(all_changed_pts, width=999)

            paintids = np.array([self.pt_to_paintid[pt_tup] for pt_tup in contour_pts], dtype=np.int32)

            for i in range(len(changed_pts_slices)):
                slices = take_intersect(contour_slices, changed_pts_slices[i])
                offset_mask = lblinfo.lbl_im[slices] == cnt_idx  # this is the contour the points split on

                if len(contour_pts) == 0:
                    splitted = np.zeros(offset_mask.shape, dtype=np.int32)
                else:
                    if are_in_slices(all_contour_pts, slices).sum() == 0:
                        splitted = np.zeros(offset_mask.shape, dtype=np.int32)
                    else:
                        stpt = np.array(tuple(s.start for s in slices), dtype=np.int32)
                        splitted = np_algorithms.voronoi_ndarray(offset_mask.shape, all_contour_pts - stpt[None, :])
                        splitted = splitted * offset_mask

                        splitted = paintids[splitted]
                uin_im[slices] = (uin_im[slices] * ~offset_mask) + (splitted * offset_mask)
        self.uin_layer.refresh()


def setup_annotation_callbacks(viewer: napari.Viewer,
                               record_manager: AnnotationRecordManager,
                               masks: list[dict[str, Any]]):
    im_arr: npt.NDArray = masks[0]['arr']

    uinb_mask = UserInputBuffer(viewer, record_manager, masks, name='uin_canvas')
    lblinfo_mask = LblInfo(im_arr > .44, 1)
    uinb_mask.update_lblinfo(lblinfo_mask)

    uinb = UserInputBuffer(viewer, record_manager, masks, name='uin_buffer', bind_key='ctrl+d')
    lblinfo = LblInfo(im_arr > .44, 1)
    uinb.update_lblinfo(lblinfo)

    def on_push(pts):
        layer = uinb_mask.uin_points_layer
        layer.data = np.concatenate((layer.data, pts), axis=0).astype(np.int32)
    uinb.connect(on_push)

