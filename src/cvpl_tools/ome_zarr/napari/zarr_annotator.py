import napari
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
import cvpl_tools.ome_zarr.napari.zarr_viewer as cvpl_napari_zarr
import numpy as np
import numpy.typing as npt
import dask.array as da
import json
from cvpl_tools.ome_zarr.napari.annotation_record_manager import AnnotationRecordManager
import cvpl_tools.ome_zarr.napari.annotation_callbacks as annotation_callbacks


def read_scale(ome_zarr_path: str, use_zip: bool = None) -> npt.NDArray[np.float32]:
    """Returns the scale of the first level of the image as a numpy array"""
    zarr_group = cvpl_ome_zarr_io.load_zarr_group_from_path(ome_zarr_path, mode='r', use_zip=use_zip)
    metadata = zarr_group.attrs['multiscales'][0]
    scale = metadata['datasets'][0]['coordinateTransformations'][0]['scale']
    return np.array(scale, dtype=np.float32)


def read_arr_shape(ome_zarr_path: str, use_zip: bool = None) -> npt.NDArray[np.int32]:
    zarr_group = cvpl_ome_zarr_io.load_zarr_group_from_path(ome_zarr_path, mode='r', use_zip=use_zip)['0']
    arr = da.from_zarr(zarr_group)
    return np.array(arr.shape, dtype=np.int32)


def mask_from_path(viewer: napari.Viewer, mask_path, viewer_args: dict, im_shape, dtype):
    if mask_path is None:
        # create new array
        mask_arr = np.zeros(im_shape, dtype=dtype)
    else:
        zarr_group = cvpl_ome_zarr_io.load_zarr_group_from_path(mask_path, level=0)
        mask_arr = da.from_zarr(zarr_group).compute()
    layer = viewer.add_labels(mask_arr, **viewer_args)
    return mask_arr, layer


def annotate(ome_zarr_path: str, output_dir: str):
    im_scale = read_scale(ome_zarr_path)
    im_scale[0] *= 2  # TODO: remove this in a real dataset
    im_shape = read_arr_shape(ome_zarr_path)
    im_shape[1:] //= 4
    im_shape_slices = tuple(slice(0, s.item()) for s in im_shape)
    record_manager = AnnotationRecordManager(output_dir)
    try:
        viewer = napari.Viewer(ndisplay=2)

        im_zarr_group = cvpl_ome_zarr_io.load_zarr_group_from_path(ome_zarr_path, level=0, mode='r')
        im_arr = np.clip(da.from_zarr(im_zarr_group)[im_shape_slices].compute() / 1000., 0., 1.)
        im_layer = viewer.add_image(im_arr, name='lightsheet', scale=im_scale, contrast_limits=(0., 1.))

        # load images
        sid = record_manager.get_sid()
        # mask_path = f'{output_dir}/annotation_mask_{sid - 1}'
        # mask, mask_layer = mask_from_path(viewer,
        #                                   mask_path if sid > 1 else None,
        #                                   dict(
        #                                       name='annotation_mask',
        #                                       scale=im_scale,
        #                                   ),
        #                                   im_shape,
        #                                   dtype=np.int32)
        completed_path = f'{output_dir}/completed_{sid - 1}'
        completed, completed_layer = mask_from_path(viewer,
                                                    completed_path if sid > 1 else None,
                                                    dict(
                                                        name='completed_mask',
                                                        scale=(im_scale * 64).astype(np.int64),
                                                        translate=tuple(s // 2 for s in im_shape)
                                                    ),
                                                    im_shape // 64,
                                                    dtype=np.uint8)
        print(im_shape // 64, im_scale * 64)

        # register images to record manager
        masks = [
            dict(
                arr=im_arr,
                name='im',
                path=None,
                is_numpy=True,
                on_update=lambda: None,
                on_close=lambda: None,  # TODO: change this to save onto disk
                layer=im_layer
            ),
            # dict(
            #     arr=mask,
            #     name='annotated_mask',
            #     path=mask_path,
            #     is_numpy=True,
            #     on_update=lambda: mask_layer.refresh(),
            #     on_close=lambda: None,  # TODO: change this to save onto disk
            #     layer=mask_layer
            # ),
            dict(
                arr=completed,
                name='completed_mask',
                path=completed_path,
                is_numpy=True,
                on_update=lambda: completed_layer.refresh(),
                on_close=lambda: None,  # TODO: change this to save onto disk
                layer=completed_layer
            )
        ]
        record_manager.register_masks(masks)

        # setup editing callbacks for mouse and keyboard
        annotation_callbacks.setup_annotation_callbacks(viewer, record_manager, masks)

        viewer.title = 'zarr_annotator'
        napari.run()
    finally:
        record_manager.close()
