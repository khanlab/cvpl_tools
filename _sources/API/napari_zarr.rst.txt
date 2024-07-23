.. _ome_zarr_io:

cvpl_tools/napari/zarr.py
=========================

View source at `io.py <https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/napari/zarr.py>`_.

For OME ZARR images, `add_ome_zarr_array_from_path` can be used generally. If an image has an
associated label OME_ZARR file(s) in the "[image_ome_zarr]/labels/label_name" path, then the
image and label(s) can be opened together with a single `add_ome_zarr_group_from_path` call.

.. autofunction:: cvpl_tools.napari.zarr.add_ome_zarr_group
.. autofunction:: cvpl_tools.napari.zarr.add_ome_zarr_array
.. autofunction:: cvpl_tools.napari.zarr.add_ome_zarr_group_from_path
.. autofunction:: cvpl_tools.napari.zarr.add_ome_zarr_array_from_path
