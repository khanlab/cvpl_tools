.. _napari_zarr:

cvpl_tools/ome_zarr/napari/add.py
=================================

View source at `zarr.py <https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/napari/zarr.py>`_.

For ome zarr images, :code:`subarray_from_path` can be used generally. If an image has an
associated label OME_ZARR file(s) in the "[image_ome_zarr]/labels/label_name" path, then the
image and label(s) can be opened together with a single :code:`group_from_path` call.

.. currentmodule:: cvpl_tools.ome_zarr.napari.add

.. rubric:: APIs

.. autofunction:: group
.. autofunction:: subarray
.. autofunction:: group_from_path
.. autofunction:: subarray_from_path
