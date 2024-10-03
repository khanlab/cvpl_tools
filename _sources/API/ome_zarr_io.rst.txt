.. _ome_zarr_io:

cvpl_tools/ome_zarr/io.py
=========================

View source at `io.py <https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/ome_zarr/io.py>`_.

Read and Write: For reading ome zarr image, use :code:`load_dask_array_from_path` to directly read the OME
ZARR file as a dask array. Alternatively, use :code:`load_zarr_group_from_path` to open a zarr group in
read mode and then use :code:`dask.array.from_zarr` to create a dask array from that group.

For writing ome zarr image, we assume you have a dask array and would like to write it as a .zip or a
directory. In such cases, :code:`write_ome_zarr_image` directly writes the dask array onto disk.

.. rubric:: APIs

.. autofunction:: cvpl_tools.ome_zarr.io.load_zarr_group_from_path
.. autofunction:: cvpl_tools.ome_zarr.io.load_dask_array_from_path
.. autofunction:: cvpl_tools.ome_zarr.io.write_ome_zarr_image
