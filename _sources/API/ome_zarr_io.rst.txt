.. _ome_zarr_io:

cvpl_tools/ome_zarr/io.py
=========================

View source at `io.py <https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/ome_zarr/io.py>`_.

Read and Write: For reading OME ZARR image, use **load_zarr_group_from_path** to open a zarr group in
read mode and then use **dask.array.from_zarr** to create a dask array from the group. For writing OME
ZARR image, we assume you have a dask array and would like to write it as a .zip or a directory. In
such cases, **write_ome_zarr_image** directly writes the dask array onto disk.

.. rubric:: APIs

.. autofunction:: cvpl_tools.ome_zarr.io.load_zarr_group_from_path
.. autofunction:: cvpl_tools.ome_zarr.io.write_ome_zarr_image
