.. cvpl_tools documentation master file, created by
   sphinx-quickstart on Tue Jul 23 11:16:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction - cvpl_tools documentation
=======================================

This tool provides useful visualization and processing APIs for image files. Built on
top of:

- Napari
- ome-zarr-py
- Numpy, Scipy and scikit-image

These pip dependencies and their versions are same as SPIMquant. If dependency
conflicts in other places, let me know.

Installation of `cvpl_tools` using pip (Note Napari viewer requires one of
`pyqt or pyside2 <https://napari.org/stable/tutorials/fundamentals/installation.html>`_ installed.)
::

   pip install cvpl_tools

Now you can go to GettingStarted/ome_zarr to learn how to view an OME_ZARR image locally
or on cloud.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction <self>
   Quickstart <GettingStarted/quick_start>
   Viewing and IO of OME Zarr <GettingStarted/ome_zarr>
   Setting Up the Script <GettingStarted/setting_up_the_script>
   Defining Segmentation Pipeline <GettingStarted/segmentation_pipeline>
   Result Caching <GettingStarted/result_caching>
   nn-UNet <GettingStarted/nnunet>
   Beta-Amyloid Plaque Detection <GettingStarted/beta_amyloid_plaque_detection>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   cvpl_tools.ome_zarr.napari.add.py <API/napari_zarr>
   cvpl_tools.ome_zarr.io.py <API/ome_zarr_io>
   cvpl_tools.tools.fs.py <API/tlfs>
   cvpl_tools.im.ndblock.py <API/ndblock>
   cvpl_tools.im.process <API/seg_process>
   cvpl_tools.im.algs <API/algs>


