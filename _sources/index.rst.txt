.. cvpl_tools documentation master file, created by
   sphinx-quickstart on Tue Jul 23 11:16:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cvpl_tools documentation
========================


This tool provides useful visualization and processing APIs for image files. Built on
top of:

- Napari
- ome-zarr-py
- Numpy, Scipy and scikit-image

These pip dependencies and their versions are same as SPIMquant and (if dependency
conflicts in other places, let me know).

Installation of `cvpl_tools` using pip (Note Napari viewer requires one of
`pyqt or pyside2 <https://napari.org/stable/tutorials/fundamentals/installation.html>`_ installed.)
::

   pip install cvpl_tools

Now you can go to GettingStarted/ome_zarr to learn how to view an OME_ZARR image locally
or on cloud.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GettingStarted/ome_zarr
   API/ome_zarr_io

