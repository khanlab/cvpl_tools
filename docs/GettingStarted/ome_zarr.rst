.. _ome_zarr:

OME_ZARR
########

Viewing of ome_zarr file
************************

Viewing of ome_zarr in a directory or as a zip file.

1.Open Napari with command :code:`napari`

2.Open the command widget with button at the bottom left corner of the window.

3.After that, type in the command window to invoke functions that add more images as layers.
To view an ome-zarr file this way with :code:`cvpl_tools`, use the command

::

    import cvpl_tools.napari.zarr as cvpl_zarr
    cvpl_zarr.add_ome_zarr_array_from_path(viewer, "/absolute/path/to/your/ome.zarr", kwargs=dict(name="displayed_name_in_ui"))

- This will create a new layer named **displayed_name_in_ui** which displays your ome zarr array.
  The underlying implementation uses Napari add_image's multiscale array display
- To open a **.zip** file, specify ome.zarr.zip in the path (and optionally set use_zip=True to
  open the file as zip regardless of its suffix)
- To open both the **ome.zarr** file and any label files located at **ome.zarr/labels/label_name**
  as specified by the ome zarr standard, use cvpl_zarr.add_ome_zarr_group_from_path instead.
- To specify additional arguments passed to Napari's add_image function, pass your arguments in
  the kwargs as a dictionary.
- To open an OME ZARR image on Google cloud storage, use the following code instead

.. code-block:: Python

    import cvpl_tools.napari.zarr as cvpl_zarr
    import gcsfs
    import zarr
    gfs = gcsfs.GCSFileSystem(token=None)
    store = gfs.get_mapper('gs://path_to_your.ome.zarr')
    zarr_group = zarr.open(store, mode='r')
    cvpl_zarr.add_ome_zarr_group(viewer, zarr_group, dict(name="displayed_name_in_ui"))

- An extra argument is_label can be passed into the function via :code:`kwargs` dictionary.
  This is a boolean value that specifies whether to use :code:`viewer.add_labels`
  (if :code:`True`) or :code:`viewer.add_image` (if :code:`False`) function. This is useful for
  displaying instance segmentaion masks, where each segmented object has a distinct color.

Similarly, you can open a zip, or an image with multiple labels this way.


Reading and Writing ome_zarr files
**********************************

Before talking about the read and write, we need to first understand the directory structure of an
OME ZARR file. A basic OME ZARR file (which is what we work with) is a directory that looks like
the following:

.. code-block::

    - image.OME.ZARR/  # The OME ZARR image, which is a directory in Windows/Linux
        - 0/  # The original image, in ZARR format
            + 0/  # Slice of image at first axis X=0
            + 1/
            ...
            .zarray  # .zarray is meta attribute for the ZARR image
        + 1/  # Downsampled image at downsample level 1, in ZARR format
        + 2/
        + 3/
        + 4/  # The smallest downsampled image at level 4, in ZARR format
        .zattrs  # .zattrs and .zgroup are meta attributes for the OME ZARR image
        .zgroup

Above + denotes collapsed folder and - denotes expanded folder. A few things to note here:

- An image does not have to end with .OME.ZARR suffix
- The image is multiscaled if the maximum downsample level is 0 instead of 4, in which case there
  will only be one 0/ folder
- You may confuse an OME ZARR image with a single ZARR image. An OME ZARR image
  is not a standard ZARR directory and contains no **.zarray** meta file. Loading an OME ZARR
  image as ZARR will crash, if you forget to specify **0/** subfolder as the path to load
- When saved as a zip file instead of a directory, the directory structure is the same except that
  the root is zipped. Loading a zipped OME ZARR, cvpl_tools uses :code:`ZipStore`'s features to
  directly reading individual chunks without having to unpack
  the entire zip file. However, writing to a :code:`ZipStore` is not supported, due to lack of
  support by either Python's :code:`zarr` or the :code:`ome-zarr` library.
- An HPC system like Compute Canada may work better with one large files than many small files,
  thus the result should be zipped. This can be done by first writing the folder to somewhere
  that allows creating many small files and then zip the result into a single zip in the target
  directory
- As of the time of writing (2024.8.14), ome-zarr library's :code:`Writer` class has a
  `double computation issue <https://github.com/ome/ome-zarr-py/issues/392>`_. To temporary patch
  this for our use case, I've added a :code:`write_ome_zarr_image`
  function to write a dask array as an OME ZARR
  file. This function also adds support for reading images stored as a **.zip** file.

See the API page for cvpl_tools.ome_zarr.io.py for how to read and write OME
ZARR files if you want to use :code:`cvpl_tools` for such tasks. This file provides two functions
:code:`load_zarr_group_from_path` and :code:`write_ome_zarr_image` which allows you to read and write OME
ZARR files, respectively.
