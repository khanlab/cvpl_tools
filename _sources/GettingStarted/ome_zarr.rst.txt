.. _ome_zarr:

OME_ZARR
########

Viewing of ome zarr file
************************

Viewing of ome_zarr in a directory or as a zip file.

1.Open Napari with command :code:`napari`

2.Open the command widget with button at the bottom left corner of the window.

3.After that, type in the command window to invoke functions that add more images as layers.
To view an ome-zarr file this way with :code:`cvpl_tools`, use the command

::

    import cvpl_tools.ome_zarr.napari.zarr_viewer as cvpl_zarr
    cvpl_zarr.add_ome_zarr_array_from_path(viewer, "/absolute/path/to/your/ome.zarr", kwargs=dict(name="displayed_name_in_ui"))

- This will create a new layer named **displayed_name_in_ui** which displays your ome zarr array.
  The underlying implementation uses Napari add_image's multiscale array display
- To open a **.zip** file, specify ome.zarr.zip in the path (and optionally set use_zip=True to
  open the file as zip regardless of its suffix)
- To open both the **ome.zarr** file and any label files located at **ome.zarr/labels/label_name**
  as specified by the ome zarr standard, use cvpl_zarr.add_ome_zarr_group_from_path instead.
- To specify additional arguments passed to Napari's add_image function, pass your arguments in
  the kwargs as a dictionary.
- To open an ome zarr image on Google cloud storage, use the following code instead

.. code-block:: Python

    import cvpl_tools.ome_zarr.napari.zarr_viewer as zarr_viewer
    import gcsfs
    import zarr
    gfs = gcsfs.GCSFileSystem(token=None)
    store = gfs.get_mapper('gs://path_to_your.ome.zarr')
    zarr_group = zarr.open(store, mode='r')
    zarr_viewer.add_ome_zarr_group(viewer, zarr_group, kwargs=dict(name="displayed_name_in_ui"))

- An extra argument is_label can be passed into the function via :code:`kwargs` dictionary.
  This is a boolean value that specifies whether to use :code:`viewer.add_labels`
  (if :code:`True`) or :code:`viewer.add_image` (if :code:`False`) function. This is useful for
  displaying instance segmentaion masks, where each segmented object has a distinct color.

Similarly, you can open a zip, or an image with multiple labels this way.


Reading and Writing ome zarr files
**********************************

Before talking about the read and write, we need to first understand the directory structure of an
ome zarr file. A basic ome zarr file (which is what we work with) is a directory that looks like
the following:

.. code-block::

    - image.OME.ZARR/  # The ome zarr image, which is a directory in Windows/Linux
        - 0/  # The original image, in ZARR format
            + 0/  # Slice of image at first axis X=0
            + 1/
            ...
            .zarray  # .zarray is meta attribute for the ZARR image
        + 1/  # Downsampled image at downsample level 1, in ZARR format
        + 2/
        + 3/
        + 4/  # The smallest downsampled image at level 4, in ZARR format
        .zattrs  # .zattrs and .zgroup are meta attributes for the ome zarr image
        .zgroup

Above + denotes collapsed folder and - denotes expanded folder. A few things to note here:

- An image does not have to end with .OME.ZARR suffix
- The image is multiscaled if the maximum downsample level is 0 instead of 4, in which case there
  will only be one 0/ folder
- You may confuse an ome zarr image with a single ZARR image. An ome zarr image
  is not a standard ZARR directory and contains no **.zarray** meta file. Loading an ome zarr
  image as ZARR will crash, if you forget to specify **0/** subfolder as the path to load
- When saved as a zip file instead of a directory, the directory structure is the same except that
  the root is zipped. Loading a zipped ome zarr, cvpl_tools uses :code:`ZipStore`'s features to
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
  function to write a dask array as an ome zarr
  file. This function also adds support for reading images stored as a **.zip** file.

See the API page for cvpl_tools.ome_zarr.io.py for how to read and write OME
ZARR files if you want to use :code:`cvpl_tools` for such tasks. This file provides two functions
:code:`load_zarr_group_from_path` and :code:`write_ome_zarr_image` which allows you to read and write OME
ZARR files, respectively.


Specifying slices in path
*************************

:code:`cvpl_tools` allows specifying the channel, or x/y/z slices to use in the path string when
reading or viewing an ome zarr file for convenience.

The functions :code:`load_dask_array_from_path` in :code:`cvpl_tools.ome_zarr.io`, and
:code:`load_zarr_group_from_path` as well as :code:`load_ome_zarr_array_from_path` from
:code:`cvpl_tools.ome_zarr.napari.zarr_viewer` support specifying the slices in the following
syntax, much similar to torch or numpy array slicing:

.. code-block:: Python

    arr_original = load_dask_array_from_path('file.ome.zarr', level=0)  # shape=(2, 200, 1000, 1000)
    arr1 = load_dask_array_from_path('file.ome.zarr?slices=[0]', level=0)  # shape=(200, 1000, 1000)
    arr2 = load_dask_array_from_path('file.ome.zarr?slices=[:, :100]', level=0)  # shape=(2, 100, 1000, 1000)
    arr3 = load_dask_array_from_path('file.ome.zarr?slices=[0:1, 0, -1:, ::2]', level=0)  # shape=(1, 1, 500)

The idea of this syntax attributes to Davis Bennett
`(see this discussion) <https://forum.image.sc/t/loading-only-one-channel-from-an-ome-zarr/97798>`_.

Why do we need to specify slices this way? Commonly, we pass in an ome
zarr path to specify the input image of a script. If we want to run the script on the first channel
of a multi-channel image, both a :code:`path` to ome zarr and an :code:`in_channel`
integer specifying the channel to use are needed.
With this syntax, we only need one input variable to specify
the channel to use, as well as a sub-region of the image if we want to crop the input.

