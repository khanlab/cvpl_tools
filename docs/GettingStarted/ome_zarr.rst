.. _ome_zarr:

OME_ZARR
========

Viewing of ome_zarr in a directory or as a zip file.

1.Open Napari with command `napari`

2.Open the command widget with button at the bottom left corner of the window.

3.After that, type in the command window to invoke functions that add more images as layers.
To view an ome-zarr file this way with `cvpl_tools`, use the command

::

    import cvpl_tools.napari.zarr as cvpl_zarr
    cvpl_zarr.add_ome_zarr_array_from_path(viewer, "/absolute/path/to/your/ome.zarr", kwargs=dict(name="displayed_name_in_ui"))

- This will create a new layer named `displayed_name_in_ui` which displays your ome zarr array.
  The underlying implementation uses Napari add_image's multiscale array display
  |
- To open a `.zip` file, specify ome.zarr.zip in the path (and optionally set use_zip=True to
  open the file as zip regardless of its suffix)
  |
- To open both the `ome.zarr` file and any label files located at `ome.zarr/labels/label_name`
  as specified by the ome zarr standard, use cvpl_zarr.add_ome_zarr_group_from_path instead.
  |
- To specify additional arguments passed to Napari's add_image function, pass your arguments in
  the kwargs as a dictionary.
  |
- To open an OME ZARR image on Google cloud storage, use the following code instead

::

    import cvpl_tools.napari.zarr as cvpl_zarr
    import gcsfs
    import zarr
    gfs = gcsfs.GCSFileSystem(token=None)
    store = gfs.get_mapper('gs://path_to_your.ome.zarr')
    zarr_group = zarr.open(store, mode='r')
    cvpl_zarr.add_ome_zarr_group(viewer, zarr_group, dict(name="displayed_name_in_ui"))

Similarly, you can open a zip, or an image with multiple labels this way.
