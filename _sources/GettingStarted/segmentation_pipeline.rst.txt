.. _segmentation_pipeline:

Segmentation Pipeline
#####################

Motivation: Microscope, Cell Counting, Atlas map and Object Segmentation
************************************************************************

In our use case, lightsheet microscopy of mouse brain produces several hundreds of GBs of
data, represented as a 3-dimensional array. This array is stored as an OME
ZARR file in our case. Distributed processing of the data is necessary to make the analysis time
trackable, and we choose to use Dask as the distributed computing library for our project.

As part of our research, we need to use automated method to count different types of visible objects in
the image. After counting cells, we use a map that maps from pixel location of
the image to brain regions (this is the **atlas map**) to obtain the density of cells in each region of
the mouse brain. We need to test several methods and find one that would give the most
accurate counts, and for new incoming data volumes, we want to be able to quickly find a set of parameters
that works on the new data.

On the algorithm side, object counting consists of performing a sequence of steps to process an input image
into densities over regions of the brain, and is relatively simple to understand and implement on a Numpy
array with small dataset size. On larger datasets, we need to do long-running distributed computation
that are hard to debug and requires tens of minutes or hours if we need to rerun the computation
(often just to display the results again).

SegProcess
**********

Below we use examples from :code:`cvpl_tools.im.process` to talk about a convenient way
to define a function in multi-step image processing pipeline for distributed, interpretable and cached image
data analysis.

Consider a function that counts the number of cells in a 3d-block of brightness map:

.. code-block:: Python

    import dask.array as da
    def cell_count(block3d: da.Array):
        """Count the number of cells in a dask 3d brightness image"""
        mask = threshold(block3d, 0.4)  # use simple thresholding to mark pixels where cells are as 1s
        inst = instance_segmentation(mask)  # segment the mask into contours
        cell_cnt = count_inst(inst)  # count each contour as a cell
        return cell_cnt

This code may seem complete, but it has a few issues:

1. Lack of interpretability. Often when we first run this function some bug will show up, for example
when the output cell count is unexpectedly large. Debugging this becomes a problem since we don't know
if one of the three steps in the cell_count function did not work as expected, or if the algorithm does
not work well on the input data for some reason. In either case, if we want to find the root cause of
the problem we very often end up adding code and rerun the pipeline to display the output of each step
to see if they match with our expectations.

2. Result is not cached. Ideally the pipeline is run once and we get the result, but more often than
not the result may need to be used in different places (visualization, analysis etc.). Caching these
results makes sure computation is done only once, which is necessary when we work with costly algorithms
on hundreds of GBs of data.

The basic idea to address 1) is to put visualization as part of the cell_count function, and to address
2) is to cache the result of each step into a file in a :code:`CacheDirectory`. It will provide:

1. dask-support. Inputs are expected to be either numpy array, dask array, or
:code:`cvpl.im.ndblock.NDBlock` objects. In particular, dask.Array and NDBlock are suitable for
parallel or distributed image processing workflows

2. integration of Napari. The function has an attribute :code:`context_args` that has a keyed item
:code:`viewer_args` defaults to None. By passing a Napari viewer as :code:`viewer_args["viewer"]`,
the function will add intermediate images or centroids to the Napari viewer for easier debugging.
After the function returns, we can call :code:`viewer.show()` to display all added images

3. intermediate result caching. It provides a hierarchical caching directory,
where in a call to the function it will either create a new directory, or load from existing
cache directory based on the :code:`cache_url` parameter in :code:`context_args` parameter

Now we discuss how to define a process function

Extending the Pipeline
**********************

The first step of building a pipeline is to break a segmentation algorithm down to steps that process the
image in different formats. As an example, we may implement a pipeline as IN -> BS -> OS -> CC, where:

- IN - Input Image (:code:`np.float32`) between min=0 and max=1, this is the brightness dask image as input
- BS - Binary Segmentation (3d, :code:`np.uint8`), this is the binary mask single class segmentation
- OS - Ordinal Segmentation (3d, :code:`np.int32`), this is the 0-N where contour 1-N each denotes an object; also single class
- CC - Cell Count Map (3d, :code:`np.float64`), a cell count number (estimate, can be float) for each block

Mapping from IN to BS comes in two choices. One is to simply take threshold > some number as cells and the
rest as background. Another is to use a trained machine learned algorithm to do binary segmentation. Mapping
from BS to OS also comes in two choices. Either directly treating each connected volume as a separate cell,
or use watershed to get finner segmentation mask. Finally, We can count cells in the instance
segmentation mask by perhaps look at how many seperate contours we have found.

In some cases this is not necessary if we know what algorithm works best, but abstracting the algorithm
intermediate results as four types IN, BS, OS, CC have helped us identify which part of the pipeline can
be reused and which part may have variations in the algorithm used.

We can then plan the processing steps we need to define as follows:

1. thresholding (IN -> BS)
2. model_prediction (IN -> BS)
3. direct_inst_segmentation (BS -> OS)
4. watershed_inst_segmentation (BS -> OS)
5. cell_cnt_from_inst (OS -> CC)

How do we go from this plan to actually code these steps? For each step, we define a function :code:`process()`,
which takes arbitrary inputs and one parameter: :code:`context_args`, which will contain keyed items as follows:

- cache_url (str | RDirFileSystem, optional): Pointing to a directory to store the cached image; if not
  provided, then the image will be cached via dask's persist() and its loaded copy will be returned

- storage_option (dict, optional): If provided, specifies the compression method to use for image chunks
  - preferred_chunksize (tuple, optional): Re-chunk before save; this rechunking will be undone in load
  - multiscale (int, optional): Specifies the number of downsampling levels on OME ZARR
  - compressor (numcodecs.abc.Codec, optional): Compressor used to compress the chunks

- viewer_args (dict, optional): If provided, an image will be displayed as a new layer in Napari viewer
  - viewer (napari.Viewer, optional): Only display if a viewer is provided
  - is_label (bool, optional): defaults to False; if True, use viewer's add_labels() instead of
  add_image() to display the array

- layer_args (dict, optional): If provided, used along with viewer_args to specify add_image() kwargs

- As a convention, :code:`context_args` contains :code:`cache_url` which is required only if the function
  needs some place to store intermediate results:

  .. code-block:: Python

      async def process(im, context_args: dict):
          cache_url = context_args['cache_url']
          query = tlfs.cdir_commit(cache_url)
          # in the case cache does not exists, cache_path.url is an empty path we can create a folder in:
          if not query.commit:
              result = compute_result(im)
              save(cache_url, result)
          result = load(cache_url)
          return result

- The :code:`viewer_args` parameter specifies the napari viewer to display the intermediate results. If not provided
  (:code:`viewer_args=None`), then no computation will be done to visualize the image. Within the forward() method, you
  should use :code:`viewer.add_labels()`, :code:`lc_interpretable_napari()` or :code:`temp_directory.cache_im()`
  while passing in :code:`viewer_args` argument to display your results:

  .. code-block:: Python

      async def process(im, context_args):
          result = compute_result(im)
          result = await tlfs.cache_im(lambda: result, context_args=dict(
            cache_url=context_args.get('cache_url'),
            viewer_args=context_args.get('viewer_args')))
          return result
      # ...
      viewer = napari.Viewer(ndisplay=2)
      viewer_args = dict(
          viewer=viewer,  # The napari viewer, visualization will be skipped if viewer is None
          is_label=True,  # If True, viewer.add_labels() will be called; if False, viewer.add_image() will be called
          preferred_chunksize=(1, 4096, 4096),  # image will be converted to this chunksize when saved, and converted back when loaded
          multiscale=4,  # maximum downsampling level of ome zarr files, necessary for very large images
      )
      context_args = dict(
          cache_url='gcs://example/cloud/path',
          viewer_args=viewer_args
      )
      await process(im, context_args=context_args)

  :code:`viewer_args` is a parameter that allows us to visualize the saved results as part of the caching
  function. The reason we need this is that displaying the saved result often requires a different (flatter)
  chunk size for fast loading of cross-sectional image, in the above example it is converted from the original
  chunk size e.g. (256, 256, 256) to (1, 4096, 4096) and also requires downsampling for zooming in/out of
  larger images, which the built-in persist() function of dask library does not provide good support of.

Running the Pipeline
********************

See `Setting Up the Script <GettingStarted/setting_up_the_script>`_ to understand boilerplate code used below,
which is required to understand the following example.

Now we have defined a :code:`process` function, the next step is to write our script that uses the pipeline
to segment an input dataset. Note we need a dask cluster and a temporary directory setup before running the
:code:`forward()` method.

.. code-block:: Python

    if __name__ == '__main__':  # Only for main thread, worker threads will not run this
        TMP_PATH = "path/to/temporary/directory"
        import dask
        from dask.distributed import Client
        import napari
        with dask.config.set({'temporary_directory': TMP_PATH}:
            temp_directory = f'{TMP_PATH}/CacheDirectory'

            im = load_im(path)  # this is our input dask.Array object to be segmented
            viewer = napari.Viewer()
            viewer_args = dict(viewer=viewer)
            context_args = dict(
                cache_url=f'{temp_directory}/example_seg_process',
                viewer_args=viewer_args
            )
            await example_seg_process(im, context_args=context_args)

            client.close()
            viewer.show(block=True)

If instead :code:`viewer_args=None` is passed the :code:`example_seg_process()` function will display
nothing, process the image and cache it.

- A process function has signature :code:`process(arg1, ..., argn, context_args)`, where
  arg1 to n are arbitrary arguments and :code:`context_args` is a dictionary
- For parameters that changes how the viewer displays the image, these parameters are provided through
  the :code:`viewer_args` argument of the :code:`context_args` dictionary.
- For parameters that specifies how the image is cached and stored locally (storing is often required
  for display), these parameters are provided through the :code:`storage_options` argument of the
  :code:`context_args` dictionary.

To learn more, see the API pages for :code:`cvpl_tools.im.process`, :code:`cvpl_tools.tools.fs` and
:code:`cvpl_tools.im.ndblock` modules.
