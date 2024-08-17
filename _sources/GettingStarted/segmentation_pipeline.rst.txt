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

The SegProcess Class
********************

The **SegProcess** class in module **cvpl_tools.im.seg_process** provides a convenient way for us to define
a step in multi-step image processing pipeline for distributed, interpretable and cached image data analysis.

Consider a function that counts the number of cells in a 3d-block of brightness map:

.. code-block:: Python

    import dask.array as da
    def cell_count(block3d: da.Array):
        """Count the number of cells in a dask 3d brightness image"""
        mask = threshold(block3d, 0.4)  # use simple thresholding to mark pixels where cells are as 1s
        inst = instance_segmentation(mask)  # segment the mask into contours
        cell_cnt = count_inst(inst)  # count each contour as a cell
        return cell_cnt

There are a few issues:

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

SegProcess is designed to address these issues, with the basic idea to integrate visualization as
part of the cell_count function, and cache the result of each step into a file in a **CacheDirectory**.

The class supports the following use cases:

1. dask-support. Inputs are expected to be either numpy array, dask array, or cvpl.im.ndblock.NDBlock
objects. In particular, dask.Array and NDBlock are suitable for parallel or distributed image processing
workflows.

2. integration of Napari. **forward()** function of a SegProcess object has a viewer attribute that
defaults to None. By passing a Napari viewer to this parameter, the forward process will add intermediate
images or centroids to the Napari viewer for easier debugging.

3. intermediate result caching. **CacheDirectory** class provides a hierarchical caching directory,
where each **forward()** call will either create a new directory or load from existing cache directory
based on the **cid** parameter passed to the function.

Now we discuss how to define such a pipeline.

Extending the Pipeline
**********************

The first step of building a pipeline is to break a segmentation algorithm down to steps that process the
image in different formats. As an example, we may implement a pipeline as IN -> BS -> OS -> CC, where:

- IN - Input Image (np.float32) between min=0 and max=1, this is the brightness dask image as input
- BS - Binary Segmentation (3d, np.uint8), this is the binary mask single class segmentation
- OS - Ordinal Segmentation (3d, np.int32), this is the 0-N where contour 1-N each denotes an object; also single class
- CC - Cell Count Map (3d, np.float32), a cell count number (estimate, can be float) for each block

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

How do we go from this plan to actually code these steps? Subclassing **SegProcess** is the recommended way.
This means to create a subclass that defines the **forward()** method, which takes arbitrary inputs
and two optional parameters: cid and viewer.

- cid specifies the subdirectory under the cache directory (set by the **set_tmpdir** method of the base
  class) to save intermediate files. If not provided (cid=None),
  then the cache will be saved in a temporary directory that will be removed when the CacheDirectory is
  closed. If provided, this cache file will persist. Within the forward() method, you should use
  self.tmpdir.cache() and self.tmpdir.cache_im() to create cache files:

  .. code-block:: Python

      class ExampleSegProcess(SegProcess):
          def forward(self, im, cid: str = None, viewer: napari.Viewer = None):
              cache_exists, cache_path = self.tmpdir.cache(is_dir=True, cid=cid)

              # in the case cache does not exists, cache_path.path is an empty path we can create a folder in:
              if not cache_exists:
                  os.makedirs(cache_path.path)
                  result = compute_result(im)
                  save(cache_path.path, result)
              result = load(cache_path.path)

          # ...

- The viewer parameter specifies the napari viewer to display the intermediate results. If not provided
  (viewer=None), then no computation will be done to visualize the image. Within the forward() method, you
  should use viewer.add_labels(), lc_interpretable_napari() or temp_directory.cache_im() while passing in
  viewer_args argument to display your results:

  .. code-block:: Python

      class ExampleSegProcess(SegProcess):
          def forward(self, im, cid: str = None, viewer: napari.Viewer = None):
              result = compute_result(im)
              result = self.tmpdir.cache_im(lambda: result, cid=cid, viewer_args=dict(
                  viewer=viewer,  # The napari viewer, visualization will be skipped if viewer is None
                  is_label=True,  # If True, viewer.add_labels() will be called; if False, viewer.add_image() will be called
                  preferred_chunksize=(1, 4096, 4096),  # image will be converted to this chunksize when saved, and converted back when loaded
                  multiscale=4 if viewer else 0,  # maximum downsampling level of OME ZARR files, necessary for very large images
              ))
              return result

  viewer_args is a parameter that allows us to visualize the saved results as part of the caching function. The
  reason we need this is that displaying the saved result often requires a different (flatter) chunk size for
  fast loading of cross-sectional image, and also requires downsampling for zooming in/out of larger images.

Running the Pipeline
********************

See `Boilerplate Code <GettingStarted/boilerplate>`_ to understand boilerplate code used below. It's required
to do the following example.

Now we have defined a ExampleSegProcess class, the next step is to write our script that uses the pipeline to
segment an input dataset. Note we need a dask cluster and a temporary directory setup before running the
forward() method.

.. code-block:: Python

    if __name__ == '__main__':  # Only for main thread, worker threads will not run this
        TMP_PATH = "path/to/temporary/directory"
        import dask
        from dask.distributed import Client
        import napari
        with (dask.config.set({'temporary_directory': TMP_PATH}),
              imfs.CacheDirectory(
                  f'{TMP_PATH}/CacheDirectory',
                  remove_when_done=False,
                  read_if_exists=True) as temp_directory):

            client = Client(threads_per_worker=12, n_workers=1)

            im = load_im(path)  # this is our input dask.Array object to be segmented
            process = ExampleSegProcess()
            viewer = napari.Viewer()
            process.forward(im, cid='cell_count_cache', viewer=viewer)

            viewer.show(block=True)

            client.close()

To learn more, see the API pages for cvpl_tools.im.seg_process, cvpl_tools.im.fs and
cvpl_tools.im.ndblock modules.
