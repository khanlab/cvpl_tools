.. _boilerplate:

Boilerplate Code
################

Writing code using cvpl_tools requires some setup code that configures the dask library and other
utilities we need for our image processing pipeline. Below gives a brief description for the setup of these
utilities and how they can be used when we are writing our image processing pipelines with
cvpl_tools.im.seg_process.SegProcess class.

Dask Cluster and temporary directory
************************************

Dask is a multithreaded and distributed computing library, in which temporary results can not all be saved in
memory. When the intermediate results do not fit in memory, they are written in the temporary directory set in
the dask config's temporary_directory variable. When working on HPC system on Compute Canada, make sure
this path is set to a /scratch directory where number of file allowed to be created is large enough.

Setting up the client is described in the `Dask quickstart <https://distributed.dask.org/en/stable/quickstart.html>`_
page. We will use a local cluster setup on a local computer as example.

.. code-block:: Python

    if __name__ == '__main__':
        import dask
        import dask.config
        import dask.array as da
        from dask.distributed import Client

        TMP_PATH = "path/to/tmp/dir"  # CHANGE THIS TO AN EMPTY PATH YOU WOULD LIKE TO USE FOR CACHING
        with dask.config.set({'temporary_directory': TMP_PATH}):
            client = Client(threads_per_worker=6, n_workers=1)

            print((da.zeros((3, 3)) + 1).compute().sum().item())  # will output 9

Note the **if __name__' == '__main__':** line is necessary to ensure only the main thread executes the task creation
code. When dask starts a client in the **Client()** call, it will spawn worker threads that run the same script
this file is in. This creates threads that re-executes the Client creation code into an dead loop if the guarding
statement is not present. However, this approach has some complications, one can be seen from the following
example (one that I've encountered with showing large multiscale images):

.. code-block:: Python

    if __name__ == '__main__':
        import dask
        import dask.config
        import dask.array as da
        import napari
        from dask.distributed import Client

        TMP_PATH = "path/to/tmp/dir"
        with dask.config.set({'temporary_directory': TMP_PATH}):
            client = Client(threads_per_worker=6, n_workers=1)
            viewer = napari.Viewer(ndisplay=2)
            viewer.add_ome_zarr_array_from_path(...)  # add a large OME ZARR image
            viewer.show(block=True)  # here all threads available will be used to fetch data to show image

Napari will utilize all threads on current process to load image, but
on a typical dask setup, the **Client()** call will take up all resources available except one thread left for main.
If image is added to napari and displayed with **viewer.show(block=True)** on the main thread, then Napari
does not get the rest of the threads, and the loading speed is slow (working but with some lagging).

solution: the existence of a dask Client object seems to be the cause of the problem, as loading speed is slow
no matter if threads_per_worker=1 or threads_per_worker=12. Calling client.close() before viewer.show(block=True)
solves the problem:

.. code-block:: Python

    viewer.add_ome_zarr_array_from_path(...)  # adding image itself does not take too much time
    client.close()  # here resources are freed up
    viewer.show(block=True)  # threads are now available to fetch data to show image

This does mean any dask loading will not use the cluster, but usually things will work. See this
`SO post <https://stackoverflow.com/questions/71470336/using-dask-without-client-client>`_ for explanations

Dask Logging Setup
******************

Distributed logging setup. Python's logging module is supported by Dask but I've had some issues to get it
right, so I looked and found Dask provides a simple strategy for debug logging as described in `this page
<https://docs.dask.org/en/latest/how-to/debug.html>`_. The solution is to use the same logging as usual for
the main threads, and use dask.distributed.print to print debugging messages if inside a worker thread. For
convenience I also echo the stdout and stderr outputs into separate logfiles so they will persist even if you
accidentally close the command window. Below is an example:

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.im.fs as imfs
        import numpy as np
        from dask.distributed import print as dprint

        logfile_stdout = open('log_stdout.txt', mode='w')
        logfile_stderr = open('log_stderr.txt', mode='w')
        sys.stdout = fs.MultiOutputStream(sys.stdout, logfile_stdout)
        sys.stderr = fs.MultiOutputStream(sys.stderr, logfile_stderr)

        import dask
        import dask.config
        import dask.array as da
        from dask.distributed import Client

        TMP_PATH = "path/to/tmp/dir"
        with dask.config.set({'temporary_directory': TMP_PATH}):
            client = Client(threads_per_worker=6, n_workers=1)

            print((da.zeros((3, 3)) + 1).compute().sum().item())  # will output 9

            def map_fn(block, block_info=None):
                dprint(f'map_fn is called with input {block}')
                return block + 1

            arr = da.zeros((3, 3), dtype=np.uint8).map_blocks(map_fn, meta=np.array(tuple(), dtype=np.uint8))
            print('result is:', arr.compute())

After running this program, you should see outputs in both the command window and the log_stdout.txt and
log_stderr.txt files under your working directory.

CacheDirectory
**************

Different from Dask's temporary directory, cvpl_tool's CacheDirectory class provides intermediate result
caching APIs. A multi-step segmentation pipeline may produce many intermediate results, for some of them we
may compute once discard, and for the others (like the final output) we may want to cache them on the disk
for access later without having to redo the computation. In order to cache the result, we need a fixed path
that do not change across execution of the program. The **CacheDirectory** class is one that manages and
assigns paths for these intermediate results, based on their cache ID (cid) and the parent CacheDirectory
they belongs to.

In cvpl_tool's model of caching, there is a root cache directory that is created or loaded when the program
starts to run, and every cache directory may contain many sub-cache-directory or data directories within
which are intermediate files. To create a cache directory, we can write

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.im.fs as imfs
        with imfs.CacheDirectory(
              f'{TMP_PATH}/CacheDirectory',
              remove_when_done=False,
              read_if_exists=True) as temp_directory):

            # Use case #1. Create a data directory for caching computation results
            cache_exists, cache_path = temp_directory.cache(is_dir=False, cid='some_cache_path')
            if not cache_exists:
                os.makedirs(cache_path.path, exists_ok=True)
                # PUT CODE HERE: Now write your data into cache_path.path and load it back later

            # Use case #2. Create a sub-directory and pass it to other processes for caching
            def multi_step_computation(cache_at: imfs.CacheDirectory):
                cache_exists, cache_path = cache_at.cache(is_dir=False, cid='A')
                if not cache_exists:
                    A = computeA()
                    save(cache_path.path, A)
                A = load(cache_path.path)

                cache_exists_B, cache_path_B = cache_at.cache(is_dir=False, cid='B')
                if not cache_exists_B:
                    B = computeBFromA()
                    save(cache_path_B.path, B)
                B = load(cache_path_B.path)
                return B

            sub_temp_directory = temp_directory.cache(is_dir=True, cid='mult_step_cache')
            result = multi_step_computation(cache_at=sub_temp_directory)

After running the above code once, caching file will be created. The second time the code is run, the computation steps
will be skipped. This sort of hierarchical caching is convenient for working with complex processes that can be broken
down to smaller and simpler compute steps.
