.. _setting_up_the_script:

Setting Up the Script
#####################

Writing code using cvpl_tools requires some setup code that configures the dask library and other
utilities we need for our image processing pipeline. Below gives a brief description for the setup of these
utilities and how they can be used when we are writing our image processing pipelines with
:code:`cvpl_tools.im.seg_process.SegProcess` class.

Dask Cluster and temporary directory
************************************

Dask is a multithreaded and distributed computing library, in which temporary results may not fit in
memory. In such cases, they are written in the temporary directory set in
the dask config's :code:`temporary_directory` variable. When working on HPC system on Compute Canada,
make sure this path is set to a /scratch directory where number of file allowed to be created is large
enough.

Dask Client setup is described in the `Dask quickstart <https://distributed.dask.org/en/stable/quickstart.html>`_
page. See `this SO post <https://stackoverflow.com/questions/71470336/using-dask-without-client-client>`_ to
determine if you need to initialize a client or not. Below examples are modified from the simplest setup
in the quickstart guide.

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

The line :code:`if __name__ == '__main__':` ensures only the main thread executes the task creation
code. The line of the :code:`Client()` call spawns worker threads that executes exactly the same script back from
the top. If the guarding statement is not present, the worker re-executes the Client creation code and which
leads to into an infinite loop. However, some complications may stem from this, one of which is described in the
following example: (I encountered this when displaying large multiscale images):

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
            viewer.add_ome_zarr_array_from_path(...)  # add a large ome zarr image
            viewer.show(block=True)  # here all threads available will be used to fetch data to show image

Napari utilizes all threads on the current process to load chunks from the image when we drag mouse to navigation
across chunks. With typical dask setup, however, worker threads spawned by the :code:`Client()` call take up all
threads except one for the main thread. If viewer.show(block=True) is called
on the main thread, then Napari viewer does not get the rest of the threads, and the loading speed is slow
(working but with some lagging). The issue is not in adding the image to the viewer, but in the call to
viewer.show() where the loading happens. I also found that the loading speed is slow regardless if the value
of threads_per_worker is set to 1 or more in the Client() initialization.

My solution: Since a running dask Client object seems to be the cause of the problem, calling client.close()
before viewer.show(block=True) solves the problem:

.. code-block:: Python

    viewer.add_ome_zarr_array_from_path(...)  # adding image itself does not take too much time
    client.close()  # here resources are freed up
    viewer.show(block=True)  # threads are now available to fetch data to show image

This does mean any Napari will not use the cluster if images are added as dask arrays, though Dask arrays
are processed multithreaded by default without the need of a cluster.

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
        sys.stdout = imfs.MultiOutputStream(sys.stdout, logfile_stdout)
        sys.stderr = imfs.MultiOutputStream(sys.stderr, logfile_stderr)

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
may discard once computed, and for the others (like the final output) we may want to cache them on the disk
for access later without having to redo the computation. In order to cache the result, we need a fixed path
that do not change across program executions. The :code:`CacheDirectory` class is one that manages and
assigns paths for these intermediate results, based on their cache ID (cid) and the parent CacheDirectory
they belongs to. :code:`CacheRootDirectory` is a subclass of :code:`CacheDirectory` that acts as the root
of the cache directory structure.

In cvpl_tool's model of caching, there is a root cache directory that is created or loaded when the program
starts to run, and every cache directory may contain many sub-cache-directory or data directories in
which there are intermediate files. To create a cache directory, we write

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.im.fs as imfs
        with imfs.CacheRootDirectory(
              f'{TMP_PATH}/CacheDirectory',
              remove_when_done=False,
              read_if_exists=True) as temp_directory):

            # Use case #1. Create a data directory for caching computation results
            cache_path = temp_directory.cache_subpath(cid='some_cache_path')
            if not cache_path.exists:
                os.makedirs(cache_path.url, exists_ok=True)
                # PUT CODE HERE: Now write your data into cache_path.url and load it back later

            # Use case #2. Create a sub-directory and pass it to other processes for caching
            def multi_step_computation(cache_at: imfs.CacheDirectory):
                cache_path = cache_at.cache_subpath(cid='A')
                if not cache_path.exists:
                    A = computeA()
                    save(cache_path.url, A)
                A = load(cache_path.url)

                cache_path_B = cache_at.cache_subpath(cid='B')
                if not cache_path_B.exists:
                    B = computeBFromA()
                    save(cache_path_B.url, B)
                B = load(cache_path_B.url)
                return B

            sub_temp_directory = temp_directory.cache_subdir(cid='mult_step_cache')
            result = multi_step_computation(cache_at=sub_temp_directory)

After running the above code once, caching files will be created. The second time the code is run, the computation
steps will be skipped. This sort of hierarchical caching is convenient for working with complex processes that
can be hierarchically broken down to smaller and simpler compute steps.

A Quicker Setup
***************

If the amount of code used to setup and tear down the dask client, napari viewer and cache directory is bothering you,
you can use the following code to get a quick start locally. This is currently pretty bare-boned, but should allow you
to run any dask-computation defined in the cvpl_tools library and your custom :code:`SegProcess` functions. The
qsetup.py code automatically creates two log files in your current directory, containing the program's stdout and
stderr, since those capture Dask's distributed print function's text output.

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.im.process.qsetup as qsetup
        # IMPORT YOUR LIBRARIES HERE

        TMP_PATH = "C:/ProgrammingTools/ComputerVision/RobartsResearch/data/lightsheet/tmp"
        with qsetup.PLComponents(TMP_PATH, 'CacheDirectory',
                                 client_args=dict(threads_per_worker=12, n_workers=1),
                                 viewer_args=dict(use_viewer=True)) as plc:
            # DO DASK COMPUTATION, AND SHOW RESULTS IN plc.viewer

            plc.viewer.show(block=True)

If anyone would like more features witht this setup, please let me know.
