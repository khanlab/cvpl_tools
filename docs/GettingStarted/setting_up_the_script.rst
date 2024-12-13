.. _setting_up_the_script:

Setting Up the Script
#####################

Writing code using cvpl_tools requires some setup code that configures the dask library and other
utilities we need for our image processing pipeline. Below briefly describes the setup.

Dask Cluster and temporary directory
************************************

cvpl_tools depends on `dask <https://github.com/dask/dask>`_ library, where many functions of
cvpl_tools takes dask array as input or output.

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
code. :code:`Client()` spawns worker threads that executes the code from top. If the guard if is absent,
re-execution of the Client creation will crash the program.

Next problem is relevant to contention of thread resources between dask and Napari:

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
            viewer.add_image(...)  # add a large ome zarr image
            viewer.show(block=True)  # here all threads available will be used to fetch data to show image

Napari utilizes all threads on the current process to load image chunks when we navigation camera
across chunks. With typical dask setup, however, threads spawned by the :code:`Client()` take up all
threads except the main thread. If :code:`viewer.show(block=True)` is called
on the main thread, then Napari viewer does not get the rest of the threads, and loading speed is slow.
The issue is not in adding the image to the viewer, but in the call to
:code:`viewer.show()` where the loading happens. It's slow regardless if the value
of :code:`threads_per_worker` is 1 or more.

Calling :code:`client.close()` to release threads before :code:`viewer.show(block=True)`
solves the problem:

.. code-block:: Python

    viewer.add_image(...)  # adding image itself does not take too much time
    client.close()  # here resources are freed up
    viewer.show(block=True)  # threads are now available to fetch data to show image

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
        import cvpl_tools.tools.fs as tlfs
        import numpy as np
        from dask.distributed import print as dprint

        logfile_stdout = open('log_stdout.txt', mode='w')
        logfile_stderr = open('log_stderr.txt', mode='w')
        sys.stdout = tlfs.MultiOutputStream(sys.stdout, logfile_stdout)
        sys.stderr = tlfs.MultiOutputStream(sys.stderr, logfile_stderr)

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

cache directory
***************

Different from Dask's temporary directory, cvpl_tools.tools.fs provides intermediate result
caching APIs. A multi-step segmentation pipeline may produce many intermediate results, for some of them we
may discard once computed, and for the others (like the final output) we may want to cache them on the disk
for access later without having to redo the computation. In order to cache the result, we need a fixed path
that do not change across program executions. The :code:`cvpl_tools.tools.fs.cdir_init` and
:code:`cvpl_tools.tools.fs.cdir_commit` and ones used to commit and check if the result exist or needs to be
computed from scratch.

In a program, we may cache hierarchically, where there is a root cache directory that is created or loaded
when the program starts to run, and every cache directory contains subdirectories and step-specific caches.

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.tools.fs as tlfs

        # Use case #1. Create a data directory for caching computation results
        cache_path = f'{TMP_PATH}/CacheDirectory/some_cache_path'
        is_commit = tlfs.cdir_init(cache_path).commit
        if not is_commit:
            pass  # PUT CODE HERE: Now write your data into cache_path.url and load it back later

        # Use case #2. Create a sub-directory and pass it to other processes for caching
        def multi_step_computation(cache_at: str):
            cache_path1 = f'{cache_path}/A'
            is_commit1 = tlfs.cdir_init(cache_path1).commit
            if not is_commit1:
                A = computeA()
                save(cache_path1, A)  # note here cache_path1 is a existing directory, not a file
            A = load(cache_path1)

            cache_path2 = f'{cache_path}/B'
            is_commit2 = tlfs.cdir_init(cache_path2).commit
            if not is_commit2:
                B = computeBFromA()
                save(cache_path2, B)  # note here cache_path1 is a existing directory, not a file
            B = load(cache_path2)
            return B

        result = multi_step_computation(cache_at=f'{cache_path}/multi_step_cache')

After running the above code once, caching files will be created. The second time the code is run, the computation
steps will be skipped. This sort of hierarchical caching is convenient for working with complex processes that
can be hierarchically broken down to smaller and simpler compute steps.

A Quicker Setup
***************

You can use the following code to get a quick start locally. This is currently pretty bare-boned, but should allow you
to run any dask-computation defined in the cvpl_tools library and your custom :code:`SegProcess` functions. The
qsetup.py code automatically creates two log files in your current directory, containing the program's stdout and
stderr, since those capture Dask's distributed print function's text output.

.. code-block:: Python

    if __name__ == '__main__':
        import cvpl_tools.im.process.qsetup as qsetup
        import napari
        # IMPORT YOUR LIBRARIES HERE

        TMP_PATH = "C:/ProgrammingTools/ComputerVision/RobartsResearch/data/lightsheet/tmp"
        plc = qsetup.PLComponents(TMP_PATH,
                                  'CacheDirectory',
                                  get_client=lambda: Client(threads_per_worker=12, n_workers=1))
        viewer = napari.Viewer(ndisplay=2)
        # DO DASK COMPUTATION, AND SHOW RESULTS IN viewer
        plc.close()
        viewer.show(block=True)

If anyone would like more features with this setup, please let me know.
