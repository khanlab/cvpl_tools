.. _boilerplate:

Boilerplate Code
################

Each time we write a new script, we need to include some setup code that configures the dask library and other
utilities we need for our image processing pipeline. Below gives a brief description for the setup of these
utilities and how they can be used when we are writing our image processing pipelines with
cvpl_tools.im.seg_process.SegProcess class.

Dask Cluster and temporary directory
************************************

Dask is a multithreaded and distributed computing library, in which temporary results can not all be saved in
memory. When the intermediate results do not fit in memory, they are written in the temporary directory set in
the dask config's temporary_directory variable. When working on HPC system on Compute Canada, be careful that
this path is set to a /scratch directory where number of file allowed to be created is large enough.

Setting up the client is described in the `Dask quickstart <https://distributed.dask.org/en/stable/quickstart.html>`_
page. We will use local laptop as example.

.. code-block:: Python

    import dask
    import dask.config
    import dask.array as da
    with dask.config.set({'temporary_directory': TMP_PATH}):
        client = Client(threads_per_worker=6, n_workers=1)

        print((da.zeros((3, 3)) + 1).compute().sum().item())  # will output 9

CacheDirectory
**************

Different from Dask's temporary directory, cvpl_tool's CacheDirectory class provides intermediate result
caching APIs. A multi-step segmentation pipeline may produce many intermediate results, for some of them we
may compute once discard, and for the others (like the final output) we may want to cache them on the disk
for access later without having to redo the computation. In order to cache the result, we need a fixed path
that do not change across execution of the program. The **CacheDirectory** class is one that manages and
assigns paths for these intermediate results, based on their cache ID (cid) and the parent CacheDirectory
they belongs to.

