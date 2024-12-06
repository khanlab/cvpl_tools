if __name__ == '__main__':
    import sys
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