"""
Contains default, quick setup of the pipeline essential objects to run subclasses of SegProcess class
"""
import dataclasses
import cvpl_tools.im.fs as imfs
from dask.distributed import Client
import napari


@dataclasses.dataclass
class PLComponents:
    tmp_path: str
    cache_root: imfs.CacheRootDirectory
    dask_client: Client
    viewer: napari.Viewer

    def __init__(self, tmp_path, cachedir_name: str, client_args: dict):
        self._cachedir_name = cachedir_name
        self._dask_config = None
        assert isinstance(client_args, dict), f'Expected dictionary, got {type(client_args)}'
        self._client_args = client_args

        self.tmp_path = tmp_path
        self.cache_root = None
        self.dask_client = None
        self.viewer = None

    def __enter__(self):
        """Called using the syntax:

        with PLComponents(...) as plcs:
            ...
        """
        import sys
        import cvpl_tools.im.fs as imfs
        import dask
        from dask.distributed import Client
        import napari

        # set standard output and error output to use log file, since terminal output has some issue
        # with distributed print
        logfile_stdout = open('log_stdout.txt', mode='w')
        logfile_stderr = open('log_stderr.txt', mode='w')
        sys.stdout = imfs.MultiOutputStream(sys.stdout, logfile_stdout)
        sys.stderr = imfs.MultiOutputStream(sys.stderr, logfile_stderr)

        self._dask_config = dask.config.set({'temporary_directory': self.tmp_path})
        self._dask_config.__enter__()  # emulate the with clause which is what dask.config.set is used in

        self.cache_root = imfs.CacheRootDirectory(
            f'{self.tmp_path}/{self._cachedir_name}',
            remove_when_done=False,
            read_if_exists=True,
        )
        self.cache_root.__enter__()

        self.dask_client = Client()
        self.viewer = napari.Viewer(ndisplay=2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dask_client.close()

        self._dask_config.__exit__(exc_type, exc_val, exc_tb)
        self.cache_root.__exit__(exc_type, exc_val, exc_tb)
