"""
Contains default, quick setup of the pipeline essential objects to run subclasses of SegProcess class
"""
import asyncio
import dataclasses
from typing import Callable

import cvpl_tools.tools.fs as tlfs
from cvpl_tools.fsspec import RDirFileSystem, ensure_rdir_filesystem
from dask.distributed import Client
import napari


@dataclasses.dataclass
class PLComponents:
    tmp_path: tlfs.RDirFileSystem
    cache_root: tlfs.RDirFileSystem
    dask_client: Client
    viewer: napari.Viewer

    def __init__(self, tmp_path: str | tlfs.RDirFileSystem, cachedir_name: str, get_client: Callable):
        """Create a PLComponents object

        on __enter__, the instance will set up necessary components for running most SegProcess classes

        Args:
            tmp_path: temporary path where cache directory and dask temporary files will be written
            cachedir_name: name of the cache directory to be created under tmp_path
            get_client: A callback to return a client instance
        """

        self._cachedir_name = cachedir_name
        self._dask_config = None
        self._get_client = get_client
        self.tmp_path: RDirFileSystem = ensure_rdir_filesystem(tmp_path)
        self.cache_root: RDirFileSystem = self.tmp_path[cachedir_name]

        import sys
        import dask

        # set standard output and error output to use log file, since terminal output has some issue
        # with distributed print
        logfile_stdout = open('log_stdout.txt', mode='w')
        logfile_stderr = open('log_stderr.txt', mode='w')
        sys.stdout = tlfs.MultiOutputStream(sys.stdout, logfile_stdout)
        sys.stderr = tlfs.MultiOutputStream(sys.stderr, logfile_stderr)

        self.tmp_path.ensure_dir_exists(remove_if_already_exists=False)

        self._dask_config = dask.config.set({'temporary_directory': self.tmp_path.url})
        self._dask_config.__enter__()  # emulate the with clause which is what dask.config.set is used in
        self.dask_client = self._get_client()

    def close(self):
        self.dask_client.close()
