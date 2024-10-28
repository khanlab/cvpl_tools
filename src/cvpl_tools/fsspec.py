"""
In FSSpec, a file can be opened with a file system object along with the path of the file within the file system. The
file system object may be a LocalFileSystem object or remote file systems like Google storage bucket's GCSFileSystem.
The file system and file path can be represented together in a url like "C://path/to/object" or "gcs://project/data".

This file provides utilities to create a AbstractFileSystem object from a possibly remote url, and open file and
create/read subfolders from it.
"""


from __future__ import annotations

from fsspec.core import url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.asyn import AsyncFileSystem
import zarr


class RDirFileSystem(DirFileSystem):
    """Recursive DirFileSystem, where you can use [] operator to open subdirectory as a DirFileSystem object"""

    def __init__(
        self,
        url,
        parent: RDirFileSystem = None,
        **storage_options,
    ):
        """
        Parameters
        ----------
        url: str
            Path to the directory.
        fs: AbstractFileSystem
            An instantiated filesystem to wrap.
        """
        AsyncFileSystem.__init__(self, **storage_options)

        if parent is None:
            base_fs, path = url_to_fs(url)

            if self.asynchronous and not base_fs.async_impl:
                raise ValueError("can't use asynchronous with non-async fs")

            if base_fs.async_impl and self.asynchronous != base_fs.asynchronous:
                raise ValueError("both dirfs and fs should be in the same sync/async mode")
        else:
            assert isinstance(parent, RDirFileSystem)
            base_fs, path = parent.fs, '/'.join((parent.path, url))
            url = '/'.join((parent.url, url))

        self.url = url
        self.path = path
        self.fs = base_fs

    def __getitem__(self, item):
        assert isinstance(item, str), type(item)
        return RDirFileSystem(url=item, parent=self)

    def ensure_dir_exists(self, remove_if_already_exists: bool):
        """
        If a directory does not exist, make a new directory with the name.
        This assumes the parent directory must exists; otherwise a path not
        found error will be thrown.
        Args:
            dir_path: The path of folder
            remove_if_already_exists: if True and the folder already exists, then remove it and make a new one.
        """
        if self.fs.exists(self.path):
            if remove_if_already_exists:
                self.fs.rm(self.path, recursive=True)
                self.mkdir(self.path)
        else:
            self.mkdir(self.path)

    def mkdir(self, path, *args, **kwargs):
        if 'gcs' in self.fs.protocol:
            return self.fs.touch(f'{path}/.gcs_placeholder')
        else:
            return self.fs.mkdir(self._join(path), *args, **kwargs)
