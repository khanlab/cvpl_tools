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
            url = url.rstrip('/')
            base_fs, path = url_to_fs(url)

            if self.asynchronous and not base_fs.async_impl:
                raise ValueError("can't use asynchronous with non-async fs")

            if base_fs.async_impl and self.asynchronous != base_fs.asynchronous:
                raise ValueError("both dirfs and fs should be in the same sync/async mode")
        else:
            assert isinstance(parent, RDirFileSystem)
            base_fs, path = parent.fs, '/'.join((parent.path.rstrip('/'), url)).rstrip('/')
            url = '/'.join((parent.url.rstrip('/'), url)).rstrip('/')

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
                self.makedirs_cur()
        else:
            self.makedirs_cur()

    def makedirs_cur(self, exists_ok: bool = False):
        if exists_ok and self.exists(''):
            return
        if 'gcs' in self.fs.protocol:
            return self.fs.touch(f'{self.path}/.gcs_placeholder')
        else:
            return self.fs.makedirs(self.path)


def copyfile(src: str | RDirFileSystem, tgt: str | RDirFileSystem, chunksize: int | None = None):
    """Copy a file from src to tgt

    reference: martindurant's reply in thread https://github.com/fsspec/filesystem_spec/issues/909
    """
    if isinstance(src, str):
        src = RDirFileSystem(src)
    if isinstance(tgt, str):
        tgt = RDirFileSystem(tgt)
    with src.open('', mode='rb') as src_file, tgt.open('', mode='wb') as tgt_file:
        while True:
            data = src_file.read(chunksize)
            if not data:
                break
            tgt_file.write(data)
