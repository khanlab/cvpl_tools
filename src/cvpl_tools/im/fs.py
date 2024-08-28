"""This file provides code for image I/O operations, including multithreaded settings
"""
from __future__ import annotations

import copy
import enum
import json
from typing import Any

import napari
import numpy as np
import cvpl_tools.im.ndblock as cvpl_ndblock
from cvpl_tools.im.ndblock import NDBlock
import dask.array as da
import shutil
import os
from cvpl_tools.fs import ensure_dir_exists as ensure_dir_exists
from cvpl_tools.ome_zarr.napari.zarr_viewer import add_ome_zarr_array_from_path


class ImageFormat(enum.Enum):
    NUMPY = 0
    DASK_ARRAY = 1
    NDBLOCK = 2


def chunksize_to_str(chunksize: tuple[int, ...]):
    return ','.join(str(s) for s in chunksize)


def str_to_chunksize(chunksize_str: str):
    return tuple(int(s) for s in chunksize_str.split(','))


def save(file: str,
         im,
         preferred_chunksize: tuple[int, ...] = None,
         multiscale: int = 0):
    """Save an image object into given path

    Supported im object types:
    - np.ndarray
    - dask.Array
    - cvpl_tools.im.ndblock.NDBlock

    Args:
        file: The full/relative path to the directory to be saved to
        im: Object to be saved
        preferred_chunksize: chunk sizes to save as; will rechunk if different from current size; only applies to
            dask arrays.
        multiscale: The number of downsample layers for save ome-zarr; only applies if the image is a dask image
    """
    if isinstance(im, np.ndarray):
        old_chunksize = im.shape
        fmt = ImageFormat.NUMPY
    elif isinstance(im, da.Array):
        old_chunksize = im.chunksize
        fmt = ImageFormat.DASK_ARRAY
    elif isinstance(im, NDBlock):
        old_chunksize = im.get_chunksize()
        fmt = ImageFormat.NDBLOCK
    else:
        raise ValueError(f'Unexpected input type im {type(im)}')
    if preferred_chunksize is None:
        preferred_chunksize = old_chunksize

    if isinstance(im, np.ndarray):
        NDBlock.save(file, NDBlock(im))
    elif isinstance(im, da.Array):
        if old_chunksize != preferred_chunksize:
            im = im.rechunk(preferred_chunksize)
        NDBlock.save(file, NDBlock(im), downsample_level=multiscale)
    elif isinstance(im, NDBlock):
        if im.get_repr_format() == cvpl_ndblock.ReprFormat.DASK_ARRAY and old_chunksize != preferred_chunksize:
            im = NDBlock(im.get_arr().rechunk(preferred_chunksize))
        NDBlock.save(file, im, downsample_level=multiscale)
    else:
        raise ValueError(f'Unexpected input type im {type(im)}')
    with open(f'{file}/.save_meta.txt', mode='w') as outfile:
        outfile.write(str(fmt.value))
        outfile.write(f'\n{chunksize_to_str(old_chunksize)}\n{chunksize_to_str(preferred_chunksize)}')


def load(file: str):
    """Load an image from the given directory.

    The image is one saved by cvpl_tools.im.fs.save()

    Args:
        file: Full path to the directory to be read from

    Returns:
        Recreated image; this method attempts to keep meta and content of the loaded image stays
        the same as when they are saved
    """
    with open(f'{file}/.save_meta.txt') as outfile:
        items = outfile.read().split('\n')
        fmt = ImageFormat(int(items[0]))
        old_chunksize, preferred_chunksize = str_to_chunksize(items[1]), str_to_chunksize(items[2])
    if fmt == ImageFormat.NUMPY:
        im = NDBlock.load(file).get_arr()
    elif fmt == ImageFormat.DASK_ARRAY:
        im = NDBlock.load(file).get_arr()
        if old_chunksize != preferred_chunksize:
            im = im.rechunk(old_chunksize)
    elif fmt == ImageFormat.NDBLOCK:
        im = NDBlock.load(file)
        if im.get_repr_format() == cvpl_ndblock.ReprFormat.DASK_ARRAY and old_chunksize != preferred_chunksize:
            im = NDBlock(im.get_arr().rechunk(old_chunksize))
    else:
        raise ValueError(f'Unexpected input type im {fmt}')
    return im


def display(file: str, viewer_args: dict):
    """Display an image in the viewer; supports numpy or dask ome zarr image

    The image is one saved by cvpl_tools.im.fs.save()

    Args:
        file: Full path to the directory to be read from
        viewer_args: contains viewer and arguments passed to the viewer's add image functions
    """
    viewer_args = copy.copy(viewer_args)
    viewer: napari.Viewer = viewer_args.pop('viewer')
    layer_args = viewer_args.get('layer_args', {})

    with open(f'{file}/.save_meta.txt') as outfile:
        fmt = ImageFormat(int(outfile.read().split('\n')[0]))
    if fmt == ImageFormat.NUMPY:
        is_numpy = True
    elif fmt == ImageFormat.DASK_ARRAY:
        is_numpy = False
    elif fmt == ImageFormat.NDBLOCK:
        properties = NDBlock.load_properties(f'{file}/properties.json')
        repr_fmt: cvpl_ndblock.ReprFormat = properties['repr_format']
        if repr_fmt == cvpl_ndblock.ReprFormat.NUMPY_ARRAY:
            is_numpy = True
        elif repr_fmt == cvpl_ndblock.ReprFormat.NUMPY_ARRAY:
            is_numpy = False
        else:
            raise ValueError(f'Image to be displayed can not be a dict of blocks that is {repr_fmt}')

    is_label: bool = viewer_args.pop('is_label', False)
    if is_numpy:
        fn = viewer.add_labels if is_label else viewer.add_image
        im = NDBlock.load(file).get_arr()
        fn(im, **layer_args)
    else:
        # image saved by NDBlock.save(file)
        add_ome_zarr_array_from_path(viewer, f'{file}/dask_im', use_zip=False,
                                     kwargs=layer_args | dict(is_label=is_label))


class CachePath:
    """A CachePath class is a pointer to a cached location within a hierarchical directory structure.

    CachePath and CacheDirectory are two classes that implements the file-directory programming pattern,
    where CacheDirectory is a subclass of CachePath and contains zero or more CachePath as its children.
    To create a CachePath object, use the CacheDirectory's cache() function to allocate a new or find
    an existing cache location.
    """

    def __init__(self, path: str, meta: dict = None):
        """Create a CachePath object that manages meta info about the cache file or directory

        Args:
            path: The path associated with this CachePath object
            meta: The meta information associated with this object; will be automatically inferred
                from the path (only able to do so in some situations) if None is provided
        """
        self._path = path

        if meta is None:
            meta = CachePath.meta_from_filename(os.path.split(path)[1])

        self._meta = meta
        for key in ('is_dir', 'is_tmp', 'cid'):
            assert key in meta, f'Missing key {key}'

    @property
    def path(self):
        """Obtain the os path under which you can create a directory

        The first time a CachePath object is created for this path, cache_path.path will point to an empty location;
        second time onwards if the directory is not removed, then the returned cache_path.path will point to the
        previously existing directory
        """
        return self._path

    @property
    def is_dir(self):
        """Returns True if this is a directory object instead of a file.

        In other words, this function returns False if this is a leaf node.
        """
        return self._meta['is_dir']

    @property
    def is_tmp(self):
        return self._meta['is_tmp']

    @property
    def cid(self):
        return self._meta['cid']

    @property
    def meta(self):
        return self._meta

    @staticmethod
    def meta_from_filename(file: str, return_none_if_malform=False) -> dict[str, Any] | None:
        """Retrieve meta information from the path

        Args:
            file: filename of the (existing to planning to be created) CachePath object
            return_none_if_malform: If True, return None instead of throwing error if a malformed
                filename is given

        Returns:
            A dictionary of the meta information
        """
        if file.startswith('file_'):
            is_dir = False
            rest = file[len('file_'):]
        elif file.startswith('dir_'):
            is_dir = True
            rest = file[len('dir_'):]
        else:
            if return_none_if_malform:
                return None
            else:
                raise ValueError(f'path is not expected when parsing is_file: {file}')
        if rest.startswith('tmp_'):
            is_tmp = True
            rest = rest[len('tmp_'):]
        elif rest.startswith('cache_'):
            is_tmp = False
            rest = rest[len('cache_'):]
        else:
            if return_none_if_malform:
                return None
            else:
                raise ValueError(f'path is not expected when parsing is_tmp: {file}')
        return dict(
            is_dir=is_dir,
            is_tmp=is_tmp,
            cid=rest
        )

    @staticmethod
    def filename_form_meta(meta: dict[str, Any]) -> str:
        """Obtain filename from the meta dict

        Args:
            meta: The dictionary containing meta information for the CachePath object

        Returns:
            A string as the filename of the cached directory or file
        """
        s1 = 'dir_' if meta['is_dir'] else 'file_'
        s2 = 'tmp_' if meta['is_tmp'] else 'cache_'
        cid = meta['cid']
        return f'{s1}{s2}{cid}'


class CacheDirectory(CachePath):
    """A CacheDirectory is a hierarchical directory structure, corresponding to a directory in the os

    CachePath and CacheDirectory are two classes that implements the file-directory programming pattern.
    """

    def __init__(self, path: str, remove_when_done: bool = True, read_if_exists: bool = True,
                 cache_level: int | float = np.inf):
        """Creates a CacheDirectory instance

        Args:
            path: The os path to which the directory is to be created; must be empty if read_if_exists=True
            remove_when_done: If True, the entire directory will be removed when it is closed by __exit__; if
                False, then only the temporary folders within the directory will be removed. (The entire subtree
                will be traversed to find any file or directory whose is_tmp is True and they will be removed)
            read_if_exists: If True, will read from the existing directory at the given path
            cache_level: specifies how much caching to be done; caching operations with level > this will be ignored;
                default to inf (cache all)
        """

        super().__init__(path, dict(
            is_dir=True,
            is_tmp=remove_when_done,
            cid='_RootDirectory'
        ))
        self.cur_idx = 0
        self.read_if_exists = read_if_exists
        self.cache_level = cache_level
        self.children: dict[str, CachePath] = {}

        ensure_dir_exists(path, remove_if_already_exists=False)
        path = self.path
        if self.read_if_exists:
            self.children = CacheDirectory.children_from_path(path, self.cache_level)
        else:
            for _ in os.listdir(path):
                raise FileExistsError('when read_if_exists=False, directory must not contain existing files, '
                                      f'please check if any file exists under {path}.')

    def get_children_json(self) -> dict:
        children_json = {}
        for key, child in self.children.items():
            if child.is_dir:
                child: CacheDirectory
                children_json[key] = dict(
                    children=child.get_children_json(),
                    meta=child.meta
                )
            else:
                children_json[key] = child.meta
        return children_json

    def get_children_str(self):
        return json.dumps(self.get_children_json(), indent=2)

    @staticmethod
    def children_from_path(path: str, cache_level: int | float = np.inf) -> dict[str, CachePath]:
        """Examine an existing directory path, return recursively all files and directories as json.

        Args:
            path: The path to be examined
            cache_level: Level of caching to assign to all descendants

        Returns:
            Returned json dictionary contains a hierarchical str -> CachePath map; use CachePath.is_dir to
            determine if they contain more children
        """
        children = {}
        for filename in os.listdir(path):
            subpath = f'{path}/{filename}'
            meta = CachePath.meta_from_filename(filename, return_none_if_malform=True)
            if meta is not None:
                if meta['is_dir']:
                    child = CacheDirectory(subpath,
                                           remove_when_done=meta['is_tmp'],
                                           read_if_exists=True,
                                           cache_level=cache_level)
                    child.children = CacheDirectory.children_from_path(subpath, cache_level)
                else:
                    child = CachePath(subpath, meta)
                children[meta['cid']] = child
        return children

    def __getitem__(self, cid: str) -> CachePath | CacheDirectory:
        """Get a CachePath object by its cid"""
        return self.children[cid]

    def __contains__(self, item: str):
        """Checks if an object is cached"""
        return item in self.children

    def cache(self,
              is_dir=False,
              cid: str = None
              ) -> tuple[bool, CachePath | CacheDirectory]:
        """Return a directory that is guaranteed to be empty within the temporary directory

        This is the interface to create new CachePath or CacheDirectory within this directory.
        The directory will not be immediately created but need to be done manually if is_dir=False

        Args:
            is_dir: If False, this creates a subfolder that have no children; if True, this creates
                a CacheDirectory recursively
            cid: If specified, will attempt to find cache if already exists; otherwise a temporary
                cache will be returned

        Returns:
            A tuple (is_cached, CachePath), is_cached giving whether the file is cached or is newly
            created. If is_cached is True, then the user should directly read from the cached file
            instead
        """
        is_tmp = cid is None
        if is_tmp:
            cid = f'_{self.cur_idx}'
            self.cur_idx += 1
        else:
            if cid in self.children:
                file = self.children[cid]
                assert file.is_dir == is_dir, f'Unexpected file/directory at {file.path}'
                return True, self.children[cid]

        meta = dict(
            is_dir=is_dir,
            is_tmp=is_tmp,
            cid=cid
        )
        filename = CachePath.filename_form_meta(meta)
        tmppath = f'{self.path}/{filename}'
        if is_dir:
            cache_path = CacheDirectory(tmppath, is_tmp, self.read_if_exists)
        else:
            cache_path = CachePath(tmppath, meta)
        self.children[cid] = cache_path
        return False, cache_path

    def cache_im(self,
                 fn,
                 cid: str = None,
                 save_fn=save,
                 load_fn=load,
                 cache_level: int | float = 0,
                 viewer_args: dict = None):
        """Caches an image object

        Args:
            fn: Computes the image if it's not already cached
            cid: The cache ID within this directory
            save_fn: fn(file: str, im) Used to save the image to file
            load_fn: fn(file: str) Used to load the image from file
            cache_level: cache level of this operation; note even if the caching is skipped, if there is
                a cache file already available on disk then the file will still be read
            viewer_args: contains viewer and arguments passed to the viewer's add image functions

        Returns:
            The cached image loaded
        """
        if viewer_args is None:
            viewer_args = {}
        else:
            viewer_args = copy.copy(viewer_args)  # since we will pop off some attributes

        preferred_chunksize = viewer_args.pop('preferred_chunksize', None)
        multiscale = viewer_args.pop('multiscale', 0)

        is_cached, cache_path = self.cache(is_dir=False, cid=cid)
        raw_path = cache_path.path
        skip_cache = viewer_args.get('skip_cache', False) or cache_level > self.cache_level
        if not is_cached:
            im = fn()
            if skip_cache:
                return im
            save_fn(raw_path, im, preferred_chunksize=preferred_chunksize, multiscale=multiscale)

        assert os.path.exists(raw_path), f'Directory should be created at path {raw_path}, but it is not found'
        if not skip_cache and viewer_args.get('viewer', None) is not None:
            viewer_args['layer_args'] = copy.copy(viewer_args.get('layer_args', {}))
            viewer_args['layer_args'].setdefault('name', cid)
            display(raw_path, viewer_args)

        loaded = load_fn(raw_path)

        return loaded

    def remove_tmp(self):
        """traverse all subnodes and self, removing those with is_tmp=True"""
        if self.is_tmp:
            shutil.rmtree(self.path)
        else:
            for ch in self.children.values():
                if ch.is_tmp:
                    shutil.rmtree(ch.path)
                elif ch.is_dir:
                    assert isinstance(ch, CacheDirectory)
                    ch.remove_tmp()

    def __enter__(self):
        """Called using the syntax:

        with CacheDirectory(...) as cache_dir:
            ...
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_tmp()


class MultiOutputStream:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            file.write(message)

    def flush(self):
        for file in self.files:
            file.flush()
