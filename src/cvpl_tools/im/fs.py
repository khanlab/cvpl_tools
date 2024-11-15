"""
This file provides code for image I/O operations, including multithreaded settings
"""
from __future__ import annotations

import copy
import enum
import json
from typing import Any

import numpy as np
import cvpl_tools.im.ndblock as cvpl_ndblock
from cvpl_tools.im.ndblock import NDBlock
import dask.array as da
from cvpl_tools.fsspec import RDirFileSystem
from distributed import get_client
import asyncio
import inspect


class ImageFormat(enum.Enum):
    NUMPY = 0
    DASK_ARRAY = 1
    NDBLOCK = 2


def chunksize_to_str(chunksize: tuple[int, ...]):
    return ','.join(str(s) for s in chunksize)


def str_to_chunksize(chunksize_str: str):
    return tuple(int(s) for s in chunksize_str.split(','))


def persist(im, storage_options: dict = None):
    """Use dask built-in persist to save the image object

    Args:
        im: Image object to be saved
        storage_options: Under which 'compressor' specifies the compression algorithm to use for saving

    Returns:
        Image object loaded from persist(), or return the input if is numpy
    """
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, da.Array):
        return im.persist(compressor=storage_options.get('compressor'))
    elif isinstance(im, NDBlock):
        return im.persist(compressor=storage_options.get('compressor'))
    else:
        raise ValueError(f'Unrecognized object type, im={im}')


async def save(file: str,
               im,
               storage_options: dict = None):
    """Save an image object into given path

    Supported im object types:
    - np.ndarray
    - dask.Array
    - cvpl_tools.im.ndblock.NDBlock

    Storage options
        preferred_chunksize (tuple[int, ...]) = None
            chunk sizes to save as; will rechunk if different from current size; only applies to dask arrays.
        multiscale (int) = 0
            The number of downsample layers for save ome-zarr; only applies if the image is a dask image
        compressor = None
            The compressor to use to compress array or chunks

    Args:
        file: The full/relative path to the directory to be saved to
        im: Object to be saved
        storage_options: Specifies options in saving method and saved file format
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

    if storage_options is None:
        preferred_chunksize = old_chunksize
    else:
        preferred_chunksize = storage_options.get('preferred_chunksize') or old_chunksize

    if isinstance(im, np.ndarray):
        await NDBlock.save(file, NDBlock(im), storage_options=storage_options)
    elif isinstance(im, da.Array):
        if old_chunksize != preferred_chunksize:
            im = im.rechunk(preferred_chunksize)
        await NDBlock.save(file, NDBlock(im), storage_options=storage_options)
    elif isinstance(im, NDBlock):
        if im.get_repr_format() == cvpl_ndblock.ReprFormat.DASK_ARRAY and old_chunksize != preferred_chunksize:
            im = NDBlock(im.get_arr().rechunk(preferred_chunksize))
        await NDBlock.save(file, im, storage_options=storage_options)
    else:
        raise ValueError(f'Unexpected input type im {type(im)}')

    fs = RDirFileSystem(file)
    with fs.open('.save_meta.txt', mode='w') as outfile:
        outfile.write(str(fmt.value))
        outfile.write(f'\n{chunksize_to_str(old_chunksize)}\n{chunksize_to_str(preferred_chunksize)}')

        compressor = storage_options.get('compressor')
        outfile.write(f'\ncompressor:{repr(compressor)}')


def load(file: str, storage_options: dict = None):
    """Load an image from the given directory.

    The image is one saved by cvpl_tools.im.fs.save()

    Args:
        file: Full path to the directory to be read from
        storage_options: Specifies options in saving method and saved file format

    Returns:
        Recreated image; this method attempts to keep meta and content of the loaded image stays
        the same as when they are saved
    """
    fs = RDirFileSystem(file)
    with fs.open(f'.save_meta.txt', 'r') as infile:
        items = infile.read().split('\n')
        fmt = ImageFormat(int(items[0]))
        old_chunksize, preferred_chunksize = str_to_chunksize(items[1]), str_to_chunksize(items[2])
    if fmt == ImageFormat.NUMPY:
        im = NDBlock.load(file, storage_options=storage_options).get_arr()
    elif fmt == ImageFormat.DASK_ARRAY:
        im = NDBlock.load(file, storage_options=storage_options).get_arr()
        if old_chunksize != preferred_chunksize:
            im = im.rechunk(old_chunksize)
    elif fmt == ImageFormat.NDBLOCK:
        im = NDBlock.load(file, storage_options=storage_options)
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
    import napari
    import cvpl_tools.ome_zarr.napari.add as napari_add_ome_zarr

    viewer_args = copy.copy(viewer_args)
    viewer: napari.Viewer = viewer_args.pop('viewer')
    layer_args = viewer_args.get('layer_args', {})

    fs = RDirFileSystem(file)
    with fs.open(f'.save_meta.txt', mode='r') as infile:
        fmt = ImageFormat(int(infile.read().split('\n')[0]))
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
        napari_add_ome_zarr.subarray_from_path(viewer, f'{file}/dask_im', use_zip=False, merge_channels=True,
                                               kwargs=layer_args | dict(is_label=is_label))


# -------------------------------------File Caching---------------------------------------------


async def cache_im(fn,
                   cpath: CachePath,
                   cache_level: int | float = 0,
                   viewer_args: dict = None):
    """Caches an image object

    Args:
        fn: Computes the image if it's not already cached
        cpath: The cache ID within this directory
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
    multiscale = viewer_args.pop('multiscale', None)
    storage_options = viewer_args.pop('storage_options', {})
    if preferred_chunksize is not None:
        storage_options['preferred_chunksize'] = preferred_chunksize
    if multiscale is not None:
        storage_options['multiscale'] = multiscale

    raw_path = cpath.abs_path
    skip_cache = viewer_args.get('skip_cache', False) or cache_level > viewer_args.get('cache_level', np.inf)
    if skip_cache:
        if inspect.iscoroutine(fn):
            im = await fn
        elif inspect.iscoroutinefunction(fn):
            im = await fn()
        else:
            im = fn()

        im = get_client().persist(im)
        return im
    elif not cpath.exists:
        if inspect.iscoroutine(fn):
            im = await fn
        elif inspect.iscoroutinefunction(fn):
            im = await fn()
        else:
            im = fn()

        await save(raw_path, im, storage_options)

    if not skip_cache and viewer_args.get('viewer') is not None:
        viewer_args['layer_args'] = copy.copy(viewer_args.get('layer_args', {}))
        viewer_args['layer_args'].setdefault('name', cpath.cid)
        display(raw_path, viewer_args)

    loaded = load(raw_path, storage_options)

    return loaded


class CachePath:
    """A CachePath class is a pointer to a cached location within a hierarchical directory structure.

    CachePath and CacheDirectory are two classes that implements the file-directory programming pattern,
    where CacheDirectory is a subclass of CachePath and contains zero or more CachePath as its children.
    To create a CachePath object, use the CacheDirectory's cache() function to allocate a new or find
    an existing cache location.

    CachePath as its name suggests is not a folder but only a path. Creating a CachePath object will not
    create the associated file automatically.
    """

    def __init__(self, root: CacheRootDirectory, path_segs: tuple[str, ...], meta: dict = None,
                 parent: CacheDirectory | None = None, exists: bool = False,
                 fs: RDirFileSystem = None):
        """Create a CachePath object that manages meta info about the cache file or directory

        Args:
            root: The CacheDirectory that is the root of the directory tree where this file locates under
            path_segs: The path segments associated with this CachePath object, relative to the root
            meta: The meta information associated with this object; will be automatically inferred
                from the path (only able to do so in some situations) if None is provided
            parent: parent of the CachePath object in the cache directory tree; None if this is root
            exists: True if this object references a existing cached object, False if the referenced
                object is newly created
        """
        assert isinstance(path_segs, tuple), f'Expected tuple for path_segs, got {type(path_segs)}'
        self._root = root
        self._path_segs = path_segs

        if fs is None:
            # initialize fsspec's file system object for file system access operations
            self._rel_path = '/'.join(self._path_segs)
            self._fs = self._root._fs[self._rel_path]
        else:
            self._fs = fs

        if meta is None:
            filename = path_segs[-1]
            meta = CachePath.meta_from_filename(filename)

        self._meta = meta
        self._parent = parent
        self._exists = exists
        for key in ('is_dir', 'is_tmp', 'cid'):
            assert key in meta, f'Missing key {key}'

    @property
    def root(self) -> CacheRootDirectory:
        return self._root

    @property
    def parent(self) -> CacheDirectory | None:
        return self._parent

    @property
    def exists(self) -> bool:
        return self._exists

    @property
    def filename(self) -> str:
        """Returns the filename, the last segment of the path

        This is typically cid prepended with is_dir and is_tmp information; do not call this
        on root directory.
        """
        return self._path_segs[-1]

    @property
    def fs(self) -> RDirFileSystem:
        return self._fs

    @property
    def rel_path(self) -> str:
        """Obtain the relative os path to the root"""
        return self._rel_path

    @property
    def url(self):
        """Obtain the url associated with this CachePath object

        The first time this url is associated with a CachePath object, it's supposed to be empty and can be used as the
        address to store cache files. Afterward the next time app starts the cache can be read from this url.
        """
        return self._fs.url

    @property
    def abs_path(self):
        """Obtain the absolute path associated with this CachePath object

        The first time this url is associated with a CachePath object, it's supposed to be empty and can be used as the
        address to store cache files. Afterward the next time app starts the cache can be read from this url.
        """
        return self._fs.path

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

    def __init__(self,
                 cid: str,
                 root: CacheRootDirectory,
                 path_segs: tuple[str, ...],
                 remove_when_done: bool = True,
                 read_if_exists: bool = True,
                 parent: CacheDirectory | None = None,
                 exists: bool = False):
        """Creates a CacheDirectory instance
        
        Unlike CachePath, calling the __init__ of CacheDirectory will create a physical directory 
        on the disk if one does not already exist.

        Args:
            cid: cid of the directory
            root: The root directory containing this tree of cache directories
            path_segs: The os path to which the directory is to be created; must be empty if read_if_exists=True
            remove_when_done: If True, the entire directory will be removed when it is closed by __exit__; if
                False, then only the temporary folders within the directory will be removed. (The entire subtree
                will be traversed to find any file or directory whose is_tmp is True, and they will be removed)
            read_if_exists: If True, will read from the existing directory at the given path
            parent: parent of the directory in the directory structure; None if is root
            exists: True if the directory is read from an already existing cache; False if it is created anew
        """
        super().__init__(
            root=root,
            path_segs=path_segs,
            meta=dict(
                is_dir=True,
                is_tmp=remove_when_done,
                cid=cid
            ),
            parent=parent,
            exists=exists
        )
        self.cur_idx = 0
        self.read_if_exists = read_if_exists
        self.children: dict[str, CachePath] = {}

        if self.read_if_exists:
            self._fs.ensure_dir_exists(remove_if_already_exists=False)
            self.children = self.children_from_path(
                prefix_path_segs=self._path_segs)
        else:
            assert not exists, 'when read_if_exists=False, directory must be created so must not already exists, ' \
                               f'please check if any file exists under {self.url}.'

    def children_from_path(self, prefix_path_segs: tuple[str, ...] = None) -> dict[str, CachePath]:
        """Examine an existing directory path, return recursively all files and directories as dict.

        Args:
            prefix_path_segs: segments prefixing the filenames found under this directory

        Returns:
            Returned json dictionary contains a hierarchical str -> CachePath map; use CachePath.is_dir to
            determine if they contain more children
        """
        children = {}
        for di in self._fs.ls(''):
            filename = di['name']
            subpath_segs = prefix_path_segs + (filename,)
            meta = CachePath.meta_from_filename(filename, return_none_if_malform=True)
            if meta is not None:
                if meta['is_dir']:
                    child = CacheDirectory(
                        cid=meta['cid'],
                        root=self.root,
                        path_segs=subpath_segs,
                        remove_when_done=meta['is_tmp'],
                        read_if_exists=True,
                        parent=self,
                        exists=True
                    )
                    child.children = child.children_from_path(prefix_path_segs=subpath_segs)
                else:
                    child = CachePath(root=self.root, path_segs=subpath_segs, meta=meta, parent=self, exists=True)
                children[meta['cid']] = child
        return children

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

    def __getitem__(self, cid: str) -> CachePath | CacheDirectory:
        """Get a CachePath object by its cid"""
        return self.children[cid]

    def __contains__(self, item: str):
        """Checks if an object is cached"""
        return item in self.children

    def _create_cache(self,
                      is_dir=False,
                      cid: str = None
                      ) -> CachePath | CacheDirectory:
        """Return a directory that is guaranteed to be empty within the temporary directory

        This is the interface to create new CachePath or CacheDirectory within this directory.
        The directory will not be immediately created but need to be done manually if is_dir=False.
        When cid=None, the first returned variable (is_cached) will always be False

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
                assert file.is_dir == is_dir, f'Unexpected file/directory at {file.abs_path}'
                return self.children[cid]

        # create a new cached object, exists=False
        meta = dict(
            is_dir=is_dir,
            is_tmp=is_tmp,
            cid=cid
        )
        filename = CachePath.filename_form_meta(meta)
        subpath_segs = self._path_segs + (filename,)
        if is_dir:
            cache_path = CacheDirectory(
                cid=cid,
                root=self.root,
                path_segs=subpath_segs,
                remove_when_done=is_tmp,
                read_if_exists=self.read_if_exists,
                parent=self,
                exists=False
            )
        else:
            cache_path = CachePath(
                self.root,
                subpath_segs,
                meta=meta,
                parent=self,
                exists=False
            )
        self.children[cid] = cache_path
        return cache_path

    def cache(self, cid: str = None):
        """Similar to _create_cache(), but use an intermediate object to offset the decision of is_dir
        
        Often we pass an empty CacheDirectory object to a function, which saves the intermediate results
        in the empty directory. When the function saves a single file, it will prefer to save to a
        CachePath instead of a CacheDirectory. This makes creating CacheDirectory before function call
        inappropriate. To avoid this, we pass in a struct that contains the parent directory and the
        desired path to create either a CachePath or a CacheDirectory, and offset the decision to
        the function itself.

        Args:
            cid: The cid of either the CachePath or the CacheDirectory

        Returns:
            A CachePointer object of the cache to be created
        """
        return CachePointer(self, cid)

    def cache_subpath(self, cid: str = None) -> CachePath:
        """Wrapper, calls cache(is_dir=False)

        See the docs for CacheDirectory.cache() for more information.
        """
        return self._create_cache(is_dir=False, cid=cid)

    def cache_subdir(self, cid: str = None) -> CacheDirectory:
        """Wrapper, calls cache(is_dir=True)

        See the docs for CacheDirectory.cache() for more information.
        """
        return self._create_cache(is_dir=True, cid=cid)

    async def cache_im(self,
                       fn,
                       cid: str = None,
                       cache_level: int | float = 0,
                       viewer_args: dict = None):
        """Caches an image object

        Args:
            fn: Computes the image if it's not already cached
            cid: The cache ID within this directory
            cache_level: cache level of this operation; note even if the caching is skipped, if there is
                a cache file already available on disk then the file will still be read
            viewer_args: contains viewer and arguments passed to the viewer's add image functions

        Returns:
            The cached image loaded
        """
        cpath = self.cache_subpath(cid=cid)
        return await cache_im(fn, cpath, cache_level, viewer_args)

    def remove_tmp(self):
        """traverse all subnodes and self, removing those with is_tmp=True"""
        if self.is_tmp:
            self._fs.rm('', recursive=True)
        else:
            for ch in self.children.values():
                if ch.is_tmp:
                    ch._fs.rm('', recursive=True)
                elif ch.is_dir:
                    assert isinstance(ch, CacheDirectory)
                    ch.remove_tmp()


class CacheRootDirectory(CacheDirectory):
    def __init__(self,
                 path: str,
                 remove_when_done: bool = True,
                 read_if_exists: bool = True):
        """Creates a CacheDirectory instance

        For users of this class, use this interface to create CacheDirectory

        Args:
            path: The os path to which the directory is to be created; must be empty if read_if_exists=True
            remove_when_done: If True, the entire directory will be removed when it is closed by __exit__; if
                False, then only the temporary folders within the directory will be removed. (The entire subtree
                will be traversed to find any file or directory whose is_tmp is True and remote them)
            read_if_exists: If True, will read from the existing directory at the given path
        """
        self._fs = RDirFileSystem(url=path)
        super().__init__(
            cid='_RootDirectory',  # not used anywhere else, put a non-empty cid for debugging
            root=self,
            path_segs=tuple(),
            remove_when_done=remove_when_done,
            read_if_exists=read_if_exists,
            parent=None,
            exists=self._fs.exists('')
        )

    def __enter__(self):
        """Called using the syntax:

        with CacheRootDirectory(...) as cache_dir:
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


class CachePointer(CachePath):
    def __init__(self, cdir: CacheDirectory, cid: str | None):
        super().__init__(
            root=cdir.root,
            path_segs=cdir._path_segs + (cid,),
            meta=dict(
                is_tmp=cid is None,
                is_dir=False,
                cid=cid
            ),
            parent=cdir,
            exists=cid is not None and cid in cdir.children
        )

    def subpath(self):
        """Create a CachePath under the parent of this pointer, at the location pointed by this pointer"""
        return self.parent.cache_subpath(cid=self.cid)

    def subdir(self):
        """Create a CacheDirectory under the parent of this pointer, at the location pointed by this pointer"""
        return self.parent.cache_subdir(cid=self.cid)

    async def im(self, *args, **kwargs):
        return await self.parent.cache_im(cid=self.cid, *args, **kwargs)

    @property
    def is_dir(self):
        return NotImplementedError("A pointer offsets decision of is_dir, don't call is_dir on CachePointer")
