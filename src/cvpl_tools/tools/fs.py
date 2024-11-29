"""
This file provides code for image I/O operations, including multithreaded settings
"""
from __future__ import annotations

import copy
import dataclasses
import enum

import numpy as np
import cvpl_tools.im.ndblock as cvpl_ndblock
from cvpl_tools.im.ndblock import NDBlock, ReprFormat
import dask.array as da
from cvpl_tools.fsspec import RDirFileSystem, ensure_rdir_filesystem
import inspect
import logging
from datetime import datetime


DEBUG = True
logger = logging.getLogger('coiled')


# ---------------------------------------Local Caching Mechanisms--------------------------------------------


TIMESTAMP_FILE_NAME = '.cache_url_timestamp'


@dataclasses.dataclass
class CDirQueryResult:
    commit: None | datetime  # None if result is not committed; otherwise return the finish timestamp


def cdir_commit(cache_url: str | RDirFileSystem):
    """Create a commit timestamp"""
    fs = ensure_rdir_filesystem(cache_url)
    with fs.open(TIMESTAMP_FILE_NAME, mode='w') as outfile:
        outfile.write(datetime.now().isoformat())


def cdir_query(cache_url: str | RDirFileSystem) -> CDirQueryResult:
    """Query the commit status of a directory

    1. If the url points to a directory that does not exists or does not have '.cache_url_timestamp' file created
    under, then the returned query object will have `query.commit = None`
    2. Otherwise the directory is a committed directory, and the returned query object will contain the time stamp
    of commit in its commit attribute `query.commit`

    Args:
        cache_url: The location to be queried

    Returns:
        A CDirQueryResult containing the commit status of the given cache url will be returned
    """
    fs = ensure_rdir_filesystem(cache_url)
    if not fs.exists(path=TIMESTAMP_FILE_NAME):
        result = CDirQueryResult(commit=None)
    else:
        with fs.open(TIMESTAMP_FILE_NAME, mode='r') as outfile:
            timestamp = datetime.fromisoformat(outfile.read())
        result = CDirQueryResult(commit=timestamp)
    return result


def cdir_init(cache_url: str | RDirFileSystem) -> CDirQueryResult:
    """Returned CDirQueryResult is the same as calling cdir_query()

    Function behavior:
    1. If directory does not exist, make a new empty directory
    2. If directory exists but timestamp is not found (not committed), then remove it and make a new directory
    3. If commit timestamp is found, do nothing

    Args:
        cache_url: Pointing to the directory location
    """
    fs = ensure_rdir_filesystem(cache_url)
    query = cdir_query(fs)
    if query.commit is None:
        fs.ensure_dir_exists(remove_if_already_exists=True)
    return query


# -----------------------------------save, load, persist and display-----------------------------------------


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
    if DEBUG:
        logger.error(f'persist() being called on image of type {type(im)}')
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, da.Array):
        return im.persist(compressor=storage_options.get('compressor'))
    elif isinstance(im, NDBlock):
        return im.persist(compressor=storage_options.get('compressor'))
    else:
        raise ValueError(f'Unrecognized object type, im={im}')


async def save(file: str | RDirFileSystem,
               im,
               storage_options: dict = None):
    """Save an image object into given path

    Supported im object types:
    - np.ndarray
    - dask.Array
    - cvpl_tools.im.ndblock.NDBlock

    Args:
        file: The full/relative path to the directory to be saved to
        im: Object to be saved
        storage_options: Specifies options in saving method and saved file format
            -preferred_chunksize (tuple[int, ...]): chunk sizes to save as; will rechunk if different from current
                size; only applies to dask arrays.
            -multiscale (int): The number of downsample layers for save ome-zarr; only applies if the image is a dask
                image
            -compressor: The compressor to use to compress array or chunks
    """
    if DEBUG:
        logger.error(f'Saving image to path {file}')
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

    fs = ensure_rdir_filesystem(file)
    if isinstance(im, np.ndarray):
        await NDBlock.save(fs, NDBlock(im), storage_options=storage_options)
    elif isinstance(im, da.Array):
        if old_chunksize != preferred_chunksize:
            im = im.rechunk(preferred_chunksize)
        await NDBlock.save(fs, NDBlock(im), storage_options=storage_options)
    elif isinstance(im, NDBlock):
        if im.get_repr_format() == cvpl_ndblock.ReprFormat.DASK_ARRAY and old_chunksize != preferred_chunksize:
            im = NDBlock(im.get_arr().rechunk(preferred_chunksize))
        await NDBlock.save(fs, im, storage_options=storage_options)
    else:
        raise ValueError(f'Unexpected input type im {type(im)}')

    with fs.open('.save_meta.txt', mode='w') as outfile:
        outfile.write(str(fmt.value))
        outfile.write(f'\n{chunksize_to_str(old_chunksize)}\n{chunksize_to_str(preferred_chunksize)}')

        compressor = storage_options.get('compressor')
        outfile.write(f'\ncompressor:{repr(compressor)}')


def load(file: str | RDirFileSystem, storage_options: dict = None):
    """Load an image from the given directory.

    The image is one saved by cvpl_tools.tools.fs.save()

    Args:
        file: Full path to the directory to be read from
        storage_options: Specifies options in saving method and saved file format
            - compressor (numcodecs.abc.Codec, optional): Compressor used to compress the chunks

    Returns:
        Recreated image; this method attempts to keep meta and content of the loaded image stays
        the same as when they are saved
    """
    logger.error(f'Loading image from path {file}')
    fs = ensure_rdir_filesystem(file)
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


def display(file: str | RDirFileSystem, context_args: dict):
    """Display an image in the viewer; supports numpy or dask ome zarr image

    The image is one saved by cvpl_tools.tools.fs.save()

    Args:
        file: Full path to the directory to be read from
        context_args: contains viewer and arguments passed to the viewer's add image functions
            - viewer (napari.Viewer, optional): Only display if a viewer is provided
            - is_label (bool, optional): defaults to False; if True, use viewer's add_labels() instead of
                add_image() to display the array
            - layer_args (dict, optional)
    """
    import napari
    import cvpl_tools.ome_zarr.napari.add as napari_add_ome_zarr

    context_args = copy.copy(context_args)
    viewer: napari.Viewer = context_args.pop('viewer')
    layer_args = context_args.get('layer_args', {})

    fs = ensure_rdir_filesystem(file)
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

    is_label: bool = context_args.pop('is_label', False)
    if is_numpy:
        fn = viewer.add_labels if is_label else viewer.add_image
        im = NDBlock.load(file).get_arr()
        fn(im, **layer_args)
    else:
        # image saved by NDBlock.save(file)
        napari_add_ome_zarr.subarray_from_path(viewer, f'{file}/dask_im', use_zip=False, merge_channels=True,
                                               kwargs=layer_args | dict(is_label=is_label))


# -------------------------------------File Caching---------------------------------------------

async def cache_im(fn, context_args: dict = None):
    """Caches an image object

    Image format supported: numpy array, dask array, NDBlock

    Args:
        fn: Computes the image if it's not already cached; this may be a function returning an image, or a coroutine or
            coroutine function
        context_args (dict): dictionary of contextual options
            - cache_url (str | RDirFileSystem, optional): Pointing to a directory to store the cached image; if not
                provided, then the image will be cached via dask's persist() and its loaded copy will be returned
            - storage_option (dict, optional): If provided, specifies the compression method to use for image chunks
                - preferred_chunksize (tuple, optional): Re-chunk before save; this rechunking will be undone in load
                - multiscale (int, optional): Specifies the number of downsampling levels on OME ZARR
                - compressor (numcodecs.abc.Codec, optional): Compressor used to compress the chunks
            - viewer_args (dict, optional): If provided, an image will be displayed as a new layer in Napari viewer
                - viewer (napari.Viewer, optional): Only display if a viewer is provided
                - is_label (bool, optional): defaults to False; if True, use viewer's add_labels() instead of
                    add_image() to display the array
            - layer_args (dict, optional): If provided, used along with viewer_args to specify add_image() kwargs

    Returns:
        The cached image loaded from the location just saved
    """
    if context_args is None:
        context_args = {}
    else:
        context_args = copy.copy(context_args)  # since we will pop off some attributes

    preferred_chunksize = context_args.pop('preferred_chunksize', None)
    multiscale = context_args.pop('multiscale', None)
    storage_options = context_args.pop('storage_options', {})
    if preferred_chunksize is not None:
        storage_options['preferred_chunksize'] = preferred_chunksize
    if multiscale is not None:
        storage_options['multiscale'] = multiscale

    fs: None | str | RDirFileSystem = context_args.pop('cache_url', None)
    skip_cache = context_args.get('skip_cache', False)
    if skip_cache or fs is None:
        do_compute = True
    else:
        fs = ensure_rdir_filesystem(fs)
        do_compute = cdir_init(fs).commit is None  # compute result if cache is not found

    if do_compute:
        if inspect.iscoroutine(fn):
            im = await fn
        elif inspect.iscoroutinefunction(fn):
            im = await fn()
        else:
            im = fn()

        # DICT_BLOCK_INDEX_SLICES is a special case when save() is not implemneted for this type of NDBlock
        # In this case, persist it regardless of skip_cache
        if skip_cache or (isinstance(im, NDBlock) and im.get_repr_format() == ReprFormat.DICT_BLOCK_INDEX_SLICES):
            im = persist(im, storage_options=storage_options)
            return im
        else:
            await save(fs, im, storage_options)
            cdir_commit(fs)

    if not skip_cache and context_args.get('viewer') is not None:
        if fs is None or fs.url == '':
            name = 'no_name'
        else:
            name = fs.url
        context_args['layer_args'] = copy.copy(context_args.get('layer_args', {}))
        context_args['layer_args'].setdefault('name', name)
        display(fs, context_args)

    loaded = load(fs, storage_options)

    return loaded


# ------------------------------------------Temporary Storage-----------------------------------------------


class TempDirectory:

    def __init__(self, cache_url: str | RDirFileSystem):
        """Creates a CacheDirectory instance

        Args:
            cache_url: Point to where the cache will be created
        """
        self.cur_idx = 0
        self._fs = ensure_rdir_filesystem(cache_url)
        self._fs.ensure_dir_exists(remove_if_already_exists=True)

    def maketemp(self) -> RDirFileSystem:
        """Return a location that is guaranteed to be empty within the temporary directory

        Returns:
            A RDirFileSystem object pointing to an empty new location
        """
        self.cur_idx += 1
        fs = self._fs[f'{self.cur_idx}']
        return fs

    def remove_tmp(self):
        """traverse all subnodes and self, removing those with is_tmp=True"""
        self._fs.rm('', recursive=True)

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
