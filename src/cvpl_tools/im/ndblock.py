"""
This file defines the NDBlock class and related enum, helper functions related to it
"""
from __future__ import annotations

import os
import abc
import enum
import json
import pickle
import numcodecs
import threading
from typing import Callable, Sequence, Any, Generic, TypeVar
import copy

import partd
from dask.distributed import print as dprint
import numpy as np
import numpy.typing as npt
import dask.array as da
import dask
import functools
import operator
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
from cvpl_tools.im.partd_server import SQLitePartd, SqliteServer


class ReprFormat(enum.Enum):
    """ReprFormat specifies all possible NDBlock formats to use.

    NUMPY_ARRAY = 0
    DASK_ARRAY = 1
    DICT_BLOCK_INDEX_SLICES = 2  # block can be either numpy or dask
    """

    NUMPY_ARRAY = 0
    DASK_ARRAY = 1
    DICT_BLOCK_INDEX_SLICES = 2  # block can be either numpy or dask


ElementType = TypeVar('ElementType')


def dumps_numpy(arr: npt.NDArray, compressor) -> bytes:
    buf = pickle.dumps(arr)
    if compressor is not None:
        buf = compressor.encode(buf)
    return buf


def loads_numpy(buf: bytes, compressor) -> npt.NDArray:
    if compressor is not None:
        buf = compressor.decode(buf)
    return pickle.loads(buf)


class NDBlockLazyLoadDictBlockInfo:
    def __init__(self, db_path: str, compressor=None):
        """Initialize an info object containing information about how to load the NDBlock

        This class is for optimization purpose (lazy loading) where in some cases the computation is not
        necessary

        Args:
            db_path: path to the database object where NDBlock.save saves a DICT_BLOCK_INDEX_SLICES object
            compressor: A compressor that supports encode and decode interfaces on bytes to bytes compression
        """
        self.db_path = db_path
        self.compressor = compressor

    def get_db_path(self):
        return self.db_path

    def is_just_loaded(self):
        """Returns True if just loaded from files

        In some cases we save and then load an NDBlock to avoid double computation, but if the NDBlock
        is just loaded from disk then it is unnecessary to do save and load again. This function is for
        testing for that to avoid such situation
        """
        return True

    def materialize(self, ndblock: NDBlock, force_numpy=False):
        block_indices = ndblock.properties['block_indices']
        slices_list = ndblock.properties['slices_list']
        ndblock.arr = {}

        def load_block_with_compression(sqlite_file, block_id):
            return loads_numpy(sqlite_file.get(block_id, lock=False), self.compressor)

        if force_numpy or ndblock.properties['is_numpy']:
            sqlite_file = SQLitePartd(self.db_path)

            for i in range(len(slices_list)):
                block_index, slices = block_indices[i], slices_list[i]

                block_id = ('_'.join(str(i) for i in block_index)).encode('utf-8')
                block = load_block_with_compression(sqlite_file, block_id)

                ndblock.arr[block_index] = (block, slices)
            sqlite_file.close()
            ndblock.properties['is_numpy'] = True
        else:
            @dask.delayed
            def load_block(block_index):
                block_id = ('_'.join(str(i) for i in block_index)).encode('utf-8')
                sqlite_file = SQLitePartd(self.db_path)
                block = load_block_with_compression(sqlite_file, block_id)
                assert isinstance(block, np.ndarray)
                sqlite_file.close()
                return block

            for i in range(len(slices_list)):
                block_index, slices = block_indices[i], slices_list[i]

                ndblock.arr[block_index] = (load_block(block_index), slices)


class NDBlock(Generic[ElementType], abc.ABC):
    """This class represents an N-dimensional grid, each block in the grid is of arbitrary ndarray shape

    When the grid is of size 1 in all axes, it represents a Numpy array;
    When the grid has blocks of matching size on all axes, it represents a Dask array;
    In the general case the blocks can be of varying sizes e.g. A block of size (2, 2) may be neighboring to a block
    of size (5, 10)

    Currently, we assume block_index is a list always ordered in (0, ..., 0), (0, ..., 1)... (N, ..., M-1) Increasing
    from the side of the tail first
    """

    def __init__(self, arr: npt.NDArray | da.Array | NDBlock | None):
        """Initializes an NDBlock object

        Args:
            arr: If Numpy, create a NDBlock with repr_format=ReprFormat.NUMPY_ARRAY; if Dask, create a NDBlock
                with repr_format=ReprFormat.DASK_ARRAY; if arr is an NDBlock, then a copy of arr will be created
        """
        self.arr = None
        self.properties = None

        if arr is None:
            return  # don't fully initialize; used in case properties can't be inferred from arr

        if isinstance(arr, np.ndarray):
            assert arr.ndim >= 1, f'NDBlock does not save 0-dimension array, ensure array is at least 1-d'
            self.arr = arr
            self._set_properties_by_numpy_array(arr)
        elif isinstance(arr, da.Array):
            self.arr = arr
            self._set_properties_by_dask_array(arr)
        else:
            assert isinstance(arr, NDBlock), f'Unexpected type {type(arr)}'
            self.arr = arr.arr
            self.properties = NDBlock._copy_properties(arr.properties)

    @staticmethod
    def create_from_dict_and_properties(d: dict, properties: dict[str, Any]):
        ndblock = NDBlock(None)
        ndblock.arr = d
        ndblock.properties = NDBlock._copy_properties(properties)
        is_numpy = False
        for block_index, (block, slices) in d.items():
            is_numpy = isinstance(block, np.ndarray)
            break
        assert properties['is_numpy'] == is_numpy, (f"Got is_numpy={properties['is_numpy']} in properties but "
                                                    f"is_numpy={is_numpy} for dictionary of blocks")
        return ndblock

    @staticmethod
    def properties_consistency_check(properties: dict[str, Any]):
        block_indices = properties['block_indices']
        assert isinstance(block_indices, list)
        assert isinstance(block_indices[0], tuple)

        slices_list = properties['slices_list']
        assert isinstance(slices_list, list)
        assert isinstance(slices_list[0], tuple)
        assert isinstance(slices_list[0][0], slice)

        repr_format = properties['repr_format']
        assert isinstance(repr_format, ReprFormat)

        numblocks = properties['numblocks']
        assert isinstance(numblocks, tuple)
        assert isinstance(numblocks[0], int)

    def get_arr(self):
        self.ensure_materialized()
        return self.arr

    def ensure_materialized(self, **kwargs):
        if isinstance(self.arr, NDBlockLazyLoadDictBlockInfo):
            self.arr.materialize(self, **kwargs)

    @staticmethod
    def save_properties(file: str, properties: dict):
        with open(file, mode='w') as outfile:
            # convert repr_format and dtype to appropriate format for serialization
            # reference: https://stackoverflow.com/questions/47641404/serializing-numpy-dtype-objects-human-readable
            properties = copy.copy(properties)
            properties['repr_format'] = properties['repr_format'].value
            properties['dtype'] = np.dtype(properties['dtype']).str
            properties['slices_list'] = [tuple((s.start, s.stop) for s in si) for si in properties['slices_list']]

            block_indices = properties['block_indices']
            assert isinstance(properties['block_indices'], list), (f'Expected block indices to be of type list, '
                                                                   f'got {block_indices}')
            json.dump(properties, fp=outfile, indent=2)

    @staticmethod
    def save(file: str, ndblock: NDBlock, storage_options: dict | None = None):
        """Save the NDBlock to the given path

        Will compute immediately if the ndblock is delayed dask computations

        Storage Opitons
            multiscale (int) = 0
                This only applies if ndblock is dask, will write multilevel if non-zero
            compressor = None
                The compressor to use to compress array or chunks

        Args:
            file: The file to save to
            ndblock: The block which will be saved
            storage_options: Specifies options in saving method and saved file format
        """
        if storage_options is None:
            storage_options = {}
        compressor = storage_options.get('compressor', None)

        ndblock.ensure_materialized()

        fmt = ndblock.get_repr_format()
        os.makedirs(file, exist_ok=False)
        if fmt == ReprFormat.NUMPY_ARRAY:
            if compressor is None:
                np.save(f'{file}/im.npy', ndblock.arr)
            else:
                with open(f'{file}/im_compressed.bin', 'wb') as binfile:
                    binfile.write(dumps_numpy(ndblock.arr, compressor=compressor))
        elif fmt == ReprFormat.DASK_ARRAY:
            MAX_LAYER = storage_options.get('multiscale') or 0
            cvpl_ome_zarr_io.write_ome_zarr_image(f'{file}/dask_im', da_arr=ndblock.arr,
                                                  make_zip=False, MAX_LAYER=MAX_LAYER,
                                                  storage_options=storage_options)
        else:
            server = SqliteServer(f'{file}/blocks_kvstore', nappend=len(ndblock.arr), available_memory=1e6,
                                  port_protocol=storage_options.get('port_protocol', 'tcp'))
            server_address = server.address

            @dask.delayed
            def save_block(block_index, block):
                block_id = ('_'.join(str(i) for i in block_index)).encode('utf-8')
                store = partd.Client(server_address)
                assert isinstance(block, np.ndarray), type(block)
                store.append({
                    block_id: dumps_numpy(block, compressor=compressor)
                })
                store.close()

            tasks = [save_block(block_index, block) for block_index, (block, _) in ndblock.arr.items()]
            dask.compute(*tasks)
            server.wait_join()
        NDBlock.save_properties(f'{file}/properties.json', ndblock.properties)

    @staticmethod
    def load_properties(file: str) -> dict:
        with open(file, mode='r') as infile:
            properties = json.load(fp=infile)

            properties['repr_format'] = ReprFormat(properties['repr_format'])
            properties['dtype'] = np.dtype(properties['dtype'])
            properties['block_indices'] = [tuple(idx) for idx in properties['block_indices']]
            properties['slices_list'] = [tuple(slice(s[0], s[1]) for s in si) for si in properties['slices_list']]
            properties['numblocks'] = tuple(properties['numblocks'])
        NDBlock.properties_consistency_check(properties)
        return properties

    @staticmethod
    def load(file: str, storage_options: dict | None = None) -> NDBlock:
        """Load the NDBlock from the given path.

        Storage Opitons
            compressor = None
                The compressor to use to compress array or chunks

        Args:
            file: The path to load from, same as used in the save() function
            storage_options: Specifies options in saving method and saved file format

        Returns:
            The loaded NDBlock. Guaranteed to have the same properties as the saved one, and the content of each
            block will be the same as when they are saved.
        """
        if storage_options is None:
            storage_options = {}
        compressor = storage_options.get('compressor', None)

        properties = NDBlock.load_properties(f'{file}/properties.json')

        fmt = properties['repr_format']
        ndblock = NDBlock(None)
        ndblock.properties = properties
        if fmt == ReprFormat.NUMPY_ARRAY:
            if compressor is None:
                ndblock.arr = np.load(f'{file}/im.npy')
            else:
                with open(f'{file}/im_compressed.bin', 'rb') as binfile:
                    ndblock.arr = loads_numpy(binfile.read(), compressor)
        elif fmt == ReprFormat.DASK_ARRAY:
            ndblock.arr = da.from_zarr(cvpl_ome_zarr_io.load_zarr_group_from_path(
                f'{file}/dask_im',
                mode='r',
                level=0
            ))
        else:
            ndblock.arr = NDBlockLazyLoadDictBlockInfo(db_path=f'{file}/blocks_kvstore', compressor=compressor)
        return ndblock

    def is_numpy(self) -> bool:
        """Returns True if this is Numpy array

        Note besides type ReprFormat.NUMPY, ReprFormat.DICT_BLOCK_INDEX_SLICES may have either Numpy arrays as each
        block, or Dask delayed objects each returning a Numpy array; in the former case is_numpy() will return True,
        and in the latter case is_numpy() will return False.

        Returns:
            Returns True if this is Numpy array
        """
        return self.properties['is_numpy']

    def get_properties(self) -> dict[str, Any]:
        return self.properties

    def get_repr_format(self) -> ReprFormat:
        return self.properties['repr_format']

    @property
    def ndim(self) -> int:
        return self.properties['ndim']

    def get_ndim(self) -> int:
        return self.properties['ndim']

    def get_numblocks(self) -> tuple[int]:
        return self.properties['numblocks']

    def get_block_indices(self) -> list:
        return self.properties['block_indices']

    def get_slices_list(self) -> list:
        return self.properties['slices_list']

    def get_dtype(self):
        return self.properties['dtype']

    def get_chunksize(self) -> tuple[int, ...]:
        """Get a single tuple of chunk size on each axis

        Returns:
            A tuple of int of length ndim
        """
        slices = self.properties['slices_list'][0]
        chunksize = tuple(s.stop - s.start for s in slices)
        return chunksize

    def as_numpy(self) -> npt.NDArray:
        other = NDBlock(self)
        other.to_numpy_array()
        return other.arr

    def as_dask_array(self, tmp_dirpath: str | None = None) -> da.Array:
        """Get a copy of the array value as dask array

        Args:
            tmp_dirpath: Avoid double computation, best provided if input format
                is ReprFormat.DICT_BLOCK_INDEX_SLICES and is dask instead of numpy
        Returns:
            converted/retrieved dask array
        """
        other = NDBlock(self)
        other.to_dask_array(tmp_dirpath)
        return other.arr

    @staticmethod
    def _copy_properties(properties: dict):
        NDBlock.properties_consistency_check(properties)
        return dict(
            repr_format=properties['repr_format'],
            ndim=properties['ndim'],
            numblocks=properties['numblocks'],
            block_indices=properties['block_indices'],
            slices_list=[item for item in properties['slices_list']],
            is_numpy=properties['is_numpy'],
            dtype=properties['dtype']
        )

    def _set_properties_by_numpy_array(self, arr: npt.NDArray):
        ndim: int = arr.ndim
        self.properties = dict(
            repr_format=ReprFormat.NUMPY_ARRAY,
            ndim=ndim,
            numblocks=(1,) * ndim,
            block_indices=[(0,) * ndim],
            slices_list=[tuple(slice(0, s) for s in arr.shape)],
            is_numpy=True,
            dtype=arr.dtype
        )
        NDBlock.properties_consistency_check(self.properties)

    def _set_properties_by_dask_array(self, arr: da.Array):
        ndim: int = arr.ndim
        numblocks = arr.numblocks
        block_indices = list(np.ndindex(*arr.numblocks))
        slices_list: list = da.core.slices_from_chunks(arr.chunks)

        self.properties = dict(
            repr_format=ReprFormat.DASK_ARRAY,
            ndim=ndim,
            numblocks=numblocks,
            block_indices=block_indices,
            slices_list=slices_list,
            is_numpy=False,
            dtype=arr.dtype
        )
        NDBlock.properties_consistency_check(self.properties)

    # ReprFormat conversion functions

    def to_numpy_array(self):
        """Convert representation format to numpy array"""
        rformat = self.properties['repr_format']
        if rformat == ReprFormat.NUMPY_ARRAY:
            return

        self.ensure_materialized()

        if rformat == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            # TODO: optimize this
            self.to_dask_array()
            rformat = self.get_repr_format()

        assert rformat == ReprFormat.DASK_ARRAY
        self.arr = self.arr.compute()
        self._set_properties_by_numpy_array(self.arr)  # here some blocks may be merged to one

        self.properties['repr_format'] = ReprFormat.NUMPY_ARRAY
        self.properties['is_numpy'] = True

    def to_dask_array(self, tmp_dirpath: str | None = None):
        """Convert representation format to dask array

        Args:
            tmp_dirpath: An empty path to create a temporary save file, best provided if input format
                is ReprFormat.DICT_BLOCK_INDEX_SLICES and is dask instead of numpy, and you want to
                avoid repeated computations
        """

        rformat = self.properties['repr_format']
        if rformat == ReprFormat.DASK_ARRAY:
            return

        just_loaded_flag = isinstance(self.arr, NDBlockLazyLoadDictBlockInfo) and self.arr.is_just_loaded()

        if rformat == ReprFormat.NUMPY_ARRAY:
            self.ensure_materialized()
            self.arr = da.from_array(self.arr)
        else:
            assert rformat == ReprFormat.DICT_BLOCK_INDEX_SLICES
            numblocks = self.get_numblocks()
            dtype = self.get_dtype()

            if not self.properties['is_numpy'] and tmp_dirpath is not None and not just_loaded_flag:
                """Explanation of the branching statement: this if branch is optional and purely for 
                optimization purpose i.e. some dask computation may be triggered twice unexpectedly and 
                this avoids it
                
                not self.properties['is_numpy']
                - If numpy, the array is already computed and no double computation possible
                
                tmp_dirpath is not None
                - If tmp_dirpath is None, then we don't have a place to put our temporary save files
                
                not just_loaded_flag
                - Tests if the array if just loaded from disk; if it is, then no double computation possible since 
                    there are no intermediate steps in the first place
                """
                if not os.path.exists(tmp_dirpath):
                    NDBlock.save(tmp_dirpath, self)
                ndblock_to_be_combined = NDBlock.load(tmp_dirpath)
            else:
                ndblock_to_be_combined = self
            ndblock_to_be_combined.ensure_materialized()

            # reference: https://github.com/dask/dask-image/blob/adcb217de766dd6fef99895ed1a33bf78a97d14b/dask_image/ndmeasure/__init__.py#L299
            ndlists = np.empty(numblocks, dtype=object)
            for block_index, (block, slices) in ndblock_to_be_combined.arr.items():
                block_shape = tuple(s.stop - s.start for s in slices)
                if not isinstance(block, np.ndarray):
                    block = da.from_delayed(block, shape=block_shape, dtype=dtype)
                ndlists[block_index] = block
            ndlists = ndlists.tolist()
            self.arr = da.block(ndlists)
        self.properties['repr_format'] = ReprFormat.DASK_ARRAY
        self.properties['is_numpy'] = False

    def to_dict_block_index_slices(self):
        """Convert representation format to ReprFormat.DICT_BLOCK_INDEX_SLICES"""
        rformat = self.properties['repr_format']
        if rformat == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            return

        self.ensure_materialized()

        slices_list = self.properties['slices_list']
        block_indices = self.properties['block_indices']
        arr = self.arr
        if rformat == ReprFormat.NUMPY_ARRAY:
            blocks = [arr]
        else:
            assert isinstance(arr, da.Array)
            blocks = list(map(functools.partial(operator.getitem, arr), slices_list))
        nblocks = len(blocks)
        assert len(slices_list) == nblocks and len(block_indices) == nblocks, \
            (f'Expected equal length of block, indices and slices, got len(blocks)={nblocks} '
             f'len(block_indices)={len(block_indices)} len(slices_list)={len(slices_list)}')

        self.arr = {
            block_indices[i]: (blocks[i], slices_list[i]) for i in range(len(blocks))
        }

        self.properties['repr_format'] = ReprFormat.DICT_BLOCK_INDEX_SLICES

    def reduce(self, force_numpy: bool = False) -> npt.NDArray | da.Array:
        """Concatenate all blocks on the first axis

        Args:
            force_numpy: If True, the result will be forced from dask to a numpy array, if not already; useful for
                outputting to analysis that requires numpy input

        Returns:
            The concatenated result, is Numpy if previous array is Numpy, or if force_numpy is True
        """
        other = NDBlock(self)
        other.ensure_materialized(force_numpy=force_numpy)

        if other.properties['repr_format'] == ReprFormat.NUMPY_ARRAY:
            return np.copy(other.arr)
        else:
            other.to_dict_block_index_slices()

            if other.is_numpy():
                blocks = [block for _, (block, _) in other.arr.items()]
                assert len(blocks) > 0, 'Need at least one row for NDBlock to be reduced'
                assert isinstance(blocks[0], np.ndarray), f'Expected ndarray, got type(blocks[0])={type(blocks[0])}'
                return np.concatenate(blocks, axis=0)
            else:
                shape = None
                for block_index, (block, slices) in other.arr.items():
                    if not isinstance(block, np.ndarray):
                        block = block.compute()
                    shape = (np.nan,) + block.shape[1:]
                    break
                blocks = [da.from_delayed(block,
                                          shape=shape,
                                          dtype=other.get_dtype())
                          for block_index, (block, slices) in other.arr.items()]
                assert len(blocks) > 0, 'Need at least one row for NDBlock to be reduced'

                reduced = da.concatenate(blocks, axis=0)
                if force_numpy:
                    return reduced.compute()
                else:
                    return reduced

    def sum(self, axis: Sequence | None = None, keepdims: bool = False):
        """sum over axes for each block"""
        new_ndblock = NDBlock(self)
        new_ndblock.ensure_materialized()
        if self.properties['repr_format'] == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            for block_index, (block, slices) in self.arr.items():
                new_ndblock.arr[block_index] = (block.sum(axis=axis, keepdims=keepdims), slices)
            return new_ndblock
        else:
            new_ndblock.arr = self.arr.sum(axis=axis)

    @staticmethod
    def map_ndblocks(
            inputs: Sequence[NDBlock],
            fn: Callable,
            out_dtype: np.dtype,
            use_input_index_as_arrloc: int = 0,
            new_slices: list = None,
            fn_args: dict = None
    ) -> NDBlock:
        """Similar to da.map_blocks, but works with NDBlock.

        Different from dask array's map_blocks: For each input i, the block_info[i] provided to the mapping function
        will contain only two keys, 'chunk-location' and 'array-location'.

        Args:
            inputs: A list of inputs to be mapped, either all dask or all numpy; All inputs must have the same number
                of blocks, block indices, and over the same slices as well if inputs are dask images
            fn: fn(\*block, block_info) maps input blocks to an output block
            out_dtype: Output block type (provide a Numpy dtype to this)
            use_input_index_as_arrloc: output slices_list will be the same as this input (ignore this for variable
                sized blocks)
            new_slices: will be used to replace the slices attribute if specified
            fn_args: extra arguments to be passed to the mapping function

        Returns:
            Result mapped, of format ReprFormat.DICT_BLOCK_INDEX_SLICES
        """
        if fn_args is None:
            fn_args = dict()

        inputs = list(inputs)
        assert len(inputs) > 0, 'Must have at least one input!'

        is_numpy = inputs[0].get_repr_format() == ReprFormat.NUMPY_ARRAY
        for i in range(1, len(inputs)):
            assert is_numpy == (inputs[i].get_repr_format() == ReprFormat.NUMPY_ARRAY), \
                ('All inputs must either be all dask or all Numpy, expected '
                 f'is_numpy={is_numpy} but found wrong typed input at '
                 f'location {i}')

        if is_numpy:
            block_info = []
            for i in range(len(inputs)):
                ndblock = inputs[i]
                arr = ndblock.as_numpy()
                block_info.append({
                    'chunk-location': (0,) * arr.ndim,
                    'array-location': list((0, s) for s in arr.shape)
                })
            return NDBlock(fn(*inputs, block_info, **fn_args))

        # we can assume the inputs are all dask, now turn them all into block iterator
        block_iterators = []
        for ndblock in inputs:
            new_block = NDBlock(ndblock)
            new_block.ensure_materialized()
            if ndblock.get_repr_format() != ReprFormat.DICT_BLOCK_INDEX_SLICES:
                new_block.to_dict_block_index_slices()
            block_iterators.append(new_block)

        # write in this form mostly for debugging convenience, equivalently:
        # delayed_fn = dask.delayed(fn)
        @dask.delayed
        def delayed_fn(*blocks, block_info, **kwargs):
            return fn(*blocks, block_info, **kwargs)

        result = {}
        for block_index in block_iterators[use_input_index_as_arrloc].arr.keys():
            blocks = []
            block_info = []
            for i in range(len(block_iterators)):
                block, slices = block_iterators[i].arr[block_index]
                blocks.append(block)
                block_info.append({
                    'chunk-location': block_index,
                    'array-location': slices,
                })
            result[block_index] = (delayed_fn(*blocks, block_info=block_info, **fn_args),
                                   block_info[use_input_index_as_arrloc]['array-location'])

        ndim = block_iterators[0].get_ndim()
        numblocks = block_iterators[0].get_numblocks()
        result_ndblock = NDBlock(None)
        result_ndblock.arr = result
        if new_slices is None:
            slices_list = block_iterators[use_input_index_as_arrloc].properties['slices_list']
        else:
            slices_list = new_slices
        result_ndblock.properties = dict(
            repr_format=ReprFormat.DICT_BLOCK_INDEX_SLICES,
            ndim=ndim,
            numblocks=numblocks,
            block_indices=[k for k in block_iterators[use_input_index_as_arrloc].arr.keys()],
            slices_list=slices_list,
            is_numpy=False,
            dtype=out_dtype
        )
        NDBlock.properties_consistency_check(result_ndblock.properties)
        return result_ndblock

    def select_columns(self, cols: slice | Sequence[int] | int) -> NDBlock:
        """Performs columns selection on a 2d array"""
        ndblock = NDBlock(self)
        ndblock.ensure_materialized()
        if self.properties['repr_format'] == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            results = {
                block_index: (block[:, cols], slices) for block_index, (block, slices) in ndblock.arr.items()
            }
            ndblock.arr = results
        else:
            ndblock.arr = ndblock.arr[:, cols]
        return ndblock

    def run_delayed_discard_results(self):
        assert (self.properties['repr_format'] == ReprFormat.DICT_BLOCK_INDEX_SLICES and
                not self.properties['is_numpy'])
        for value in self.arr.values():
            value[0]()
