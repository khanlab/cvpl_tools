"""
This file defines the NDBlock class and related enum, helper functions related to it
"""
from __future__ import annotations

import os
import abc
import enum
import json
from typing import Callable, Sequence, Any, Generic, TypeVar
import copy

import numpy as np
import numpy.typing as npt
import dask.array as da
import dask
import functools
import operator
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io


class ReprFormat(enum.Enum):
    NUMPY_ARRAY = 0
    DASK_ARRAY = 1
    DICT_BLOCK_INDEX_SLICES = 2  # block can be either numpy or dask


ElementType = TypeVar('ElementType')


class NDBlock(Generic[ElementType], abc.ABC):
    """This class represents a nd arrangement of arbitrarily sized blocks

    Currently, we assume block_index is a list always ordered in (0, ..., 0), (0, ..., 1)... (N, ..., M-1) Increasing
    from the side of the tail first
    """

    def __init__(self, arr: npt.NDArray | da.Array | NDBlock | None):
        self.arr = None
        self.properties = None

        if arr is None:
            return  # don't fully initialize; used in case properties can't be inferred from arr

        if isinstance(arr, np.ndarray):
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

    @staticmethod
    def save_properties(file: str, properties: dict):
        with open(file, mode='w') as outfile:
            # convert repr_format and dtype to appropriate format for serialization
            # reference: https://stackoverflow.com/questions/47641404/serializing-numpy-dtype-objects-human-readable
            properties = copy.copy(properties)
            properties['repr_format'] = properties['repr_format'].value
            dt = np.dtype(properties['dtype'])
            properties['dtype'] = dt.descr
            properties['slices_list'] = [tuple((s.start, s.stop) for s in si) for si in properties['slices_list']]

            block_indices = properties['block_indices']
            assert isinstance(properties['block_indices'], list), (f'Expected block indices to be of type list, '
                                                                   f'got {block_indices}')
            json.dump(properties, fp=outfile, indent=2)

    @staticmethod
    def save(file: str, ndblock: NDBlock, downsample_level: int = 0):
        """Save the NDBlock to the given path

        Will compute immediately if the ndblock is delayed dask computations

        Args:
            file: The file to save to
            ndblock: The block which will be saved
            downsample_level: This only applies if ndblock is dask, will write multilevel if non-zero
        """
        fmt = ndblock.get_repr_format()
        os.makedirs(file, exist_ok=False)
        if fmt == ReprFormat.NUMPY_ARRAY:
            np.save(f'{file}/im.npy', ndblock.arr)
        elif fmt == ReprFormat.DASK_ARRAY:
            cvpl_ome_zarr_io.write_ome_zarr_image(f'{file}/dask_im', da_arr=ndblock.arr,
                                                  make_zip=False, MAX_LAYER=downsample_level)
        else:
            os.makedirs(f'{file}/blocks', exist_ok=False)
            @dask.delayed
            def save_block(block_index, block):
                suffix = '_'.join(str(i) for i in block_index) + '.npy'
                np.save(f'{file}/blocks/{suffix}', block)

            tasks = [save_block(block_index, block) for block_index, (block, _) in ndblock.arr.items()]
            dask.compute(*tasks)
        NDBlock.save_properties(f'{file}/properties.json', ndblock.properties)

    @staticmethod
    def load_properties(file: str) -> dict:
        with open(file, mode='r') as infile:
            properties = json.load(fp=infile)

            properties['repr_format'] = ReprFormat(properties['repr_format'])
            dt = properties['dtype']
            properties['dtype'] = np.dtype([tuple(i) for i in dt])
            properties['block_indices'] = [tuple(idx) for idx in properties['block_indices']]
            properties['slices_list'] = [tuple(slice(s[0], s[1]) for s in si) for si in properties['slices_list']]
            properties['numblocks'] = tuple(properties['numblocks'])
        NDBlock.properties_consistency_check(properties)
        return properties

    @staticmethod
    def load(file: str):
        properties = NDBlock.load_properties(f'{file}/properties.json')

        fmt = properties['repr_format']
        ndblock = NDBlock(None)
        ndblock.properties = properties
        if fmt == ReprFormat.NUMPY_ARRAY:
            ndblock.arr = np.load(f'{file}/im.npy')
        elif fmt == ReprFormat.DASK_ARRAY:
            ndblock.arr = da.from_zarr(cvpl_ome_zarr_io.load_zarr_group_from_path(
                f'{file}/dask_im',
                mode='r',
                level=0
            ))
        else:
            @dask.delayed
            def load_block(block_index):
                suffix = '_'.join(str(i) for i in block_index) + '.npy'
                return np.load(f'{file}/blocks/{suffix}')

            ndblock.arr = {}
            block_indices = properties['block_indices']
            slices_list = properties['slices_list']
            for i in range(len(slices_list)):
                block_index, slices = block_indices[i], slices_list[i]

                ndblock.arr[block_index] = (load_block(block_index), slices)
        return ndblock

    def is_numpy(self) -> bool:
        return self.properties['is_numpy']

    def get_repr_format(self) -> ReprFormat:
        return self.properties['repr_format']

    def get_ndim(self) -> int:
        return self.properties['ndim']

    def get_numblocks(self) -> tuple[int]:
        return self.properties['numblocks']

    def get_block_indices(self) -> list:
        return self.properties['block_indices']

    def get_dtype(self):
        return self.properties['dtype']

    def get_chunksize(self) -> tuple[int, ...]:
        slices = self.properties['slices_list'][0]
        chunksize = tuple(s.stop - s.start + 1 for s in slices)
        return chunksize

    def as_numpy(self) -> da.Array:
        if self.properties['repr_format'] == ReprFormat.NUMPY_ARRAY:
            return self.arr
        else:
            other = NDBlock(self)
            other.to_numpy_array()
            return other.arr

    def as_dask_array(self, tmp_dirpath: str | None = None) -> da.Array:
        if self.properties['repr_format'] == ReprFormat.DASK_ARRAY:
            return self.arr
        else:
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
        if self.properties['repr_format'] == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            # TODO: optimize this
            self.to_dask_array()

        rformat = self.properties['repr_format']
        if rformat == ReprFormat.NUMPY_ARRAY:
            return

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

        if rformat == ReprFormat.NUMPY_ARRAY:
            self.arr = da.from_array(self.arr)
        else:
            assert rformat == ReprFormat.DICT_BLOCK_INDEX_SLICES
            numblocks = self.get_numblocks()
            dtype = self.get_dtype()

            if not self.properties['is_numpy'] and tmp_dirpath is not None:
                if not os.path.exists(tmp_dirpath):
                    NDBlock.save(tmp_dirpath, self)
                ndblock_to_be_combined = NDBlock.load(tmp_dirpath)
            else:
                ndblock_to_be_combined = self

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
        rformat = self.properties['repr_format']
        if rformat == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            return

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
        """concatenate all blocks on the first axis"""
        if self.properties['repr_format'] == ReprFormat.NUMPY_ARRAY:
            return np.copy(self.arr)
        else:
            other = NDBlock(self)
            other.to_dict_block_index_slices()

            shape = None
            for block_index, (block, slices) in other.arr.items():
                if not isinstance(block, np.ndarray):
                    block = block.compute()
                shape = (np.nan,) + block.shape[1:]
                break
            blocks = [da.from_delayed(block,
                                      shape=shape,
                                      dtype=self.get_dtype())
                      for block_index, (block, slices) in other.arr.items()]
            assert len(blocks) > 0, 'Need at least one row for NDBlock to be reduced'
            if other.is_numpy():
                return np.concatenate(blocks, axis=0)
            else:
                reduced = da.concatenate(blocks, axis=0)
                if force_numpy:
                    return reduced.compute()
                else:
                    return reduced

    def sum(self, axis: Sequence | None = None, keepdims: bool = False):
        """sum over axes for each block"""
        new_ndblock = NDBlock(self)
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
        use_input_index_as_arrloc: int = 0
    ) -> NDBlock:
        """similar to da.map_blocks but works with NDBlock

        Args:
            inputs: A list of inputs to be mapped, either all dask or all numpy; All inputs must have the same number
                of blocks, block indices, and over the same slices as well if inputs are dask images
            fn: fn(*block, block_info) maps input blocks to an output block
            use_input_index_as_arrloc: output slices_list will be the same as this input (ignore this for variable
                sized blocks)

        Returns:
            Result mapped, of format ReprFormat.DICT_BLOCK_INDEX_SLICES
        """
        inputs = list(inputs)
        assert len(inputs) > 0, 'Must have at least one input!'

        is_numpy = inputs[0].is_numpy()
        for i in range(1, len(inputs)):
            assert is_numpy == inputs[i].is_numpy(), ('All inputs must either be all dask or all Numpy, expected '
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
            return NDBlock(fn(*inputs, block_info))

        # we can assume the inputs are all dask, now turn them all into block iterator
        block_iterators = []
        for ndblock in inputs:
            if ndblock.get_repr_format() != ReprFormat.DICT_BLOCK_INDEX_SLICES:
                new_block = NDBlock(ndblock)
                new_block.to_dict_block_index_slices()
                block_iterators.append(new_block)
            else:
                block_iterators.append(ndblock)

        @dask.delayed
        def delayed_fn(*blocks, block_info):
            return fn(*blocks, block_info)

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
            result[block_index] = (delayed_fn(*blocks, block_info=block_info),
                                   block_info[use_input_index_as_arrloc]['array-location'])

        ndim = block_iterators[0].get_ndim()
        numblocks = block_iterators[0].get_numblocks()
        result_ndblock = NDBlock(None)
        result_ndblock.arr = result
        result_ndblock.properties = dict(
            repr_format=ReprFormat.DICT_BLOCK_INDEX_SLICES,
            ndim=ndim,
            numblocks=numblocks,
            block_indices=[k for k in block_iterators[use_input_index_as_arrloc].arr.keys()],
            slices_list=block_iterators[use_input_index_as_arrloc].properties['slices_list'],
            is_numpy=False,
            dtype=out_dtype
        )
        NDBlock.properties_consistency_check(result_ndblock.properties)
        return result_ndblock

    def select_columns(self, cols: slice | Sequence[int] | int) -> NDBlock:
        """Performs columns selection on a 2d array"""
        if self.properties['repr_format'] == ReprFormat.DICT_BLOCK_INDEX_SLICES:
            results = {
                block_index: (block[:, cols], slices) for block_index, (block, slices) in self.arr.items()
            }
            ndblock = NDBlock(self)
            ndblock.arr = results
        else:
            ndblock = NDBlock(self)
            ndblock.arr = ndblock.arr[:, cols]
        return ndblock
