"""
This file defines some helper functions and algorithms for parallel dask image processing
"""
from __future__ import annotations

import os
import abc
import copy
import enum
from typing import Callable, Iterator, Iterable, Sequence, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
import dask.array as da
import dask
import functools
import operator


def _init_ndlist(idx: tuple, shape: tuple, init_element: Callable[[tuple], Any]) -> list:
    if len(shape) > 1:
        return [_init_ndlist(idx + (i,), shape[1:], init_element) for i in range(shape[0])]

    return [init_element(idx + (i,)) for i in range(shape[0])]


def _dask_ndconcat(ndlists: Sequence, ax_orders: Sequence[int] | None = None):
    if len(ax_orders) > 1:
        sub_ax_orders = ax_orders[1:]
        ndlists = [_dask_ndconcat(ndlists[i], sub_ax_orders) for i in range(len(ndlists))]
    # TODO: This does not parallelize well, how can this be delayed?
    return da.concatenate(da.compute(*ndlists), axis=ax_orders[0])


def dask_ndconcat(ndlists: Sequence, ndim: int | None = None, ax_orders: Sequence[int] | None = None):
    """Concatenate a ndlists of dask delayed into a single dask image

    This function will correctly handle both when blocks are numpy ndarray or when they are dask arrays
    """
    assert not (ndim is None and ax_orders is None), ('Expected to receive at least one of ndim or ax_order as input, '
                                                      f'got ndim={ndim} and ax_orders={ax_orders}')
    if ndim is not None and ax_orders is not None:
        assert len(ax_orders) == ndim, f'Conflicting inputs, got ndim={ndim} and ax_orders={ax_orders}'

    if ax_orders is None:
        ax_orders = tuple(range(ndim))
    return _dask_ndconcat(ndlists, ax_orders)


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
            assert isinstance(arr, NDBlock)
            self.arr = arr.arr
            self.properties = NDBlock._copy_properties(arr.properties)

    @staticmethod
    def from_ndlists(ndlists: Sequence):
        return NDBlock(dask_ndconcat(ndlists))

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
            block_indices=(0,) * ndim,
            slices_list=list((0, s) for s in arr.shape),
            is_numpy=True,
            dtype=arr.dtype
        )

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
                os.makedirs(tmp_dirpath, exist_ok=True)

                @dask.delayed
                def save_block(block_index, block):
                    suffix = '_'.join(str(i) for i in block_index) + '.npy'
                    np.save(f'{tmp_dirpath}/{suffix}', block)

                @dask.delayed
                def load_block(block_index):
                    suffix = '_'.join(str(i) for i in block_index) + '.npy'
                    return np.load(f'{tmp_dirpath}/{suffix}')

                tasks = [save_block(block_index, block) for block_index, (block, _) in self.arr.items()]
                dask.compute(*tasks)
                self.arr = {
                    block_index: (load_block(block_index), slices) for block_index, (_, slices) in self.arr.items()
                }

                # self.properties['is_numpy'] is set at the end of this function

            # reference: https://github.com/dask/dask-image/blob/adcb217de766dd6fef99895ed1a33bf78a97d14b/dask_image/ndmeasure/__init__.py#L299
            ndlists = np.empty(numblocks, dtype=object)
            for block_index in self.arr.keys():
                block, slices = self.arr[block_index]
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
            block_indices=block_iterators[use_input_index_as_arrloc].arr.keys(),
            slices_list=block_iterators[use_input_index_as_arrloc].properties['slices_list'],
            is_numpy=False,
            dtype=out_dtype
        )
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

