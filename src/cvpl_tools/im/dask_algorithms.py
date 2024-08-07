"""
This file defines some helper functions and algorithms for parallel dask image processing
"""


from typing import Callable, Iterator

import numpy as np
import dask.array as da
import dask
import functools
import operator


def map_da_to_rows(im: da.Array | np.ndarray,
                   fn: Callable,
                   return_dim: int,
                   return_dtype: np.dtype,
                   return_dask: bool = False,
                   reduce: bool = True) -> Iterator[tuple] | da.Array | np.ndarray:
    """Map each block to variable number of rows and optionally concatenate the results from all blocks

    If reduce is false, the results is an iterator of (mapped_array, block_index, block_slices)

    Args:
        im: The dask image to be mapped
        fn: takes (numpy_block, block_idx, slices) as input and output an N * D Numpy array where the first dimension
            N is of variable length for different blocks
        return_dim: the value of D, the shape of numpy array returned by fn
        return_dtype: the dtype of returned array
        return_dask: return a dask array instead of numpy
        reduce: if True, concatenate the results from all blocks

    Returns:
        (optionally concatenated) array of results from all blocks, mapped by fn
    """
    # reference: https://github.com/dask/dask/issues/7589 and
    # https://github.com/dask/dask-image/blob/adcb217de766dd6fef99895ed1a33bf78a97d14b/dask_image/ndmeasure/__init__.py#L299

    if isinstance(im, np.ndarray):
        # special case where this just reduces to a single function call
        return fn(im, (0, 0, 0), tuple(slice(0, s) for s in im.shape))

    # note the block da.Array passed as first argument to delayed_fn() will become np.ndarray in the call to fn()
    # this is due to how dask.delayed() handles input arguments
    @dask.delayed
    def delayed_fn(block, block_idx, slices):
        result = fn(block, block_idx, slices)
        return result

    slices_list = dask.array.core.slices_from_chunks(im.chunks)
    block_iter = zip(
        map(functools.partial(operator.getitem, im), slices_list),  # dask block
        np.ndindex(*im.numblocks),  # the block index
        slices_list  # the slices (index of pixels)
    )
    results = [da.from_delayed(delayed_fn(*it),
                               shape=(np.nan, return_dim),
                               dtype=return_dtype,
                               meta=np.zeros(shape=tuple(), dtype=return_dtype)) for it in block_iter]

    if not return_dask:
        results = dask.compute(*results)
    if reduce:
        if return_dask:
            raise NotImplementedError('to be implemented')
        else:
            return np.concatenate(results, axis=0)
    else:
        # add block index and slice information
        return zip(
            results,
            np.ndindex(*im.numblocks),
            slices_list
        )


def map_rows_to_rows(rows: Iterator[tuple], block_map: Callable, reduce: bool = False):
    """Input rows ('s elements) should be of type np.ndarray"""
    rows = [(block_map(r), block_index, slices) for r, block_index, slices in rows]
    if reduce:
        rows = np.concatenate([r[0] for r in rows], axis=0)
    return rows


def reduce_numpy_rows(rows: Iterator[tuple]) -> np.ndarray:
    rows = [r[0] for r in rows]
    assert len(rows) > 0, 'Need at least one row in the iterator to be reduced'
    return np.concatenate(rows, axis=0)
