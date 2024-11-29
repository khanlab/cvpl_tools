from typing import Sequence

import cvpl_tools.tools.fs as tlfs
from cvpl_tools.fsspec import ensure_rdir_filesystem, RDirFileSystem
from cvpl_tools.im.process.base import lc_interpretable_napari, heatmap_logging, map_ncell_vector_to_total
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io


def lc_to_cc_count_lc_by_size_cc_list(lc: npt.NDArray[np.float64], ndim: int, os_shape: tuple,
                                      min_size: int, size_threshold, volume_weight, border_params) -> npt.NDArray[
    np.float64]:
    """Assumption: lc[:, ndim] is nvoxel and lc[:, ndim + 1] is edge_contact"""
    ncells = {}
    dc = []
    dc_idx_to_centroid_idx = {}
    for i in range(lc.shape[0]):
        nvoxel = lc[i, ndim].item()
        if nvoxel <= min_size:
            ncells[i] = 0.
        else:
            ncells[i] = 0.

            # if no voxel touch the boundary, we do not want to apply the edge penalty
            on_edge = lc[i, ndim + 1].item()
            if on_edge:
                dc_idx_to_centroid_idx[len(dc)] = i
                dc.append(lc[i, :ndim])
            else:
                ncells[i] = 1

            if nvoxel > size_threshold:
                ncells[i] += (nvoxel - size_threshold) * volume_weight
    if len(dc) == 0:
        dc_centroids = np.zeros((0, ndim), dtype=np.float64)
    else:
        dc_centroids = np.array(dc, dtype=np.float64)
    dc_ncells = lc_to_cc_count_lc_edge_penalized_cc_list(dc_centroids, chunks=os_shape, border_params=border_params,
                                                         block_index=(0,) * ndim)
    for dc_idx in dc_idx_to_centroid_idx:
        i = dc_idx_to_centroid_idx[dc_idx]
        ncells[i] += dc_ncells[dc_idx]
    ncells = np.array([ncells[i] for i in range(len(ncells))], dtype=np.float64)
    return ncells


def lc_to_cc_count_lc_by_size_np_features(lc: npt.NDArray[np.float64],
                                          block_info=None,
                                          **kwargs) -> npt.NDArray[np.float64]:
    if block_info is None:
        return np.zeros(tuple(), dtype=np.float64)

    slices = block_info[0]['array-location']
    ndim = len(slices)
    os_shape = tuple(s.stop for s in slices)
    cc_list = lc_to_cc_count_lc_by_size_cc_list(lc, ndim, os_shape=os_shape, **kwargs)
    features = np.concatenate((lc[:, :ndim], cc_list[:, None]), axis=1)
    return features


async def lc_to_cc_count_lc_by_size(
        lc: NDBlock[np.float64],
        ndim: int,
        min_size: int,
        size_threshold,
        volume_weight,
        border_params,
        reduce: bool = False,
        context_args: dict = None
) -> npt.NDArray[np.float64]:
    """Counting list of cells by size

    Several features:
    1. A size threshold, below which each contour is counted as a single cell (or part of a single cell,
    in the case it is neighbor to boundary of the image)
    2. Above size threshold, the contour is seen as a cluster of cells an estimate of cell count is given
    based on the volume of the contour
    3. For cells on the boundary location, their estimated ncell is penalized according to the distance
    between the cell centroid and the boundary of the image; if the voxels of the cell do not touch
    edge, this penalty does not apply
    4. A min_size threshold, below (<=) which the contour is simply discarded because it's likely just
    an artifact
    """

    if context_args is None:
        context_args = {}
    cache_url = context_args.get('cache_url')
    if cache_url is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(cache_url)
    viewer_args = context_args.get('viewer_args', {})
    viewer = viewer_args.get('viewer', None)

    async def feature_forward(lc):
        return await NDBlock.map_ndblocks([NDBlock(lc)],
                                          lc_to_cc_count_lc_by_size_np_features,
                                          out_dtype=np.float64,
                                          fn_args=dict(
                                              min_size=min_size,
                                              size_threshold=size_threshold,
                                              volume_weight=volume_weight,
                                              border_params=border_params,
                                          ))

    ndblock = await tlfs.cache_im(fn=feature_forward(lc),
                                  context_args=dict(
                                      cache_url=None if fs is None else fs['lc_by_size_features']
                                  ))

    dp = viewer_args.get('display_points', True)
    if viewer:
        if dp:
            features = await ndblock.reduce(force_numpy=True)
            features = features[features[:, -1] > 0., :]
            lc_interpretable_napari('bysize_ncells',
                                    features, viewer, ndim, ['ncells'])

        aggregate_ndblock: NDBlock[np.float64] = await tlfs.cache_im(
            fn=map_ncell_vector_to_total(ndblock),
            context_args=dict(
                cache_url=None if fs is None else fs['aggregate_ndblock']
            )
        )
        if dp:
            aggregate_features: npt.NDArray[np.float64] = await tlfs.cache_im(
                fn=aggregate_ndblock.reduce(force_numpy=True),
                context_args=dict(
                    cache_url=None if fs is None else fs['block_cell_count']
                )
            )
            lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                    ndim, ['ncells'], text_color='red')

        chunk_size = lc.get_chunksize()
        await heatmap_logging(aggregate_ndblock, None if fs is None else fs['cell_density_map'], viewer_args,
                              chunk_size)

    ndblock = ndblock.select_columns([-1])

    if reduce:
        ndblock = await ndblock.reduce(force_numpy=False)
    ndblock = ndblock.sum(keepdims=True)
    return ndblock


def lc_to_cc_count_lc_edge_penalized_cc_list(lc: npt.NDArray[np.float64],
                                             chunks,
                                             border_params: tuple,
                                             block_index: tuple) -> npt.NDArray[np.float64]:
    """Returns a cell count estimate for each contour in the list of centroids

    Args:
        lc: The list of centroids to be given cell estimates for
        chunks: The chunk sizes for each axes, in a tuple of nested arrays
        block_index: The index of the block which this lc corresponds to
        border_params: a tuple containing (intercept, dist_coeff, div_max)

    Returns:
        A 1-d list, each element is a scalar cell count for the corresponding contour centroid in lc
    """
    if isinstance(chunks[0], int):
        # Turn Sequence[int] to Sequence[Sequence[int]]
        # assume single numpy block, at index (0, 0, 0)
        chunks = tuple((chunks[i],) for i in range(len(chunks)))

    block_shape = np.array(
        tuple(chunks[i][block_index[i]] for i in range(len(chunks))),
        dtype=np.float64
    )
    midpoint = (block_shape * .5)[None, :]

    # compute border distances in each axis direction
    border_dists = np.abs((lc[:, :len(chunks)] + midpoint) % block_shape - (midpoint - .5))

    intercept, dist_coeff, div_max = border_params
    mults = 1 / np.clip(intercept + border_dists * dist_coeff, 1., div_max)
    cc_list = np.prod(mults, axis=1)
    return cc_list


def lc_to_cc_count_lc_edge_penalized_np_features(lc: npt.NDArray[np.float64], chunks, border_params, block_info=None
                                                 ) -> npt.NDArray[np.float64]:
    """Calculate cell counts, then concat centroid locations to the left of cell counts"""
    cc_list = lc_to_cc_count_lc_edge_penalized_cc_list(lc, chunks, border_params, block_info[0]['chunk-location'])
    features = np.concatenate((lc[:, :len(chunks)], cc_list[:, None]), axis=1)
    return features


async def lc_to_cc_count_lc_edge_penalized(
        lc: NDBlock[np.float64],
        chunks: Sequence[Sequence[int]] | Sequence[int],
        border_params: tuple[float, float, float] = (3., -.5, 2.),
        reduce: bool = False,
        context_args: dict = None) -> NDBlock[np.float64]:
    if isinstance(chunks[0], int):
        # Turn Sequence[int] to Sequence[Sequence[int]]
        # assume single numpy block, at index (0, 0, 0)
        chunks = tuple((chunks[i],) for i in range(len(chunks)))
    numblocks = tuple(len(c) for c in chunks)

    if context_args is None:
        context_args = {}
    curl = context_args.get('cache_url')
    if curl is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(curl)
    viewer_args = context_args.get('viewer_args')
    viewer = viewer_args.get('viewer', None)

    assert lc.get_numblocks() == numblocks, ('numblocks could not match up for the chunks argument '
                                             f'provided, expected {numblocks} but got '
                                             f'{lc.get_numblocks()}')

    async def feature_forward(lc: NDBlock[np.float64]) -> NDBlock[np.float64]:
        return await NDBlock.map_ndblocks([lc], lc_to_cc_count_lc_edge_penalized_np_features,
                                          out_dtype=np.float64, fn_args=dict(
                                                chunks=chunks, border_params=border_params
                                            ))
    ndblock = await tlfs.cache_im(fn=feature_forward(lc), context_args=dict(
        cache_url=None if fs is None else fs['lc_cc_edge_penalized']
    ))

    dp = viewer_args.get('display_points', True)
    if viewer:
        if viewer_args.get('display_checkerboard', True):
            checkerboard = await tlfs.cache_im(fn=lambda: cvpl_ome_zarr_io.dask_checkerboard(chunks), context_args=dict(
                cache_url=None if fs is None else fs['checkerboard'],
                viewer_args=viewer_args | dict(is_label=True)
            ))

        if dp:
            features = await ndblock.reduce(force_numpy=True)
            lc_interpretable_napari('lc_cc_edge_penalized', features, viewer,
                                    len(chunks), ['ncells'])

        aggregate_ndblock: NDBlock[np.float64] = await tlfs.cache_im(
            fn=map_ncell_vector_to_total(ndblock),
            context_args=dict(
                cache_url=None if fs is None else fs['aggregate_ndblock']
            )
        )
        if dp:
            aggregate_features: npt.NDArray[np.float64] = await tlfs.cache_im(
                fn=aggregate_ndblock.reduce(force_numpy=True),
                context_args=dict(
                    cache_url=None if fs is None else fs['block_cell_count']
                )
            )
            lc_interpretable_napari('block_cell_count', aggregate_features, viewer,
                                    len(chunks), ['ncells'], text_color='red')

        chunk_size = tuple(ax[0] for ax in chunks)
        await heatmap_logging(aggregate_ndblock, None if fs is None else fs['cell_density_map'], viewer_args, chunk_size)

    ndblock = ndblock.select_columns([-1])
    if reduce:
        ndblock = await ndblock.reduce(force_numpy=False)
    ndblock = ndblock.sum(keepdims=True)
    return ndblock

