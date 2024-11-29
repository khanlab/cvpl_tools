import numpy as np
import numpy.typing as npt
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.im.process.os_to_lc as os_to_lc
import cvpl_tools.im.process.lc_to_cc as lc_to_cc
from cvpl_tools.fsspec import ensure_rdir_filesystem, RDirFileSystem
import dask.array as da


async def os_to_cc_count_os_by_size(
        os,
        size_threshold: int | float = 25.,
        volume_weight: float = 6e-3,
        border_params: tuple[float, float, float] = (3., -.5, 2.),
        min_size: int | float = 0,
        reduce: bool = False,
        context_args: dict = None,
):
    """Counting ordinal segmentation contours

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
    curl = context_args.get('cache_url')
    if curl is None:
        fs = None
    else:
        fs = ensure_rdir_filesystem(curl)
    viewer_args = context_args.get('viewer_args', {})

    lc = await os_to_lc.os_to_lc_direct(os, min_size=min_size, reduce=False, is_global=True,
                                        ex_statistics=['nvoxel', 'edge_contact'], context_args=dict(
            cache_url=None if fs is None else fs['os_to_lc'],
            viewer_args=viewer_args
        ))
    cc = await lc_to_cc.lc_to_cc_count_lc_by_size(lc, os.ndim, min_size=min_size,
                                                  size_threshold=size_threshold, volume_weight=volume_weight,
                                                  border_params=border_params, reduce=reduce, context_args=dict(
            cache_url=None if fs is None else fs['lc_to_cc'],
            viewer_args=viewer_args
        ))

    return cc
