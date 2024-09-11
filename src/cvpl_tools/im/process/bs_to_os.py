from cvpl_tools.im.seg_process import SegProcess, BlockToBlockProcess
import numpy as np
import numpy.typing as npt
import dask.array as da
from cvpl_tools.im.ndblock import NDBlock
import cvpl_tools.im.algorithms as algorithms
from scipy.ndimage import (
    label as instance_label
)
from dask_image.ndmeasure import label as dask_label


class DirectBSToOS(BlockToBlockProcess):
    IDX_COMPACT = 0
    IDX_UNIQUE = 1

    def __init__(self, is_global: bool = False):
        """Converts a n-dimensional binary mask to an int32 ordinal mask based on connected components.

        If is_global is True, return a global instance mask instead of chunk level instance mask, and
        in the forward() method, a second value indicating indices adjacency will be returned of type
        scipy.sparse.csr_array and each chunk will be separately segmented with globally unique
        indices. See https://github.com/dask/dask-image/blob/main/dask_image/ndmeasure/_utils/_label.py
        for more information.

        Args:
            is_global: If True, label globally instead of at chunk level
        """
        super().__init__(np.int32, is_label=True)
        self.is_global = is_global

    def forward(self,
                im: npt.NDArray | da.Array | NDBlock,
                cid: str | None = None,
                viewer_args=None
                ) -> npt.NDArray | da.Array | NDBlock:
        if not self.is_global or isinstance(im, np.ndarray):
            return super().forward(im, cid, viewer_args)

        if viewer_args is None:
            viewer_args = {}

        def compute():
            nonlocal im
            if isinstance(im, NDBlock):
                is_ndblock = True
                im = im.as_dask_array()
            else:
                is_ndblock = False
            im, nlbl = dask_label(im)
            if is_ndblock:
                im = NDBlock(im)
            return im

        im = self.tmpdir.cache_im(compute,
                                  cid=cid,
                                  cache_level=1,
                                  viewer_args=viewer_args | dict(is_label=self.is_label))
        return im

    def np_forward(self, bs: npt.NDArray[np.uint8], block_info=None) -> npt.NDArray[np.int32]:
        lbl_im, nlbl = instance_label(bs)
        return lbl_im


class Watershed3SizesBSToOS(BlockToBlockProcess):
    def __init__(self,
                 size_thres=60.,
                 dist_thres=1.,
                 rst=None,
                 size_thres2=100.,
                 dist_thres2=1.5,
                 rst2=60.):
        super().__init__(np.int32, is_label=True)
        self.size_thres = size_thres
        self.dist_thres = dist_thres
        self.rst = rst
        self.size_thres2 = size_thres2
        self.dist_thres2 = dist_thres2
        self.rst2 = rst2

    def np_forward(self, bs: npt.NDArray[np.uint8], block_info=None) -> npt.NDArray[np.int32]:
        lbl_im = algorithms.round_object_detection_3sizes(bs,
                                                          size_thres=self.size_thres,
                                                          dist_thres=self.dist_thres,
                                                          rst=self.rst,
                                                          size_thres2=self.size_thres2,
                                                          dist_thres2=self.dist_thres2,
                                                          rst2=self.rst2,
                                                          remap_indices=True)
        return lbl_im
    # TODO: better visualization of this stage
