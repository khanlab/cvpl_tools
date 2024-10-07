"""
This file defines SegProcess subclasses which converts any image to any other image of the same type (same type in
terms of the classification of different data types defined in the top of seg_process.py)
"""
import dask.array as da
import numpy.typing as npt
from skimage.transform import resize

from cvpl_tools.im.fs import CachePointer
from cvpl_tools.im.seg_process import (
    SegProcess,
    BlockToBlockProcess,
    block_to_block_forward
)


class DownsamplingByIntFactor(SegProcess):
    """*DEPRECATED*

    use cvpl_tools.im.algs.dask_ndinterp.scale_nearest instead
    """
    def __init__(self, factor: int | tuple = 2):
        """Creates a DownsamplingByIntFactor object

        Args:
            factor: Downsampling factor
        """
        super().__init__()
        self._factor = factor

    def forward(self,
                im: npt.NDArray | da.Array,
                cptr: CachePointer,
                viewer_args: dict = None
                ) -> npt.NDArray | da.Array:
        """Downsample the image and

        The function will display the downsampled image if viewer is provided in viewer_args, so make sure
        you set is_label to True if the image is a label instead of brightness/RGB

        If you want displayed image to be of the same scale of original, set the scale in viewer_args
        correspondingly

        Args:
            im: Image to be downsampled (np or dask array)
            cid: Caching id under the current tmp directory
            viewer_args: Arguments for the viewer

        Returns:
            The down-sampled image
        """
        if viewer_args is None:
            viewer_args = {}
        if isinstance(self._factor, int):
            factor = (self._factor,) * im.ndim
        else:
            factor = self._factor
        slices = tuple(slice(None, None, factor[i]) for i in range(im.ndim))
        im = cptr.im(fn=lambda: im[slices],
                     cache_level=1,
                     viewer_args=viewer_args)
        im.compute_chunk_sizes()
        return im


class UpsamplingByIntFactor(BlockToBlockProcess):
    """*DEPRECATED*

    use cvpl_tools.im.algs.dask_ndinterp.scale_nearest instead
    """

    def __init__(self, factor: int | tuple = 2, order: int = 0):
        """Creates a UpsamplingByIntFactor object

        Args:
            factor: Upsampling factor
            order: interpolation order (see skimage.transform.resize())
        """
        super().__init__(compute_chunk_sizes=True)
        self._factor = factor
        self._order = order

    def np_forward(self, im: npt.NDArray, block_info=None) -> npt.NDArray:
        """Calculate cell counts, then concat centroid locations to the left of cell counts"""
        start_dtype = im.dtype
        if isinstance(self._factor, int):
            factor = (self._factor,) * im.ndim
        else:
            factor = self._factor
        new_shape = tuple(im.shape[i] * factor[i] for i in range(im.ndim))
        im = resize(im, new_shape, order=self._order)
        end_dtype = im.dtype
        assert start_dtype == end_dtype, f'start:{start_dtype} end:{end_dtype} are not the same!'
        return im

    def forward(self,
                im: npt.NDArray | da.Array,
                cptr: CachePointer,
                viewer_args: dict = None
                ) -> npt.NDArray | da.Array:
        """Upsample the image

        The function will display the upsampled image if viewer is provided in viewer_args, so make sure
        you set is_label to True if the image is a label instead of brightness/RGB

        If you want displayed image to be of the same scale of original, set the scale in viewer_args
        correspondingly

        Args:
            im: Image to be upsampled (np or dask array)
            cid: Caching id under the current tmp directory
            viewer_args: Arguments for the viewer

        Returns:
            The up-sampled image
        """
        return block_to_block_forward(
            np_forward=self.np_forward,
            cptr=cptr,
            im=im,
            viewer_args=viewer_args,
            out_dtype=None,
            is_label=viewer_args.get('is_label', False),
            compute_chunk_sizes=True
        )
