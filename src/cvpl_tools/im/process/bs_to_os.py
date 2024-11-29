from cvpl_tools.im.process.base import block_to_block_forward
import numpy as np
import numpy.typing as npt
import cvpl_tools.im.algorithms as algorithms
import copy


async def bs_to_os_watershed3sizes(bs: npt.NDArray[np.uint8],
                                   size_thres=60.,
                                   dist_thres=1.,
                                   rst=None,
                                   size_thres2=100.,
                                   dist_thres2=1.5,
                                   rst2=60.,
                                   context_args: dict = None
                                   ):
    def np_forward(bs: npt.NDArray[np.uint8], block_info=None) -> npt.NDArray[np.int32]:
        lbl_im = algorithms.round_object_detection_3sizes(bs,
                                                          size_thres=size_thres,
                                                          dist_thres=dist_thres,
                                                          rst=rst,
                                                          size_thres2=size_thres2,
                                                          dist_thres2=dist_thres2,
                                                          rst2=rst2,
                                                          remap_indices=True)
        return lbl_im

    if context_args is None:
        context_args = dict()
    else:
        context_args = copy.copy(context_args)
    context_args['viewer_args'] = context_args.get('viewer_args', {}) | dict(is_label=True)
    # TODO: better visualization of this stage
    return await block_to_block_forward(
        np_forward=np_forward,
        im=bs,
        context_args=context_args,
        out_dtype=np.uint32
    )
