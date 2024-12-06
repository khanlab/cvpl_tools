.. _seg_process:

cvpl_tools/im/seg_process.py
============================

View source at `seg_process.py <https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/im/seg_process.py>`_.

Q: Why are there two baseclasses :code:`SegProcess` and :code:`BlockToBlockProcess`? When I define my own pipeline,
which class should I be subclassing from?

A: :code:`BlockToBlockProcess` is a wrapper around :code:`SegProcess` for code whose input and output block sizes
are the same.
For general processing whose output are list of centroids, or when input shape of any block is not the same as
output shape of that block, use :code:`BlockToBlockProcess`.

.. rubric:: APIs

.. autofunction:: cvpl_tools.im.process.base.block_to_block_forward
.. autofunction:: cvpl_tools.im.process.base.lc_interpretable_napari

.. rubric:: Built-in function processes

.. autofunction:: cvpl_tools.im.process.base.in_to_bs_custom
.. autofunction:: cvpl_tools.im.process.base.in_to_bs_simple_threshold
.. autofunction:: cvpl_tools.im.process.base.in_to_lc_blobdog_forward
.. autofunction:: cvpl_tools.im.process.base.in_to_cc_sum_scaled_intensity
.. autofunction:: cvpl_tools.im.process.base.bs_lc_to_os_forward

bs_to_os
********
binary segmentation to ordinal segmentation

This section contains algorithms whose input is binary (0-1) segmentation mask, and output is instance segmentation
(0-N) integer mask where the output ndarray is of the same shape as input.

.. autofunction:: cvpl_tools.im.process.bs_to_os.bs_to_os_watershed3sizes

lc_to_cc
********
list of centroids to cell counts

This section contains algorithms whose input is a 2d array or a 2d array of each block describing the centroid
locations and meta information about the objects associated with the centroids in each block. The output is a single
number summarizing statistics for each block.

.. autofunction:: cvpl_tools.im.process.lc_to_cc.lc_to_cc_count_lc_by_size
.. autofunction:: cvpl_tools.im.process.lc_to_cc.lc_to_cc_count_lc_edge_penalized

os_to_cc
********
oridnal segmentation to cell counts

This section contains algorithms whose input is instance segmentation (0-N) integer mask where the output is a single
number summarizing statistics for each block.

.. autofunction:: cvpl_tools.im.process.os_to_cc.os_to_cc_count_os_by_size

os_to_lc
********
ordinal segmentation to list of centroids

This section contains algorithms whose input is instance segmentation (0-N) integer mask where the output is a list
of centroids with meta information.

.. autofunction:: cvpl_tools.im.process.os_to_lc.os_to_lc_direct

any_to_any
**********
other

This sections contain image processing steps whose inputs and outputs may adapt to different types of data or are not
adequately described by the current classifications.


