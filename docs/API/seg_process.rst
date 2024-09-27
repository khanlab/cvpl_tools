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

.. autoclass:: cvpl_tools.im.seg_process.SegProcess
    :members:
.. autoclass:: cvpl_tools.im.seg_process.BlockToBlockProcess
    :members:
.. autofunction:: cvpl_tools.im.seg_process.lc_interpretable_napari

.. rubric:: Built-in Subclasses for SegProcess

.. autoclass:: cvpl_tools.im.seg_process.GaussianBlur
    :members:
.. autoclass:: cvpl_tools.im.seg_process.BSPredictor
    :members:
.. autoclass:: cvpl_tools.im.seg_process.SimpleThreshold
    :members:
.. autoclass:: cvpl_tools.im.seg_process.BlobDog
    :members:
.. autoclass:: cvpl_tools.im.seg_process.SumScaledIntensity
    :members:
.. autoclass:: cvpl_tools.im.seg_process.BinaryAndCentroidListToInstance
    :members:

bs_to_os
********
binary segmentation to ordinal segmentation

This section contains algorithms whose input is binary (0-1) segmentation mask, and output is instance segmentation
(0-N) integer mask where the output ndarray is of the same shape as input.

.. autoclass:: cvpl_tools.im.process.bs_to_os.DirectBSToOS
    :members:
.. autoclass:: cvpl_tools.im.process.bs_to_os.Watershed3SizesBSToOS
    :members:

lc_to_cc
********
list of centroids to cell counts

This section contains algorithms whose input is a 2d array or a 2d array of each block describing the centroid
locations and meta information about the objects associated with the centroids in each block. The output is a single
number summarizing statistics for each block.

.. autoclass:: cvpl_tools.im.process.lc_to_cc.CountLCBySize
    :members:
.. autoclass:: cvpl_tools.im.process.lc_to_cc.CountLCEdgePenalized
    :members:

os_to_cc
********
oridnal segmentation to cell counts

This section contains algorithms whose input is instance segmentation (0-N) integer mask where the output is a single
number summarizing statistics for each block.

.. autoclass:: cvpl_tools.im.process.os_to_cc.CountOSBySize
    :members:

os_to_lc
********
ordinal segmentation to list of centroids

This section contains algorithms whose input is instance segmentation (0-N) integer mask where the output is a list
of centroids with meta information.

.. autoclass:: cvpl_tools.im.process.os_to_lc.DirectOSToLC
    :members:

any_to_any
**********
other

This sections contain image processing steps whose inputs and outputs may adapt to different types of data or are not
adequately described by the current classifications.

.. autoclass:: cvpl_tools.im.process.any_to_any.DownsamplingByIntFactor
    :members:
.. autoclass:: cvpl_tools.im.process.any_to_any.UpsamplingByIntFactor
    :members:

