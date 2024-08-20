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
.. autoclass:: cvpl_tools.im.seg_process.ScaledSumIntensity
    :members:
.. autoclass:: cvpl_tools.im.seg_process.DirectBSToOS
    :members:
.. autoclass:: cvpl_tools.im.seg_process.Watershed3SizesBSToOS
    :members:
.. autoclass:: cvpl_tools.im.seg_process.BinaryAndCentroidListToInstance
    :members:
.. autoclass:: cvpl_tools.im.seg_process.DirectOSToLC
    :members:
.. autoclass:: cvpl_tools.im.seg_process.CountLCEdgePenalized
    :members:
.. autoclass:: cvpl_tools.im.seg_process.CountOSBySize
    :members:

