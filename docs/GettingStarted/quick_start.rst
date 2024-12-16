.. _quick_start:

Quickstart
##########

Installation
************

Create an Anaconda environment, and install `torch <https://pytorch.org/get-started/locally/>`_ (with GPU
if you need training besides prediction). Check the following environment are installed as well (below list
is for :code:`cvpl_tools` version 0.8.2; also use Python 3.11 if you are working with SPIMquant
environment):

.. code-block:: Python

    python = ">=3.9"
    numpy = ">=1.23"
    nibabel = ">=5.2.1"
    pillow = ">=7.1.0"
    scipy = ">=1.12.0"
    matplotlib = ">=3.9"
    scikit-image = ">=0.22.0"
    napari = ">=0.4.19"
    zarr = ">=2.17.0"
    dask = ">=2024.2.0"
    dask-image = ">=2024.5.3"
    ome-zarr = ">=0.9.0"
    fsspec = ">=2024.6.1"
    nnunetv2 = ">=2.5.1"

Then :code:`pip install cvpl_tools` to finish the installation.

OME ZARR
********

Create an example OME ZARR, write it to disk and read back:

.. code-block:: Python

    import dask.array as da
    import cvpl_tools.ome_zarr.io as ome_io
    import napari
    import numpy as np
    import asyncio

    viewer = napari.Viewer(ndisplay=2)
    da_array = da.from_array(np.arange(16).reshape((4, 4)))
    print(f'print array:\n{da_array.compute()}')
    asyncio.run(ome_io.write_ome_zarr_image('test.ome.zarr', da_arr=da_array))
    read_back = ome_io.load_dask_array_from_path('test.ome.zarr', mode='r', level=0)  # always use level=0 for original resolution
    print(f'read back:\n{read_back.compute()}')  # should print the same content
    viewer.add_image(read_back, contrast_limits=[0, 15])
    viewer.show(block=True)

Read and write can be done on network location. An example of read and display:

.. code-block:: Python

    import cvpl_tools.ome_zarr.io as ome_io
    import cvpl_tools.ome_zarr.napari.add as nadd  # read but only for display purpose
    import napari

    viewer = napari.Viewer(ndisplay=2)
    OME_ZARR_PATH = 'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-F1A1Te4/micr/sub-F1A1Te4_sample-brain_acq-blaze_SPIM.ome.zarr'
    read_back = ome_io.load_dask_array_from_path(OME_ZARR_PATH, mode='r', level=0)
    print(f'read back shape:{read_back.shape}')  # Reading metadata
    nadd.group_from_path(viewer, OME_ZARR_PATH)
    viewer.show(block=True)  # Displaying the image in Napari

nn-UNet
*******

Download the training (o22)/testing (o23) annotations from Google Drive as follows:

1. `o22 <https://drive.google.com/file/d/1S8xNdWD1pznAMaKsrjncBySkHVO5b3HR/view?usp=drive_link>`_

2. `o23 <https://drive.google.com/file/d/1xhxLA0RnxoL3c1ojSGTCAIB5yp-Wfrgh/view?usp=drive_link>`_

Put the canvas_o22.tiff file in the same folder as below script. Then pair o22 annotation tiff file with the
corresponding training input image volume and start training:

.. code-block:: Python

    from cvpl_tools.examples.mousebrain_processing import mousebrain_processing, get_subject
    import cvpl_tools.nnunet.triplanar as triplanar

    SUBJECT_ID = 'o22'
    SUBJECTS_DIR = f'subjects'
    NNUNET_CACHE_DIR = f'nnunet_250epoch'
    GCS_PARENT_PATH = 'gcs://khanlab-scratch/tmp'
    subject = get_subject(SUBJECT_ID, SUBJECTS_DIR, NNUNET_CACHE_DIR, GCS_PARENT_PATH)
    mousebrain_processing(subject=subject, run_nnunet=False, run_coiled_process=False)

    train_args = {
        "cache_url": NNUNET_CACHE_DIR,
        "train_im": subject.SECOND_DOWNSAMPLE_CORR_PATH,  # image
        "train_seg": 'canvas_o22.tiff',  # label
        "nepoch": 250,
        "stack_channels": 0,
        "triplanar": False,
        "dataset_id": 1,
        "fold": '0',
        "max_threshold": 7500.,
    }
    triplanar.train_triplanar(train_args)

When training is finished, you can start predicting on the o23 image volume and compare the results. Training
may take several hours, and if you want to skip this step, then you can download the result model from
`Zenodo <https://zenodo.org/records/14419797>`_ and extract the folder to rename it as :code:`'nnunet_250epoch'`.
Prediction code is as follows:

.. code-block:: Python

    from cvpl_tools.examples.mousebrain_processing import mousebrain_processing, get_subject
    import cvpl_tools.nnunet.triplanar as triplanar

    SUBJECT_ID = 'o23'  # now predict on o23
    SUBJECTS_DIR = f'subjects'
    NNUNET_CACHE_DIR = f'nnunet_250epoch'
    GCS_PARENT_PATH = 'gcs://khanlab-scratch/tmp'
    subject = get_subject(SUBJECT_ID, SUBJECTS_DIR, NNUNET_CACHE_DIR, GCS_PARENT_PATH)
    mousebrain_processing(subject=subject, run_nnunet=False, run_coiled_process=False)

    pred_args = {
        "cache_url": NNUNET_CACHE_DIR,
        "test_im": subject.SECOND_DOWNSAMPLE_CORR_PATH,
        "test_seg": None,
        "output": 'canvas_o23_pred.tiff',
        "dataset_id": 1,
        "fold": '0',
        "triplanar": False,
        "penalize_edge": False,
        "weights": None,
    }
    triplanar.predict_triplanar(pred_args)

After prediction, load up two tiffs and compute an overlap measure like IOU or DICE score:

.. code-block:: Python

    import tifffile
    import napari
    import cvpl_tools.ome_zarr.napari.add as nadd

    mask_manual = tifffile.imread('canvas_o23.tiff') > 0
    mask_pred = tifffile.imread('canvas_o23_pred.tiff') > 0
    intersect = mask_manual & mask_pred
    dice = intersect.sum().astype(np.float64) * 2 / (mask_manual.sum() + mask_pred.sum())
    print(f'DICE score obtained: {dice.item(): .4f}')

    # display the results in napari
    viewer = napari.Viewer()
    nadd.group_from_path(viewer, subject.SECOND_DOWNSAMPLE_CORR_PATH, kwargs=dict(name='image_volume'))
    viewer.add_labels(mask_manual, name='mask_manual')
    viewer.add_labels(mask_pred, name='mask_predicted')
    viewer.show(block=True)

