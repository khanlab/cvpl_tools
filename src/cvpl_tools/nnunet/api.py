import numpy as np
import os


@property
def DEVICE():
    import torch
    return torch.device("cuda:0")


SUBJECT_ID = 'F1A1Te4Blaze'
MINIMUM_SUBJECT_ID = None
BA_CHANNEL = None
PLAQUE_THRESHOLD, MAX_THRESHOLD = {
    'o22': (400., 1000.),  # 1
    'o23': (400., 1000.),  # 2
    'o24': (400., 1000.),  # 3
    'o24oldBlaze': (2000., 5000.),  # 4
    'F1A1Te4Blaze': (3000., 7500.),  # 5
    'F1A2Te3Blaze': (400., 1000.),
}[SUBJECT_ID]
if SUBJECT_ID.endswith('oldBlaze'):
    MINIMUM_SUBJECT_ID = SUBJECT_ID[:-len('oldBlaze')]
    OME_ZARR_PATH = f'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids_oldBlaze/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-blaze_SPIM.ome.zarr'
    BA_CHANNEL = np.s_[0]
elif SUBJECT_ID.endswith('Blaze'):
    MINIMUM_SUBJECT_ID = SUBJECT_ID[:-len('Blaze')]
    OME_ZARR_PATH = f'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-blaze_SPIM.ome.zarr'
    BA_CHANNEL = np.s_[0]
else:
    MINIMUM_SUBJECT_ID = SUBJECT_ID
    OME_ZARR_PATH = f'Z:/projects/lightsheet_lifecanvas/bids/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-prestitched_SPIM.ome.zarr'
    BA_CHANNEL = np.s_[0]

TARGET_PATH = f'C:/Users/than83/Documents/progtools/datasets/lightsheet_downsample/sub-{SUBJECT_ID}.ome.zarr'
CORRECTED_PATH = f'C:/Users/than83/Documents/progtools/datasets/lightsheet_downsample/sub-{SUBJECT_ID}_corrected.ome.zarr'
BIAS_PATH = f'C:/Users/than83/Documents/progtools/datasets/annotated/canvas_{SUBJECT_ID}/bias.ome.zarr'  # may not exists
def get_eff_bias_path() -> None | str:  # USE THIS INSTEAD OF BIAS_PATH TO OBTAIN BIAS IMAGE
    eff_bias_path = None
    if os.path.exists(BIAS_PATH):
        eff_bias_path = BIAS_PATH
    elif os.path.exists(f'{BIAS_PATH}.zip'):
        eff_bias_path = f'{BIAS_PATH}.zip'
    return eff_bias_path
REDUCE_FN = np.max


# COILED PROCESSING
NEG_MASK_TGT = f'C:/Users/than83/Documents/progtools/datasets/nnunet/Cache_250epoch_Run20241120/dir_cache_predict/0_{SUBJECT_ID}.tiff'
GCS_NEG_MASK_TGT = f'gcs://khanlab-scratch/tmp/0_{SUBJECT_ID}.tiff'
GCS_BIAS_PATH = f'gcs://khanlab-scratch/tmp/0_{SUBJECT_ID}_bias.tiff'
FULL_RES_IM = OME_ZARR_PATH
CACHE_DIRECTORY_NAME = f'CacheDirectoryBlaze_{SUBJECT_ID}'
