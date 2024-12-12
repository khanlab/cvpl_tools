import numpy as np

from cvpl_tools.fsspec import RDirFileSystem
from dataclasses import dataclass


@dataclass
class Subject:
    SUBJECT_ID: str = None
    MINIMUM_SUBJECT_ID: str = None
    BA_CHANNEL = None
    PLAQUE_THRESHOLD: float = None
    MAX_THRESHOLD: float = None

    OME_ZARR_PATH = None
    SUBJECT_FOLDER = None
    NNUNET_CACHE_DIR = None

    FIRST_DOWNSAMPLE_PATH = None
    SECOND_DOWNSAMPLE_PATH = None
    THIRD_DOWNSAMPLE_PATH = None

    THIRD_DOWNSAMPLE_BIAS_PATH = None
    SECOND_DOWNSAMPLE_CORR_PATH = None
    FIRST_DOWNSAMPLE_CORR_PATH = None

    NNUNET_OUTPUT_TIFF_PATH = None
    GCS_NEG_MASK_TGT = None
    GCS_BIAS_PATH = None
    COILED_CACHE_DIR_PATH = None


subjects4x = ('F4A1Te3Blaze', 'F6A2Te3Blaze', 'M1A1Te3Blaze', 'M1A2Te3Blaze', 'M7A1Te4Blaze')
THRESHOLD_TABLE = {
    'o22': (400., 1000.),  # 1
    'o23': (400., 1000.),  # 2
    # 'o24': (400., 1000.),  # 3
    # 'o24oldBlaze': (2000., 5000.),  # 4
    'F1A1Te4Blaze': (3000., 7500.),  # 5
    'F1A2Te3Blaze': (3000., 7500.),
    'F4A1Te3Blaze': (3000., 7500.),
    'F6A2Te3Blaze': (3000., 7500.),
    'M1A1Te3Blaze': (3000., 7500.),
    'M1A2Te3Blaze': (3000., 7500.),
    'M3A2Te3Blaze': (3000., 7500.),
    'M4A2Te3Blaze': (3000., 7500.),
    'M7A1Te4Blaze': (3000., 7500.),
    # **ADD MORE SUBJECTS HERE**
}
ALL_SUBJECTS = list(THRESHOLD_TABLE.keys())

def get_subject(SUBJECT_ID):
    subject = Subject()
    subject.SUBJECT_ID = SUBJECT_ID

    subject.PLAQUE_THRESHOLD, subject.MAX_THRESHOLD = THRESHOLD_TABLE[SUBJECT_ID]
    if SUBJECT_ID.endswith('oldBlaze'):
        MINIMUM_SUBJECT_ID = SUBJECT_ID[:-len('oldBlaze')]
        OME_ZARR_PATH = f'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids_oldBlaze/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-blaze_SPIM.ome.zarr'
        BA_CHANNEL = np.s_[0]
    elif SUBJECT_ID.endswith('Blaze'):
        MINIMUM_SUBJECT_ID = SUBJECT_ID[:-len('Blaze')]
        if SUBJECT_ID in subjects4x:  # **SOME SUBJECTS DO NOT FOLLOW THE ABOVE FORMAT**
            OME_ZARR_PATH = f'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-blaze4x_SPIM.ome.zarr'
        else:
            OME_ZARR_PATH = f'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-blaze_SPIM.ome.zarr'
        BA_CHANNEL = np.s_[0]
    else:
        MINIMUM_SUBJECT_ID = SUBJECT_ID
        OME_ZARR_PATH = f'Z:/projects/lightsheet_lifecanvas/bids/sub-{MINIMUM_SUBJECT_ID}/micr/sub-{MINIMUM_SUBJECT_ID}_sample-brain_acq-prestitched_SPIM.ome.zarr'
        BA_CHANNEL = np.s_[1]

    RUN_ON_FULL_IM = False
    if not RUN_ON_FULL_IM:
        BA_CHANNEL = np.s_[BA_CHANNEL, 256:512, :, :]  # **CHANGE THIS**

    subject.MINIMUM_SUBJECT_ID = MINIMUM_SUBJECT_ID
    subject.OME_ZARR_PATH = OME_ZARR_PATH
    subject.BA_CHANNEL = BA_CHANNEL

    FOLDER = 'C:/ProgrammingTools/ComputerVision/RobartsResearch/data/lightsheet/tmp/mousebrain_processing'
    subject.SUBJECT_FOLDER = f'{FOLDER}/subjects/subject_{SUBJECT_ID}'  # **CHANGE THIS**
    subject.NNUNET_CACHE_DIR = f'{FOLDER}/nnunet_250epoch_Run20241126'  # **CHANGE THIS**

    subject.FIRST_DOWNSAMPLE_PATH = f'{subject.SUBJECT_FOLDER}/first_downsample.ome.zarr'
    subject.SECOND_DOWNSAMPLE_PATH = f'{subject.SUBJECT_FOLDER}/second_downsample.ome.zarr'
    subject.THIRD_DOWNSAMPLE_PATH = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}/third_downsample.ome.zarr'

    subject.THIRD_DOWNSAMPLE_BIAS_PATH = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}/third_downsample_bias.ome.zarr'
    subject.SECOND_DOWNSAMPLE_CORR_PATH = f'{subject.SUBJECT_FOLDER}/second_downsample_corr.ome.zarr'
    subject.FIRST_DOWNSAMPLE_CORR_PATH = f'{subject.SUBJECT_FOLDER}/first_downsample_corr.ome.zarr'

    subject.NNUNET_OUTPUT_TIFF_PATH = f'{subject.SUBJECT_FOLDER}/second_downsample_nnunet.tiff'
    subject.GCS_NEG_MASK_TGT = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}_second_downsample_nnunet.tiff'
    subject.GCS_BIAS_PATH = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}_second_downsample_corr.tiff'
    subject.COILED_CACHE_DIR_PATH = f'gcs://khanlab-scratch/tmp/CacheDirectory_{SUBJECT_ID}'

    return subject


def main(subject: Subject, run_nnunet: bool = True, run_coiled_process: bool = True):
    import numpy as np
    import cvpl_tools.nnunet.lightsheet_preprocess as lightsheet_preprocess
    import cvpl_tools.nnunet.n4 as n4
    import cvpl_tools.ome_zarr.io as ome_io
    import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
    import asyncio
    import cvpl_tools.nnunet.triplanar as triplanar
    import cvpl_tools.nnunet.api as cvpl_nnunet_api

    print(f'first downsample: from path {subject.OME_ZARR_PATH}')
    first_downsample = lightsheet_preprocess.downsample(
        subject.OME_ZARR_PATH, reduce_fn=np.max, ndownsample_level=(1, 2, 2), ba_channel=subject.BA_CHANNEL,
        write_loc=subject.FIRST_DOWNSAMPLE_PATH
    )
    print(f'first downsample done. result is of shape {first_downsample.shape}')

    second_downsample = lightsheet_preprocess.downsample(
        first_downsample, reduce_fn=np.max, ndownsample_level=(1,) * 3,
        write_loc=subject.SECOND_DOWNSAMPLE_PATH
    )
    third_downsample = lightsheet_preprocess.downsample(
        second_downsample, reduce_fn=np.max, ndownsample_level=(1,) * 3,
        write_loc=subject.THIRD_DOWNSAMPLE_PATH
    )
    print(f'second and third downsample done. second_downsample.shape={second_downsample.shape}, third_downsample.shape={third_downsample.shape}')

    async def compute_bias(dask_worker):
        third_downsample = ome_io.load_dask_array_from_path(subject.THIRD_DOWNSAMPLE_PATH, mode='r', level=0)
        await n4.obtain_bias(third_downsample, write_loc=subject.THIRD_DOWNSAMPLE_BIAS_PATH)
    if not RDirFileSystem(subject.THIRD_DOWNSAMPLE_BIAS_PATH).exists(''):
        cvpl_nnunet_api.coiled_run(fn=compute_bias, nworkers=1, local_testing=False)
    third_downsample_bias = ome_io.load_dask_array_from_path(subject.THIRD_DOWNSAMPLE_BIAS_PATH, mode='r', level=0)
    print('third downsample bias done.')

    print(f'im.shape={second_downsample.shape}, bias.shape={third_downsample_bias.shape}; applying bias over image to obtain corrected image...')
    second_downsample_bias = dask_ndinterp.scale_nearest(third_downsample_bias, scale=(2, 2, 2),
                                                         output_shape=second_downsample.shape, output_chunks=(4, 4096, 4096)).persist()

    second_downsample_corr = lightsheet_preprocess.apply_bias(second_downsample, (1,) * 3, second_downsample_bias, (1,) * 3)
    asyncio.run(ome_io.write_ome_zarr_image(subject.SECOND_DOWNSAMPLE_CORR_PATH, da_arr=second_downsample_corr, MAX_LAYER=1))
    print('second downsample corrected image done')

    if run_nnunet is False:
        return

    if not RDirFileSystem(subject.NNUNET_OUTPUT_TIFF_PATH).exists(''):
        pred_args = {
            "cache_url": subject.NNUNET_CACHE_DIR,
            "test_im": subject.SECOND_DOWNSAMPLE_CORR_PATH,
            "test_seg": None,
            "output": subject.NNUNET_OUTPUT_TIFF_PATH,
            "dataset_id": 1,
            "fold": '0',
            "triplanar": False,
            "penalize_edge": False,
            "weights": None,
            "use_cache": False,
        }
        triplanar.predict_triplanar(pred_args)

    if run_coiled_process is False:
        return

    if not RDirFileSystem(subject.GCS_NEG_MASK_TGT).exists(''):
        cvpl_nnunet_api.upload_negmask(
            subject.NNUNET_OUTPUT_TIFF_PATH,
            subject.GCS_NEG_MASK_TGT,
            subject.THIRD_DOWNSAMPLE_BIAS_PATH,
            f'{subject.SUBJECT_FOLDER}/.temp',
            subject.GCS_BIAS_PATH
        )

    ppm_to_im_upscale = (4, 8, 8)
    async def fn(dask_worker):
        await cvpl_nnunet_api.mousebrain_forward(
            dask_worker=dask_worker,
            CACHE_DIR_PATH=subject.COILED_CACHE_DIR_PATH,
            ORIG_IM_PATH=subject.OME_ZARR_PATH,
            NEG_MASK_PATH=subject.GCS_NEG_MASK_TGT,
            GCS_BIAS_PATH=subject.GCS_BIAS_PATH,
            BA_CHANNEL=subject.BA_CHANNEL,
            MAX_THRESHOLD=subject.MAX_THRESHOLD,
            ppm_to_im_upscale=ppm_to_im_upscale
        )
    cvpl_nnunet_api.coiled_run(fn=fn, nworkers=10, local_testing=False)


if __name__ == '__main__':
    for ID in ('M7A1Te4Blaze',):
        if ID in ('M4A2Te3Blaze', 'o22', 'o23'):
            continue
        print(f'Starting prediction on subject {ID}')
        subject = get_subject(ID)

        main(subject=subject, run_nnunet=True, run_coiled_process=True)
        print(f'Finished predicting on subject {ID}')


