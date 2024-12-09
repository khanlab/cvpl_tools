import numpy as np

from cvpl_tools.fsspec import RDirFileSystem

SUBJECT_ID = 'o22'  # **CHANGE THIS**
MINIMUM_SUBJECT_ID = None
BA_CHANNEL = None
PLAQUE_THRESHOLD, MAX_THRESHOLD = {
    'o22': (400., 1000.),  # 1
    'o23': (400., 1000.),  # 2
    'o24': (400., 1000.),  # 3
    'o24oldBlaze': (2000., 5000.),  # 4
    'F1A1Te4Blaze': (3000., 7500.),  # 5
    'F4A1Te3Blaze': (3000., 7500.),
    # **ADD MORE SUBJECTS HERE**
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
    BA_CHANNEL = np.s_[1]

if SUBJECT_ID == 'F4A1Te3Blaze':  # **SOME SUBJECTS DO NOT FOLLOW THE ABOVE FORMAT**
    OME_ZARR_PATH = 'gcs://khanlab-lightsheet/data/mouse_appmaptapoe/bids/sub-F4A1Te3/micr/sub-F4A1Te3_sample-brain_acq-blaze4x_SPIM.ome.zarr'

RUN_ON_FULL_IM = True
if not RUN_ON_FULL_IM:
    BA_CHANNEL = np.s_[0, 256:512, :, :]  # **CHANGE THIS**

SUBJECT_FOLDER = f'C:/Users/than83/Documents/progtools/datasets/subjects/subject_{SUBJECT_ID}'  # **CHANGE THIS**
NNUNET_CACHE_DIR = 'C:/Users/than83/Documents/progtools/datasets/nnunet/Cache_250epoch_Run20241126'  # **CHANGE THIS**

FIRST_DOWNSAMPLE_PATH = f'{SUBJECT_FOLDER}/first_downsample.ome.zarr'
SECOND_DOWNSAMPLE_PATH = f'{SUBJECT_FOLDER}/second_downsample.ome.zarr'
THIRD_DOWNSAMPLE_PATH = f'{SUBJECT_FOLDER}/third_downsample.ome.zarr'

THIRD_DOWNSAMPLE_BIAS_PATH = f'{SUBJECT_FOLDER}/third_downsample_bias.ome.zarr'
SECOND_DOWNSAMPLE_CORR_PATH = f'{SUBJECT_FOLDER}/second_downsample_corr.ome.zarr'

NNUNET_OUTPUT_TIFF_PATH = f'{SUBJECT_FOLDER}/second_downsample_nnunet.tiff'
GCS_NEG_MASK_TGT = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}_second_downsample_nnunet.tiff'
GCS_BIAS_PATH = f'gcs://khanlab-scratch/tmp/{SUBJECT_ID}_second_downsample_corr.tiff'
COILED_CACHE_DIR_PATH = f'gcs://khanlab-scratch/tmp/CacheDirectory_{SUBJECT_ID}'


def main(run_nnunet: bool = True, run_coiled_process: bool = True):
    import numpy as np
    import cvpl_tools.nnunet.current_im as current_im_py
    import cvpl_tools.nnunet.n4 as n4
    import cvpl_tools.ome_zarr.io as ome_io
    import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
    import asyncio
    import cvpl_tools.nnunet.triplanar as triplanar

    print(f'first downsample: from path {OME_ZARR_PATH}')
    first_downsample = current_im_py.downsample(
        OME_ZARR_PATH, reduce_fn=np.max, ndownsample_level=(1, 2, 2), ba_channel=BA_CHANNEL,
        write_loc=FIRST_DOWNSAMPLE_PATH
    )
    print(f'first downsample done. result is of shape {first_downsample.shape}')

    second_downsample = current_im_py.downsample(
        first_downsample, reduce_fn=np.max, ndownsample_level=(1,) * 3,
        write_loc=SECOND_DOWNSAMPLE_PATH
    )
    third_downsample = current_im_py.downsample(
        second_downsample, reduce_fn=np.max, ndownsample_level=(1,) * 3,
        write_loc=THIRD_DOWNSAMPLE_PATH
    )
    print(f'second and third downsample done. second_downsample.shape={second_downsample.shape}, third_downsample.shape={third_downsample.shape}')

    third_downsample_bias = n4.obtain_bias(third_downsample,
                                           write_loc=THIRD_DOWNSAMPLE_BIAS_PATH)
    print('third downsample bias done.')

    print(f'im.shape={second_downsample.shape}, bias.shape={third_downsample_bias.shape}; applying bias over image to obtain corrected image...')
    second_downsample_bias = dask_ndinterp.scale_nearest(third_downsample_bias, scale=(2, 2, 2),
                                                         output_shape=second_downsample.shape, output_chunks=(4, 4096, 4096)).persist()

    second_downsample_corr = current_im_py.apply_bias(second_downsample, (1,) * 3, second_downsample_bias, (1,) * 3)
    asyncio.run(ome_io.write_ome_zarr_image(SECOND_DOWNSAMPLE_CORR_PATH, da_arr=second_downsample_corr, MAX_LAYER=1))
    print('second downsample corrected image done')

    # first_downsample_correct_path = f'C:/Users/than83/Documents/progtools/datasets/lightsheet_downsample/sub-{SUBJECT_ID}_corrected.ome.zarr'
    # first_downsample_bias = dask_ndinterp.scale_nearest(third_downsample_bias, scale=(4, 4, 4),
    #                                                     output_shape=first_downsample.shape,
    #                                                     output_chunks=(4, 4096, 4096)).persist()
    # first_downsample_corr = current_im_py.apply_bias(first_downsample, (1,) * 3, first_downsample_bias, (1,) * 3)
    # asyncio.run(ome_io.write_ome_zarr_image(first_downsample_correct_path, da_arr=first_downsample_corr, MAX_LAYER=2))

    if run_nnunet is False:
        return

    pred_args = {
        "cache_url": NNUNET_CACHE_DIR,
        "test_im": SECOND_DOWNSAMPLE_CORR_PATH,
        "test_seg": None,
        "output": NNUNET_OUTPUT_TIFF_PATH,
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

    import cvpl_tools.nnunet.api as cvpl_nnunet_api

    if not RDirFileSystem(NNUNET_OUTPUT_TIFF_PATH).exists(''):
        cvpl_nnunet_api.upload_negmask(
            NNUNET_OUTPUT_TIFF_PATH,
            GCS_NEG_MASK_TGT,
            THIRD_DOWNSAMPLE_BIAS_PATH,
            f'{SUBJECT_FOLDER}/.temp',
            GCS_BIAS_PATH
        )

    ppm_to_im_upscale = (4, 8, 8)
    async def fn(dask_worker):
        await cvpl_nnunet_api.mousebrain_forward(
            dask_worker=dask_worker,
            CACHE_DIR_PATH=COILED_CACHE_DIR_PATH,
            ORIG_IM_PATH=OME_ZARR_PATH,
            NEG_MASK_PATH=GCS_NEG_MASK_TGT,
            GCS_BIAS_PATH=GCS_BIAS_PATH,
            BA_CHANNEL=BA_CHANNEL,
            MAX_THRESHOLD=MAX_THRESHOLD,
            ppm_to_im_upscale=ppm_to_im_upscale
        )
    cvpl_nnunet_api.coiled_run(fn=fn, nworkers=10, local_testing=False)


if __name__ == '__main__':
    main(run_nnunet=True, run_coiled_process=False)


