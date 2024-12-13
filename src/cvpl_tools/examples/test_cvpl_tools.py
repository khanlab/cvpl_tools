from cvpl_tools.examples.mousebrain_processing import main, get_subject
import cvpl_tools.nnunet.triplanar as triplanar

SUBJECT_ID = 'o22'
SUBJECTS_DIR = f'subjects'
NNUNET_CACHE_DIR = f'nnunet_250epoch'
subject = get_subject(SUBJECT_ID, SUBJECTS_DIR, NNUNET_CACHE_DIR)
main(subject=subject, run_nnunet=False, run_coiled_process=False)

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
