import copy

from cvpl_tools.fsspec import RDirFileSystem, ensure_rdir_filesystem
from cvpl_tools.tools.fs import cdir_init, cdir_commit
import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
import torch
import glob
import os
import tifffile
import numpy.typing as npt
import numpy as np
from PIL import Image
import json
import cvpl_tools.nnunet.mwcmd as nnunet_cmd


def get_axes_names(triplanar: bool):
    if triplanar:
        axes_names = ('yx', 'xz', 'zy')
    else:
        axes_names = ('yx',)
    return axes_names


def get_axes_tups(triplanar: bool):
    if triplanar:
        tups = (('yx', (0, 1, 2)), ('xz', (1, 2, 0)), ('zy', (2, 0, 1)))
    else:
        tups = (('yx', (0, 1, 2)),)
    return tups


def calc_expect_ndim(stack_channels: int, is_label: bool) -> int:
    expect_ndim = 4
    if stack_channels == 0 or is_label:
        expect_ndim = 3
    return expect_ndim


def ensure_expect_ndim(im_slices: npt.NDArray, stack_channels: int, is_label: bool) -> int:
    expect_ndim = calc_expect_ndim(stack_channels, is_label)
    assert im_slices.ndim == expect_ndim, f'got expect_ndim={expect_ndim}, but ndim={im_slices.ndim}, shape={im_slices.shape}'
    return expect_ndim


def volume_to_slices(vol: npt.NDArray[np.uint8], stack_channels: int, is_label: bool) -> npt.NDArray[np.uint8]:
    """slices have shape (z, ch, y, x)"""
    im_slices = []
    for i in range(vol.shape[0] - 2 * stack_channels):
        im_slice = take_slices_from_volume(vol, i, stack_channels, is_label)
        im_slices.append(im_slice)
    im_slices = np.array(im_slices)
    ensure_expect_ndim(im_slices, stack_channels, is_label)
    return im_slices


def slices_to_volume(im_slices: npt.NDArray[np.uint8] | torch.Tensor,
                     stack_channels: int,
                     is_label: bool) -> npt.NDArray[np.uint8] | torch.Tensor:
    """slices have shape (z, ch, y, x) if 4d, or (z, y, x) if 3d"""
    expect_dim = ensure_expect_ndim(im_slices, stack_channels, is_label)

    if expect_dim == 3:
        return im_slices

    bottom = im_slices[:stack_channels, 0]
    middle = im_slices[:, 1]
    top = im_slices[-stack_channels:, 2]
    if isinstance(im_slices, np.ndarray):
        vol = np.concatenate((bottom, middle, top), axis=0)
    else:
        vol = torch.concatenate((bottom, middle, top), dim=0)
    return vol


def load_volume_pair(im_path: str, seg_path: str, max_threshold: float) -> tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    print(f'loading volume pairs at im_path={im_path} and seg_path={seg_path}')
    im = cvpl_ome_zarr_io.load_dask_array_from_path(im_path, mode='r', level=0)
    if im.ndim == 4:
        print(
            f'Image at im_path={im_path} found 4 dimensions, treat first dimension as channels and take the slice ch=1')
        im = im[1]
    im = (np.clip(im.compute() / max_threshold, 0., 1.) * 255).astype('uint8')

    if seg_path is None:
        seg = None
    else:
        seg = tifffile.imread(seg_path)
        # seg[im < 255 * .05] = 0
        seg = (seg > 0).astype("uint8")
    return im, seg


def recover_volumes(im_path: str, stack_channels: int, is_label: bool, ignore_suffix: int = 0, is_npz: bool = False):
    """The volumes are guaranteed to be yielded in order"""
    volumes = {}
    for path in glob.glob(im_path):
        folder, filename = os.path.split(path)
        ids = filename[:filename.find('.') - ignore_suffix].split('_')
        ids = tuple(int(i) for i in ids)
        if ids[0] not in volumes:
            volumes[ids[0]] = {ids[1]: path}
        else:
            volumes[ids[0]][ids[1]] = path
    for i in sorted(tuple(volumes.keys())):
        volume = volumes[i]
        volume = tuple(volume[j] for j in range(len(volume)))
        print(f'volume at id={i} found nslices={len(volume)}')
        if is_npz:
            im_slices = [np.load(img_path)['probabilities'][1] for img_path in volume]
        else:
            im_slices = [Image.open(img_path) for img_path in volume]
        im_slices = np.array([np.array(im) for im in im_slices])
        if im_slices.ndim == 4:
            im_slices = im_slices.transpose(0, 3, 1, 2)
        print(f'after loading from png, the im_slices is of shape {im_slices.shape}')
        vol = slices_to_volume(im_slices, stack_channels=stack_channels, is_label=is_label)
        print(f'after converting to volume, the volume is of shape {vol.shape}')
        yield vol


def dice_on_volume_pair(volume_bin1, volume_bin2) -> float:
    if isinstance(volume_bin1, torch.Tensor):
        volume_bin1 = volume_bin1.detach().cpu().numpy()
    volume_bin1 = volume_bin1 > 0
    if isinstance(volume_bin2, torch.Tensor):
        volume_bin2 = volume_bin2.detach().cpu().numpy()
    volume_bin2 = volume_bin2 > 0
    intersect = volume_bin1 & volume_bin2
    dice = intersect.sum().astype(np.float64) * 2 / (volume_bin1.sum() + volume_bin2.sum())
    return dice.item()


def load_volume_np_or_torch(path: str, convert_to_bin: bool = False) -> npt.NDArray | torch.Tensor:
    if path.endswith('.npy'):
        arr = np.load(path)
    else:
        arr = torch.load(path, map_location=torch.device('cpu')).detach().numpy()
    if convert_to_bin:
        arr = arr > 0
    return arr


def take_slices_from_volume(im_vol, j, dj, is_label: bool):
    assert dj >= 0, f'got dj={dj}'
    if dj == 0 or is_label:
        im_slice = im_vol[j + dj]
    else:
        im_slice = im_vol[j:j + dj * 3:dj]
    return im_slice


def save_im_slice_at(im_slices, j, dj, is_label: bool, path):
    expect_ndim = ensure_expect_ndim(im_slices, dj, is_label)
    if expect_ndim == 3:  # this means the slice will have ndim=2, and needs not transpose to make sure d
        im_slice = im_slices[j]
    else:
        im_slice = im_slices[j].transpose(1, 2, 0)
    Image.fromarray(im_slice).save(path)


def write_slices_to_png(im_slices: npt.NDArray[np.uint8], stack_channels: int, is_label: bool, folder_path: str,
                        vol_id: int | str, suffix: str = ''):
    if isinstance(vol_id, int):
        vol_id = str(vol_id)
    RDirFileSystem(folder_path).ensure_dir_exists(remove_if_already_exists=False)
    for j in range(im_slices.shape[0]):
        save_im_slice_at(im_slices, j, stack_channels, is_label, f'{folder_path}/{vol_id}_{j:04}{suffix}.png')


def create_nnunet_dataset_from_volumes(
        train_im_volume_paths,
        train_seg_volume_paths,
        max_threshold,
        dataset_dir: str,
        axes: tuple[int, ...] = None,
        stack_channels: int = 0,
        suffix: str = ''):
    """Create a nnunet dataset (trainset or testset) from volumes under a dataset

    Recall nnunet format specifies dataset image/label path to be:
    "nnUNet_raw/Dataset001_Lightsheet/imagesTr/..."
    "nnUNet_raw/Dataset001_Lightsheet/imagesTs/..."

    Args:
        train_im_volume_paths: Path to the image 3d volumes
        train_seg_volume_paths: Path to the segmentation 3d volumes
        max_threshold: Max threshold for rescaling the original image
        dataset_dir: Path to dataset, up to the "Dataset{id}_{name}" folder
        axes: Axes to transpose before save; this is mostly for triplanar training
        stack_channels: The jump steps in "z"-direction between adjacent channels in the inputs
    """
    if stack_channels == 0:
        channel_names = {
            '0': "grey"
        }
    else:
        channel_names = {
            '0': "bottom",
            '1': "mid",
            '2': "top",
        }

    imagesTrPath = f'{dataset_dir}/imagesTr'
    imagesTsPath = f'{dataset_dir}/imagesTs'
    labelsTrPath = f'{dataset_dir}/labelsTr'
    labelsTsPath = f'{dataset_dir}/labelsTs'

    total_ntrain = 0
    for i in range(len(train_im_volume_paths)):
        im_vol, seg_vol = load_volume_pair(train_im_volume_paths[i], train_seg_volume_paths[i], max_threshold)
        if axes is not None:
            im_vol = im_vol.transpose(*axes)
            write_slices_to_png(volume_to_slices(im_vol, stack_channels, is_label=False),
                                stack_channels,
                                is_label=False,
                                folder_path=imagesTrPath,
                                vol_id=i,
                                suffix=suffix)  # only images will have corresponding suffix
            if seg_vol is not None:
                seg_vol = seg_vol.transpose(*axes)
                write_slices_to_png(volume_to_slices(seg_vol, stack_channels, is_label=True),
                                    stack_channels,
                                    is_label=True,
                                    folder_path=labelsTrPath,
                                    vol_id=i)
        total_ntrain += im_vol.shape[0] - stack_channels * 2
        print(f'created dataset with i={i}, total_ntrain={total_ntrain}')

    dataset_json = {
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "negative": 1,
        },
        "numTraining": total_ntrain,
        "file_ending": ".png",
    }
    with open(f'{dataset_dir}/dataset.json', 'w', encoding='utf-8') as fd:
        json.dump(dataset_json, fd, indent=4, ensure_ascii=False)


def set_nnunet_train_environ(nnunet_dir: str):
    os.environ['nnUNet_raw'] = f'{nnunet_dir}/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = f'{nnunet_dir}/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = f'{nnunet_dir}/nnUNet_results'


def create_triplanar_datasets(train_im_volume_paths,
                              train_seg_volume_paths,
                              max_threshold,
                              cache_url: str,
                              triplanar: bool,
                              dataset_id: int,
                              stack_channels: int = 0,
                              run_plan_and_preprocess: bool = True,
                              ) -> tuple[str, ...]:
    """Returns the one (or 3 if triplanar) created sub-cache-dirs for datasets"""
    cpaths = []
    for axes_name, axes in get_axes_tups(triplanar):
        sub_sdir = f'{cache_url}/{axes_name}'
        cpaths.append(sub_sdir)

        nnUNet_raw_path = f'{sub_sdir}/nnUNet_raw'
        print(f'Testing if folder {nnUNet_raw_path} exists.')
        if cdir_init(nnUNet_raw_path).commit:
            print(f'Directory at path {nnUNet_raw_path} already exists, will not run plan_and_preprocess over there')
            continue
        else:
            print(f'Folder does not exist, creating dataset...')

        folder_path = f'{nnUNet_raw_path}/Dataset{dataset_id:03}_Lightsheet{dataset_id}'
        os.makedirs(folder_path)
        create_nnunet_dataset_from_volumes(
            train_im_volume_paths,
            train_seg_volume_paths,
            max_threshold,
            folder_path,
            axes,
            stack_channels=stack_channels,
            suffix='_0000'
        )

        if run_plan_and_preprocess:
            set_nnunet_train_environ(sub_sdir)
            args = ['nnUNetv2_plan_and_preprocess',
                    '-d',
                    str(dataset_id),
                    '--verify_dataset_integrity']
            nnunet_cmd.run(args)
        else:
            print(f'run_plan_and_preprocess=False, skip planning phase')

        cdir_commit(nnUNet_raw_path)
    return tuple(cpaths)


def train_triplanar(train_args: dict):
    train_args = copy.copy(train_args)
    cache_url = train_args.pop('cache_url')
    triplanar = train_args.pop('triplanar')
    stack_channels = train_args.pop('stack_channels')
    fold = train_args.pop('fold')
    nepoch = train_args.pop('nepoch')
    max_threshold = train_args.pop('max_threshold')

    train_im = train_args.pop('train_im')
    train_seg = train_args.pop('train_seg')
    print(f'globbing OME ZARR images\ntrain: {train_im}\nseg: {train_seg}')
    train_im_volume_paths = sorted(glob.glob(train_im))
    train_seg_volume_paths = sorted(glob.glob(train_seg))
    print(f'globbing done.\ntrain: {train_im_volume_paths}\nseg: {train_seg_volume_paths}')

    dataset_id = train_args.pop('dataset_id', 1)
    train_cdir = f'{cache_url}/train'

    train_cache_dirs = create_triplanar_datasets(
        train_im_volume_paths, train_seg_volume_paths,
        max_threshold,
        cache_url=train_cdir, triplanar=triplanar,
        dataset_id=dataset_id,
        stack_channels=stack_channels,
        run_plan_and_preprocess=True
    )

    axes_names = get_axes_names(triplanar)
    trainer_name = f'nnUNetTrainer_{nepoch}epochs' if nepoch != 1000 else 'nnUNetTrainer'
    with open(f'{train_cdir}/args.txt', 'w') as outfile:
        outfile.write(str(stack_channels))
        outfile.write('\n')
        if trainer_name == 'nnUNetTrainer':
            towrite = 'default_trainer'
        else:
            towrite = trainer_name
        outfile.write(towrite)
        outfile.write('\n')
        outfile.write(str(max_threshold))
    for i in range(len(axes_names)):
        nnunet_dir = train_cache_dirs[i]

        if fold == 'all':
            folds = [str(i) for i in range(5)]
        else:
            fold_int = int(fold)
            assert 4 >= fold_int >= 0, fold
            folds = [fold]
        set_nnunet_train_environ(nnunet_dir)
        for cur_fold in folds:
            fold_dir = f'{nnunet_dir}/nnUNet_results/Dataset{dataset_id:03}_Lightsheet{dataset_id}/{trainer_name}__nnUNetPlans__2d/fold_{cur_fold}'
            print(f'Testing if folder {fold_dir} exists.')
            if os.path.exists(
                    f'{nnunet_dir}/nnUNet_results/Dataset{dataset_id:03}_Lightsheet{dataset_id}/{trainer_name}__nnUNetPlans__2d/fold_{cur_fold}'):
                print(f'Folder exists, skip fold {cur_fold}')
                continue
            else:
                print(f'Folder does not exist, start training...')

            args = ['nnUNetv2_train',
                    str(dataset_id),
                    '2d',
                    cur_fold]
            if trainer_name != 'default_trainer':
                args += ['-tr', trainer_name]
            nnunet_cmd.run(args)


def predict_triplanar_volume(model_dir: str, volume: npt.NDArray[np.uint8], stack_channels: int,
                             tmp_pred_path: str, vol_id: int, dataset_id: int,
                             trainer_name: str, fold: str, probabilities: bool) -> npt.NDArray:
    tmp_pred_path_pred = RDirFileSystem(f'{tmp_pred_path}/pred')
    if not tmp_pred_path_pred.exists('') or len(tmp_pred_path_pred.listdir('')) == 0:
        print(f'predict_triplanar_volume() got input volume of shape {volume.shape}')
        im_slices = volume_to_slices(volume, stack_channels=stack_channels, is_label=False)
        print(f'predict_triplanar_volume() computed im_slices of shape {im_slices.shape}')
        for path in (tmp_pred_path, f'{tmp_pred_path}/input', tmp_pred_path_pred.url):
            RDirFileSystem(path).ensure_dir_exists(remove_if_already_exists=False)
        write_slices_to_png(im_slices, stack_channels, False, f'{tmp_pred_path}/input', vol_id=vol_id,
                            suffix='_0000')

        os.environ['nnUNet_results'] = f'{model_dir}/nnUNet_results'
        # predict on images in input_dir using model from nnunet_dir
        args = ['nnUNetv2_predict',
                '-i', f'{tmp_pred_path}/input',
                '-o', tmp_pred_path_pred.url,
                '-d', str(dataset_id),
                '-c', '2d']
        if fold != 'all':
            args += ['-f', fold]
        if trainer_name not in ('None', 'nnUNetTrainer'):
            args += ['-tr', trainer_name.strip()]
        if probabilities:
            args += ['--save_probabilities']
        if not torch.cuda.is_available():
            args += ['-device', 'cpu']
        nnunet_cmd.run(args)
    else:
        print(f'Predicted results already exist, directly retrieve...')

    print(f'predict_triplanar_volume loading results from {tmp_pred_path_pred.url}/*.png')
    pred_vol = next(recover_volumes(f'{tmp_pred_path_pred.url}/*.png',
                                    stack_channels=stack_channels,
                                    is_label=True))
    return pred_vol


def predict_triplanar(predict_args: dict):
    cache_url = predict_args["cache_url"]
    triplanar = predict_args["triplanar"]
    penalize_edge = predict_args["penalize_edge"]
    weights = predict_args["weights"]
    dataset_id = predict_args["dataset_id"]
    fold = predict_args["fold"]
    output_path = predict_args["output"]
    use_cache = predict_args.get('use_cache', False)

    print('globbing OME ZARR images')
    test_im = predict_args.pop('test_im')
    test_im_volume_paths = sorted(glob.glob(test_im))
    test_seg = predict_args.pop('test_seg')
    if test_seg is None or test_seg == '':
        test_seg_volume_paths = [None] * len(test_im_volume_paths)
    else:
        test_seg_volume_paths = sorted(glob.glob(test_seg))
    print('globbing done.')

    train_cdir = f'{cache_url}/train'
    predict_cdir = f'{cache_url}/predict'
    predict_cdir_fs = RDirFileSystem(predict_cdir)
    if not use_cache and predict_cdir_fs.exists(''):
        predict_cdir_fs.rm('', recursive=True)
    input_cdir = f'{predict_cdir}/input'  # input images in nnunet format
    prediction_cdir = f'{predict_cdir}/prediction'  # model outputs from unet

    with open(f'{train_cdir}/args.txt', 'r') as infile:
        lines = list(infile.readlines())
        stack_channels = int(lines[0])
        trainer_name = lines[1].strip()
        if trainer_name == 'default_trainer':
            trainer_name = 'nnUNetTrainer'
        max_threshold = float(lines[2])
        print(f'stack_channels={stack_channels}')
        print(f'trainer_name={trainer_name}')
        print(f'max_threshold={max_threshold}')

    axes_tups = get_axes_tups(triplanar)

    for axes_name, axes in axes_tups:
        ax_cdir = f'{input_cdir}/{axes_name}'
        if not cdir_init(ax_cdir).commit:
            create_nnunet_dataset_from_volumes(
                test_im_volume_paths,
                test_seg_volume_paths,
                max_threshold,
                ax_cdir,
                axes,
                stack_channels=stack_channels,
                suffix='_0000'
            )
            cdir_commit(ax_cdir)

    print(f'Predict triplanar using model from folder {cache_url}')

    im_vols_list = []

    for i in range(len(axes_tups)):
        axes_name, axes = axes_tups[i]
        input_dir = f'{input_cdir}/{axes_name}'
        im_vols = recover_volumes(f'{input_dir}/imagesTr/*.png',
                                  stack_channels=stack_channels,
                                  is_label=True,
                                  ignore_suffix=5)  # '_0000'
        # im_vols = map(lambda vol: vol.transpose(axes), im_vols)
        im_vols_list.append(im_vols)

    print(f'im_vols are ready in iterator form, now predict on them one by one...')

    has_any = False
    for i, im_vol in enumerate(im_vols_list[0]):
        has_any = True
        print(f'-------------------prediction on volume {i}-------------------')
        print(f'recovered volume of im_vols_list[0][{i}] is of shape {im_vol.shape}')

        if penalize_edge:
            weight_vol = np.zeros(im_vol.shape, dtype=np.float32)
        else:
            weight_vol = 0.

        if len(axes_tups) == 1:
            axes_name = axes_tups[0][0]
            nnunet_dir = f'{train_cdir}/{axes_name}'
            volj = predict_triplanar_volume(nnunet_dir,
                                            im_vol,
                                            stack_channels=stack_channels,
                                            tmp_pred_path=f'{prediction_cdir}/{axes_name}',
                                            vol_id=i,
                                            dataset_id=dataset_id,
                                            trainer_name=trainer_name,
                                            fold=fold,
                                            probabilities=False)
            seg_vol = volj > 0
        else:
            seg_vol = np.zeros(im_vol.shape, dtype=np.float32)
            for j in range(len(axes_tups)):
                axes_name = axes_tups[j][0]
                if j == 0:
                    volj = im_vol
                else:
                    volj = next(im_vols_list[j])
                nnunet_dir = f'{train_cdir}/{axes_name}'
                volj = predict_triplanar_volume(nnunet_dir,
                                                volj,
                                                stack_channels=stack_channels,
                                                tmp_pred_path=f'{prediction_cdir}/{axes_name}',
                                                vol_id=i,
                                                dataset_id=dataset_id,
                                                trainer_name=trainer_name,
                                                fold=fold,
                                                probabilities=True)
                print(f'The predicted volume is of type {volj.dtype}')
                volj = np.clip(volj, a_min=-256, a_max=256)
                volj = volj / volj.std()
                rev = np.argsort(axes_tups[j][1]).tolist()

                height = volj.shape[0]
                volj = volj.transpose(*rev)
                path_tosave = f'{predict_cdir}/{i}_{axes_name}.tiff'
                print(f'volj to be saved at {path_tosave} is of shape {volj.shape}')
                tifffile.imwrite(path_tosave, volj > 0)

                if weights is None:
                    wjf = 1.
                else:
                    assert len(weights) == len(axes_tups), f'got len={len(weights)}'
                    wjf = weights[j]
                if penalize_edge:
                    # top and bottom slices should receive less weight
                    wj = (np.arange(height, dtype=np.float32)
                          / (volj.shape[0] - 1)) * wjf
                    wj = np.sin(np.pi * wj)[:, None, None].transpose(*rev)
                    weight_vol += wj
                    seg_vol += volj * wj
                else:
                    weight_vol += wjf
                    seg_vol += volj * wjf

            seg_vol /= weight_vol
            seg_vol = seg_vol > 0  # of type np.bool
            print(f'seg_vol to be saved is of shape {seg_vol.shape}')

        if output_path == "":
            output_path = f'{predict_cdir}/{i}.tiff'
        else:
            predict_cdir_fs.rm('', recursive=True)
        tifffile.imwrite(output_path, seg_vol)

    assert has_any, f'No volume found under path {input_cdir}'
