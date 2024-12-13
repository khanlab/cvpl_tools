import numpy as np
import os


@property
def DEVICE():
    import torch
    return torch.device("cuda:0")


def coiled_run(fn, nworkers: int = 1, local_testing: bool = False):
    import coiled
    # from coiled.credentials.google import send_application_default_credentials
    import time

    if local_testing:
        from distributed import Client

        cluster = None
        client = Client(threads_per_worker=16, n_workers=1)
    else:
        cluster = coiled.Cluster(n_workers=nworkers)
        # send_application_default_credentials(cluster)
        client = cluster.get_client()

        while client.status == "running":
            cur_nworkers = len(client.scheduler_info()["workers"])
            if cur_nworkers < nworkers:
                print('Current # of workers:', cur_nworkers, '... Standby.')
            else:
                print(f'All {nworkers} workers started.')
                break
            time.sleep(1)

    workers = list(client.scheduler_info()["workers"].keys())
    print(client.run(fn, workers=[workers[0]]))

    client.close()
    if cluster is not None:
        cluster.close()


def upload_negmask(NEG_MASK_TGT: str, GCS_NEG_MASK_TGT: str, BIAS_SRC: str, LOCAL_TEMP: str, GCS_BIAS_PATH: str):
    from cvpl_tools.fsspec import RDirFileSystem, copyfile
    import cvpl_tools.ome_zarr.io as ome_io
    import tifffile

    tgt = RDirFileSystem(GCS_NEG_MASK_TGT)
    print(f'Copying negative mask from {NEG_MASK_TGT} to {GCS_NEG_MASK_TGT}...')
    assert not tgt.exists(''), f'Target already exists at path {tgt.url}!'
    print(f'Verified target is a writable location...')
    copyfile(NEG_MASK_TGT, GCS_NEG_MASK_TGT)
    print(f'Copying is done, verifying the target exists...')
    assert tgt.exists(''), f'Target does not exists at path {tgt.url}!'
    print(f'Target exists, copy is finished.')

    tgt = RDirFileSystem(GCS_BIAS_PATH)
    print(f'\nCopying bias from {BIAS_SRC} to {GCS_BIAS_PATH}...')
    assert not tgt.exists(''), f'Target already exists at path {tgt.url}!'
    print(f'Verified target is a writable location...')
    arr = ome_io.load_dask_array_from_path(BIAS_SRC, mode='r', level=0).compute()
    tifffile.imwrite(LOCAL_TEMP, arr)
    copyfile(LOCAL_TEMP, GCS_BIAS_PATH)
    print(f'Copying is done, verifying the target exists...')
    assert tgt.exists(''), f'Target does not exists at path {tgt.url}!'
    print(f'Target exists, copy is finished.')


async def mousebrain_forward(dask_worker,
                             CACHE_DIR_PATH: str,
                             ORIG_IM_PATH: str,
                             NEG_MASK_PATH: str,
                             GCS_BIAS_PATH: str,
                             BA_CHANNEL: int,
                             MAX_THRESHOLD: float,
                             ppm_to_im_upscale: tuple,
                             ):
    # passing of dask_worker is credit to fjetter at https://github.com/dask/distributed/issues/8152
    from dask.distributed import Worker
    assert isinstance(dask_worker, Worker)

    client = dask_worker._get_client()  # once _get_client() is called, the following Client.current() calls returns the same client

    import enum
    import sys
    import numcodecs

    from cvpl_tools.fsspec import RDirFileSystem

    import numpy as np
    import cvpl_tools.tools.fs as tlfs
    import cvpl_tools.im.algs.dask_label as dask_label
    import cvpl_tools.im.process.base as seg_process
    import cvpl_tools.im.process.os_to_lc as sp_os_to_lc
    import cvpl_tools.im.process.lc_to_cc as sp_lc_to_cc
    import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
    import dask.array as da

    THRESHOLD = .45

    import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
    import tifffile

    logfile_stdout = open('log_stdout.txt', mode='w')
    logfile_stderr = open('log_stderr.txt', mode='w')
    sys.stdout = tlfs.MultiOutputStream(sys.stdout, logfile_stdout)
    sys.stderr = tlfs.MultiOutputStream(sys.stderr, logfile_stderr)

    if False and RDirFileSystem(CACHE_DIR_PATH).exists(''):
        RDirFileSystem(CACHE_DIR_PATH).rm('', recursive=True)

    cache_dir_fs = RDirFileSystem(CACHE_DIR_PATH)
    cache_dir_fs.ensure_dir_exists(remove_if_already_exists=False)

    import threading
    print(f'tid:::: {threading.get_ident()}')

    np.set_printoptions(precision=1)

    cur_im = da.from_zarr(cvpl_ome_zarr_io.load_zarr_group_from_path(
        path=ORIG_IM_PATH, mode='r', level=0
    ))[BA_CHANNEL].rechunk(chunks=(256, 512, 512))
    assert cur_im.ndim == 3
    print(f'imshape={cur_im.shape}')

    viewer = None  # napari.Viewer(ndisplay=2)
    storage_options = dict(
        dimension_separator='/',
        preferred_chunksize=(2, 4096, 4096),
        multiscale=4,
        compressor=numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
    )
    viewer_args = dict(
        viewer=viewer,
        display_points=True,
        display_checkerboard=True,
    )
    context_args = dict(
        viewer_args=viewer_args,
        storage_options=storage_options,
    )

    async def compute_per_pixel_multiplier():
        with RDirFileSystem(NEG_MASK_PATH).open('', mode='rb') as infile:
            neg_mask = tifffile.imread(infile)
        with RDirFileSystem(GCS_BIAS_PATH).open('', mode='rb') as infile:
            bias = tifffile.imread(infile)
        neg_mask = da.from_array(neg_mask, chunks=(64, 64, 64))
        bias = da.from_array(bias, chunks=(32, 32, 32))
        bias = dask_ndinterp.scale_nearest(bias, scale=(2, 2, 2),
                                           output_shape=neg_mask.shape, output_chunks=(64, 64, 64))
        return (1 - neg_mask) / bias

    ppm_layer_args = dict(name='ppm', colormap='bop blue')
    ppm = (await tlfs.cache_im(fn=compute_per_pixel_multiplier(),
                               context_args=context_args | dict(
                                   cache_url=cache_dir_fs['per_pixel_multiplier'],
                                   layer_args=ppm_layer_args
                               )))

    async def compute_masking():
        im = cur_im * dask_ndinterp.scale_nearest(ppm, scale=ppm_to_im_upscale,
                                                  output_shape=cur_im.shape, output_chunks=(256, 512, 512))
        im = (im / MAX_THRESHOLD).clip(0., 1.)
        return im.astype(np.float16)

    im_layer_args = dict(name='im', colormap='bop blue')
    cur_im = (await tlfs.cache_im(compute_masking(), context_args=context_args | dict(
        cache_url=cache_dir_fs['input_im'],
        layer_args=im_layer_args
    ))).astype(np.float32)

    async def alg(im, context_args):
        fs = context_args['cache_url']
        bs = await seg_process.in_to_bs_simple_threshold(threshold=THRESHOLD, im=im,
                                                         context_args=context_args | dict(
                                                             cache_url=fs['bs']))
        os, nlbl = await dask_label.label(bs, output_dtype=np.int32,
                                          context_args=context_args | dict(cache_url=fs['os']))

        if context_args is None:
            context_args = {}
        viewer_args = context_args.get('viewer_args', {})
        lc = await sp_os_to_lc.os_to_lc_direct(os, min_size=8, reduce=False, is_global=True,
                                               ex_statistics=['nvoxel', 'edge_contact'], context_args=dict(
                cache_url=fs['lc'],
                viewer_args=viewer_args
            ))
        cc = await sp_lc_to_cc.lc_to_cc_count_lc_by_size(lc,
                                                         os.ndim,
                                                         min_size=8,
                                                         size_threshold=200.,
                                                         volume_weight=5.15e-3,
                                                         border_params=(3., -.5, 2.3),
                                                         reduce=False,
                                                         context_args=dict(
                                                             cache_url=fs['cc'],
                                                             viewer_args=viewer_args
                                                         ))
        return lc, cc

    import time
    stime = time.time()
    lc, cc = await alg(
        cur_im,
        context_args=context_args | dict(cache_url=cache_dir_fs['GLOBAL_LABEL'])
    )
    midtime = time.time()
    print(f'forward elapsed: {midtime - stime}')
    lc = lc.reduce(force_numpy=True)
    ncell_list = await cc.reduce(force_numpy=True)

    print(f'ending  elapsed: {time.time() - midtime}')
    cnt = ncell_list.sum().item()

    with cache_dir_fs.open('final_lc.npy', mode='wb') as fd:
        np.save(fd, lc)

