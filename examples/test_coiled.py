import numpy as np

LOCAL_TESTING = True
USE_GCS = False
COMP_SLI = np.index_exp[:, :256, :256]


async def main(dask_worker):
    # passing of dask_worker is credit to fjetter at https://github.com/dask/distributed/issues/8152
    from dask.distributed import Worker
    assert isinstance(dask_worker, Worker)

    client = dask_worker._get_client()  # once _get_client() is called, the following Client.current() calls returns the same client

    import enum
    import sys
    from abc import ABC
    from typing import Type
    import numcodecs

    from cvpl_tools.fsspec import RDirFileSystem

    import numpy as np
    import cvpl_tools.im.fs as imfs
    import cvpl_tools.im.seg_process as seg_process
    import cvpl_tools.im.process.bs_to_os as sp_bs_to_os
    import cvpl_tools.im.process.os_to_lc as sp_os_to_lc
    import cvpl_tools.im.process.os_to_cc as sp_os_to_cc
    import cvpl_tools.im.process.lc_to_cc as sp_lc_to_cc
    import cvpl_tools.ome_zarr.io as cvpl_ome_zarr_io
    from cvpl_tools.im.ndblock import NDBlock
    import dask.array as da

    class CountingMethod(enum.Enum):
        """Specifies the algorithm to use for cell counting"""
        # thresholding removing darker area below threshold -> sum over the intensity of the rest
        # works ok but not perfect
        SUM_INTENSITY = 0

        # simple thresholding -> watershed -> direct os to lc -> edge penalized count lc
        # does not work well because large clumps of cells are counted as one
        BORDER_PENALIZED_THRES_WATERSHED = 1

        # simple thresholding -> watershed -> count instance segmentation by contour size
        # this works better than sum intensity
        THRES_WATERSHED_BYSIZE = 2

        # use # of centroids found by blobdog to give a direct cell count
        # will overestimate the number of cells by a lot, due to over-counting cells around the edge of the image
        BLOBDOG = 3

        # same as above, but penalize the number of cells found around the edge
        BORDER_PENALIZED_BLOBDOG = 4

        # use blobdog centroids to split threshold masks into smaller contours
        THRES_BLOBDOG_BYSIZE = 5

        # Globally convert binary segmentation to ordinal segmentation, then to list of centroids
        GLOBAL_LABEL = 6

    class PipelineSuperclass(seg_process.SegProcess, ABC):
        def __init__(self):
            super().__init__()

    def get_pipeline(no: CountingMethod) -> Type[PipelineSuperclass]:
        match no:
            case CountingMethod.SUM_INTENSITY:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.ssi = seg_process.SumScaledIntensity(
                            scale=.00766,
                            min_thres=.4,
                            reduce=False,
                            spatial_box_width=None
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        return await self.ssi.forward(im, cptr=cptr, viewer_args=viewer_args)

            case CountingMethod.BORDER_PENALIZED_THRES_WATERSHED:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.thres = seg_process.SimpleThreshold(.45)
                        self.watershed = sp_bs_to_os.Watershed3SizesBSToOS(
                            size_thres=60.,
                            dist_thres=1.,
                            rst=None,
                            size_thres2=100.,
                            dist_thres2=1.5,
                            rst2=60.
                        )
                        self.os_to_lc = sp_os_to_lc.DirectOSToLC(min_size=8, reduce=False)

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        mask: np.ndarray[np.uint8] = await self.thres.forward(
                            im, cptr=cdir.cache(cid='mask'), viewer_args=viewer_args)
                        os = await self.watershed.forward(mask, cptr=cdir.cache(cid='os'), viewer_args=viewer_args)
                        lc: NDBlock[np.float32] = await self.os_to_lc.forward(
                            os, cptr=cdir.cache(cid='lc'), viewer_args=viewer_args)
                        chunks = os.shape if isinstance(os, np.ndarray) else os.chunks
                        count_lc = sp_lc_to_cc.CountLCEdgePenalized(
                            chunks=chunks,
                            border_params=(3., -.5, 2.),
                            reduce=False
                        )
                        cc = await count_lc.forward(lc, cptr=cdir.cache(cid='count_lc'), viewer_args=viewer_args)
                        return cc

            case CountingMethod.THRES_WATERSHED_BYSIZE:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.thres = seg_process.SimpleThreshold(.45)
                        self.watershed = sp_bs_to_os.Watershed3SizesBSToOS(
                            size_thres=60.,
                            dist_thres=1.,
                            rst=None,
                            size_thres2=100.,
                            dist_thres2=1.5,
                            rst2=60.
                        )
                        self.os_to_cc = sp_os_to_cc.CountOSBySize(
                            size_threshold=200.,
                            volume_weight=5.15e-3,
                            border_params=(3., -.5, 2.),
                            min_size=8,
                            reduce=False
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        mask: np.ndarray[np.uint8] = await self.thres.forward(
                            im, cptr=cdir.cache(cid='mask'), viewer_args=viewer_args)
                        os: np.ndarray[np.int32] = await self.watershed.forward(
                            mask, cptr=cdir.cache(cid='os'), viewer_args=viewer_args)
                        cc = await self.os_to_cc.forward(
                            os, cptr=cdir.cache(cid='count_by_size'), viewer_args=viewer_args)
                        return cc

            case CountingMethod.BLOBDOG:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.blobdog = seg_process.BlobDog(
                            min_sigma=2.,
                            max_sigma=4.,
                            threshold=.1,
                            reduce=False
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        lc: NDBlock = await self.blobdog.forward(
                            im, cptr=cdir.cache(cid='blobdog'), viewer_args=viewer_args)
                        chunks = im.shape if isinstance(im, np.ndarray) else im.chunks
                        count_lc = sp_lc_to_cc.CountLCEdgePenalized(
                            chunks=chunks,
                            border_params=(1., 0., 1.),
                            reduce=False
                        )
                        cc = await count_lc.forward(lc, cdir.cache(cid='count_lc'), viewer_args=viewer_args)
                        return cc

            case CountingMethod.BORDER_PENALIZED_BLOBDOG:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.blobdog = seg_process.BlobDog(
                            min_sigma=2.,
                            max_sigma=4.,
                            threshold=.1,
                            reduce=False
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        lc: NDBlock = await self.blobdog.forward(
                            im, cptr=cdir.cache(cid='blobdog'), viewer_args=viewer_args)
                        chunks = im.shape if isinstance(im, np.ndarray) else im.chunks
                        count_lc = sp_lc_to_cc.CountLCEdgePenalized(
                            chunks=chunks,
                            border_params=(3., -.5, 2.1),
                            reduce=False
                        )
                        cc = await count_lc.forward(lc, cdir.cache(cid='count_lc'), viewer_args=viewer_args)
                        return cc

            case CountingMethod.THRES_BLOBDOG_BYSIZE:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.thres = seg_process.SimpleThreshold(.45)
                        self.blobdog = seg_process.BlobDog(
                            min_sigma=2.,
                            max_sigma=4.,
                            threshold=.1,
                            reduce=False
                        )
                        self.bs_lc_to_os = seg_process.BinaryAndCentroidListToInstance(maxSplit=int(1e6))
                        self.os_to_cc = sp_os_to_cc.CountOSBySize(
                            size_threshold=200.,
                            volume_weight=5.15e-3,
                            border_params=(3., -.5, 2.3),
                            min_size=8,
                            reduce=False
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        mask = await self.thres.forward(
                            im, cptr=cdir.cache(cid='thres'), viewer_args=viewer_args)
                        blobdog: NDBlock = await self.blobdog.forward(
                            im, cptr=cdir.cache(cid='blobdog'), viewer_args=viewer_args)
                        os = await self.bs_lc_to_os.forward(
                            mask, blobdog, cptr=cdir.cache(cid='os'), viewer_args=viewer_args)
                        cc = await self.os_to_cc.forward(
                            os, cptr=cdir.cache(cid='count_by_size'), viewer_args=viewer_args)
                        return cc

            case CountingMethod.GLOBAL_LABEL:
                class Pipeline(PipelineSuperclass):
                    def __init__(self):
                        super().__init__()
                        self.in_to_bs = seg_process.SimpleThreshold(.45)
                        self.bs_to_os = sp_bs_to_os.DirectBSToOS(is_global=True)
                        self.os_to_cc = sp_os_to_cc.CountOSBySize(
                            size_threshold=200.,
                            volume_weight=5.15e-3,
                            border_params=(3., -.5, 2.3),
                            min_size=8,
                            reduce=False,
                            is_global=True
                        )

                    async def forward(self, im, cptr, viewer_args: dict = None):
                        cdir = cptr.subdir()
                        bs = await self.in_to_bs.forward(im, cptr=cdir.cache(cid='in_to_bs'), viewer_args=viewer_args)
                        os = await self.bs_to_os.forward(bs, cptr=cdir.cache(cid='bs_to_os'), viewer_args=viewer_args)
                        cc = await self.os_to_cc.forward(os, cptr=cdir.cache(cid='os_to_cc'), viewer_args=viewer_args)
                        return cc

        return Pipeline

    if LOCAL_TESTING and not USE_GCS:
        TMP_PATH = "C:/ProgrammingTools/ComputerVision/RobartsResearch/data/lightsheet/tmp"
        CACHE_DIR_PATH = f'{TMP_PATH}/CacheDirectory'
        ORIG_IM_PATH = f'{TMP_PATH}/ch2_64_stack.zip'
        NEG_MASK_PATH = f'{TMP_PATH}/ch2_64_stack_4d_maskv4.tiff'
    elif LOCAL_TESTING and USE_GCS:
        TMP_PATH = 'gcs://khanlab-scratch/tmp'
        CACHE_DIR_PATH = f'{TMP_PATH}/CacheDirectory'
        ORIG_IM_PATH = f'gcs://khanlab-scratch/tmp/ch2_64_stack'
        NEG_MASK_PATH = f'gcs://khanlab-scratch/tmp/ch2_64_stack_4d_maskv4.tiff'
    else:
        TMP_PATH = 'gcs://khanlab-scratch/tmp'
        CACHE_DIR_PATH = f'{TMP_PATH}/CacheDirectory'
        ORIG_IM_PATH = f'gcs://khanlab-scratch/tmp/ch2_64_stack'
        NEG_MASK_PATH = f'gcs://khanlab-scratch/tmp/ch2_64_stack_4d_maskv4.tiff'

    import cvpl_tools.im.algs.dask_ndinterp as dask_ndinterp
    import napari
    import tifffile
    import coiled

    logfile_stdout = open('log_stdout.txt', mode='w')
    logfile_stderr = open('log_stderr.txt', mode='w')
    sys.stdout = imfs.MultiOutputStream(sys.stdout, logfile_stdout)
    sys.stderr = imfs.MultiOutputStream(sys.stderr, logfile_stderr)

    if True and RDirFileSystem(CACHE_DIR_PATH).exists(''):
        RDirFileSystem(CACHE_DIR_PATH).rm('', recursive=True)

    with imfs.CacheRootDirectory(
            CACHE_DIR_PATH,
            remove_when_done=False,
            read_if_exists=True,
    ) as temp_directory:
        import threading
        print(f'tid:::: {threading.get_ident()}')

        np.set_printoptions(precision=1)

        cur_im = da.from_zarr(cvpl_ome_zarr_io.load_zarr_group_from_path(
            path=ORIG_IM_PATH, mode='r', level=0
        ))[COMP_SLI] / 1000
        assert cur_im.ndim == 3
        print(f'imshape={cur_im.shape}')
        cur_im = cur_im.rechunk(chunks=(64, 64, 64))

        viewer = None  # napari.Viewer(ndisplay=2)
        storage_options = dict(
            dimension_separator='/',
            preferred_chunksize=(1, 4096, 4096),
            multiscale=4 if viewer else 0,
            compressor=numcodecs.Blosc(cname='lz4', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
        )
        viewer_args = dict(
            viewer=viewer,
            display_points=True,
            display_checkerboard=True,
            storage_options=storage_options
        )

        async def compute_masking():
            with RDirFileSystem(NEG_MASK_PATH).open('', mode='rb') as infile:
                neg_mask = tifffile.imread(infile)
            neg_mask = da.from_array(neg_mask, chunks=(64, 64, 64))
            neg_mask = dask_ndinterp.scale_nearest(neg_mask,
                                                   scale=1, output_shape=cur_im.shape, output_chunks=(64, 64, 64))[COMP_SLI]
            neg_mask = await temp_directory.cache_im(fn=lambda: neg_mask,
                                                     cid='neg_mask_upsampling',
                                                     viewer_args=viewer_args | dict(is_label=True))
            return cur_im * (1 - neg_mask)

        layer_args = dict(name='im', colormap='bop blue')
        cur_im = await temp_directory.cache_im(compute_masking(),
                                               cid='input_im',
                                               cache_level=1,
                                               viewer_args=viewer_args | dict(layer_args=layer_args))

        counted = np.zeros((len(CountingMethod),), dtype=np.int32)

        X = []
        for i in range(6, 7):
            item = CountingMethod(i)
            alg = get_pipeline(item)()

            import time
            stime = time.time()
            # print(f'starting, stime: {stime}')
            ncell: NDBlock = await alg.forward(
                cur_im,
                cptr=temp_directory.cache(cid=item.name),
                viewer_args=viewer_args
            )
            midtime = time.time()
            print(f'forward elapsed: {midtime - stime}')
            ncell_list = await ncell.reduce(force_numpy=True)

            kde_in = ncell_list.flatten()
            kde_in = kde_in[kde_in >= 1]  # at least one cell in the entire ]block
            kde_in = np.log2(kde_in + 1)
            X.append(kde_in)
            print(f'ending  elapsed: {time.time() - midtime}')
            cnt = ncell_list.sum().item()
            print(f'{item.name}:', cnt)
            counted[i] = cnt

    return counted


if __name__ == '__main__':
    import coiled
    # from coiled.credentials.google import send_application_default_credentials
    import time

    if LOCAL_TESTING:
        from distributed import Client

        cluster = None
        client = Client(threads_per_worker=12, n_workers=1)
    else:
        nworkers = 3
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
    print(client.run(main, workers=[workers[0]]))

    client.close()
    if cluster is not None:
        cluster.close()
