import numpy as np
import cvpl_tools.im.process.any_to_any as ata
import cvpl_tools.im.fs as imfs

if __name__ == '__main__':  # Only for main thread, worker threads will not run this
    TMP_PATH = "C:/ProgrammingTools/ComputerVision/RobartsResearch/data/lightsheet/tmp"
    import dask
    import dask.array as da
    from dask.distributed import Client
    import napari
    import tifffile

    with (dask.config.set({'temporary_directory': TMP_PATH}),
          imfs.CacheDirectory(
              f'{TMP_PATH}/CacheDirectoryBlockwise',
              remove_when_done=False,
              read_if_exists=True) as temp_directory):
        client = Client(threads_per_worker=12, n_workers=1)

        im = da.from_array(tifffile.imread(f'{TMP_PATH}/ch2_64_stack_4d_maskv4.tiff')).rechunk((64, 64, 64))

        down = ata.DownsamplingByIntFactor(2)
        up = ata.UpsamplingByIntFactor(2)
        down.set_tmpdir(tmpdir=temp_directory)
        up.set_tmpdir(tmpdir=temp_directory)
        viewer = napari.Viewer()
        viewer_args = dict(
            viewer=viewer,
            preferred_chunksize=(1, 4096, 4096),
            multiscale=4 if viewer else 0,
            display_points=False,
            display_checkerboard=False,
        )

        down_im = down.forward(im, cid='down', viewer_args=viewer_args | dict(is_label=True))
        up_im = up.forward(im, cid='up', viewer_args=viewer_args | dict(is_label=True))

        client.close()
        viewer.show(block=True)

