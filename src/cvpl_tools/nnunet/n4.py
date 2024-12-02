import cvpl_tools.ome_zarr.io as ome_zarr_io
import dask.array as da


image_netloc = 'gcs://khanlab-scratch/tmp/CacheDirectoryBlaze_F1A1Te4Blaze/GLOBAL_LABEL/os/global_os/dask_im'
print(f'ome_io.load_dask_array_from_path from path {image_netloc}')
group = ome_zarr_io.load_zarr_group_from_path(image_netloc, mode='r')
print(ome_zarr_io.get_highest_downsample_level(group))
# arr = da.from_array(group['0'])
# print(arr.shape)