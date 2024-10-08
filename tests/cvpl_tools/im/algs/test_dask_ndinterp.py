import dask.array as da
import numpy as np
import numpy.testing
import numpy.typing as npt
from cvpl_tools.im.algs.dask_ndinterp import affine_transform_nearest, scale_nearest
from scipy.ndimage import affine_transform


def test_integer_scaling():
    # when downsample or upsample by N times, the result should be exactly like repeating neighboring elements
    # by N times; this only applies to 'nearest' mode which is used by default.
    arr: npt.NDArray[np.int32] = (np.arange(4) + 1).reshape((2, 2)).astype(np.int32)
    arr: da.Array = da.from_array(arr, chunks=(2, 2))
    arr2 = scale_nearest(arr, 2, (4, 4), output_chunks=(2, 2))
    assert arr2.chunksize == (2, 2)
    numpy.testing.assert_equal(arr2,
                               np.array(
                                   ((1, 1, 2, 2),
                                    (1, 1, 2, 2),
                                    (3, 3, 4, 4),
                                    (3, 3, 4, 4)), dtype=np.int32
                               ))

    ones: npt.NDArray[np.int32] = np.zeros((3, 3, 3), dtype=np.int32)
    ones[:2, :2, :2] = 1
    for sz in (1 / 2, 2):
        for sy in (1 / 2, 2):
            for sx in (1 / 2, 2):
                scale = (sz, sy, sx)
                arr3 = scale_nearest(ones, scale, (5, 5, 5),
                                     output_chunks=(5, 5, 5)).compute()

                refs = tuple(int(s * 2) for s in scale)
                ref_slices = tuple(slice(0, r) for r in refs)
                ref_arr = np.zeros((5, 5, 5), dtype=np.int32)
                ref_arr[ref_slices] = 1
                np.testing.assert_equal(arr3, ref_arr)


if __name__ == '__main__':
    test_integer_scaling()
