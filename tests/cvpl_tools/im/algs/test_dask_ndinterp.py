import dask.array as da
import numpy as np
import numpy.typing as npt
from cvpl_tools.im.algs.dask_ndinterp import affine_transform_nearest, scale_nearest, measure_block_reduce


class DaskNDInterpTest:
    def test_integer_scaling(self):
        # when downsample or upsample by N times, the result should be exactly like repeating neighboring elements
        # by N times; this only applies to 'nearest' mode which is used by default.
        arr: npt.NDArray[np.int32] = (np.arange(4) + 1).reshape((2, 2)).astype(np.int32)
        arr: da.Array = da.from_array(arr, chunks=(2, 2))
        arr2 = scale_nearest(arr, 2, (4, 4), output_chunks=(2, 2))
        assert arr2.chunksize == (2, 2)
        np.testing.assert_equal(arr2,
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

    def test_mirroring(self):
        arr: npt.NDArray[np.int32] = np.array(
            ((1, 2),
             (3, 4)), dtype=np.int32
        )
        arr: da.Array = da.from_array(arr, chunks=(2, 2))

        # transpose (mirror along 135 degrees line)
        print('transpose')
        matrix = np.array(
            ((0, 1, 0),
             (1, 0, 0),
             (0, 0, 1)), dtype=np.float32
        )
        arr2 = affine_transform_nearest(arr, matrix, output_shape=(2, 2), output_chunks=(2, 2))
        np.testing.assert_equal(arr2,
                                np.array(
                                    ((1, 3),
                                     (2, 4)), dtype=np.int32
                                ))

        # horizontal mirroring
        print('horizontal mirroring')
        matrix = np.array(
            ((1, 0, 0),
             (0, -1, 2),
             (0, 0, 1)), dtype=np.float32
        )
        arr3 = affine_transform_nearest(arr, matrix, output_shape=(2, 2), output_chunks=(2, 2))
        np.testing.assert_equal(arr3,
                                np.array(
                                    ((2, 1),
                                     (4, 3)), dtype=np.int32
                                ))

    def test_block_reduce(self):
        im = da.from_array(np.zeros((3, 2), dtype=np.int32))
        im[1, 1] = 2
        im[2, 0] = 1
        im = measure_block_reduce(im, block_size=(2, 3), reduce_fn=np.max)

        # chunk would be chosen to fit IDEAL_SIZE which covers the entire image in this case
        assert im.compute().size == 0, im.compute()

        im = da.from_array(np.ones((3, 3, 6), dtype=np.float32))
        im = measure_block_reduce(im, block_size=(2, 3, 3), reduce_fn=np.min)
        np.testing.assert_equal(im.compute(), np.array((((1, 1),),), dtype=np.float32))

        im = da.from_array(np.array((1, 2, 3, 4, 5), dtype=np.float32))
        im = measure_block_reduce(im, block_size=(2,), reduce_fn=np.mean)
        np.testing.assert_almost_equal(im.compute(), np.array((1.5, 3.5), dtype=np.float32))
