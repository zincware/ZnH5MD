import numpy as np
import numpy.testing as npt
import pytest

from znh5md.misc import concatenate_varying_shape_arrays, decompose_varying_shape_arrays


@pytest.mark.parametrize(
    ("inp", "expected", "fillvalue"),
    [
        # 0D examples
        ([np.array(1), np.array(2)], np.array([1, 2]), 0),
        # 1D examples
        (
            [np.array([1, 2]), np.array([3, 4, 5])],
            np.array([[1, 2, np.nan], [3, 4, 5]]),
            np.nan,
        ),
        ([np.array([1, 2]), np.array([3, 4])], np.array([[1, 2], [3, 4]]), 0),
        (
            [np.array([1, 2, 3]), np.array([4, 5])],
            np.array([[1, 2, 3], [4, 5, np.nan]]),
            np.nan,
        ),
        # Higher-dimensional examples
        (
            [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])],
            np.array([[[1, 2], [3, 4]], [[5, 6], [np.nan, np.nan]]]),
            np.nan,
        ),
        (
            [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            0,
        ),
    ],
)
def test_concatenate_varying_shape_arrays(inp, expected, fillvalue):
    result = concatenate_varying_shape_arrays(
        inp, fillvalue=fillvalue, dtype=np.float64
    )
    npt.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("inp", "shape"),
    [
        [[np.random.rand(3), np.random.rand(2)], (2, 3)],
        [[np.random.rand(10, 3), np.random.rand(20, 3)], (2, 20, 3)],
        [[np.random.rand(10, 10, 3), np.random.rand(20, 10, 3)], (2, 20, 10, 3)],
        [[np.random.rand(10, 10, 3), np.random.rand(10, 20, 3)], (2, 10, 20, 3)],
        [[np.random.rand(20, 10, 3), np.random.rand(10, 20, 3)], (2, 20, 20, 3)],
        [
            [
                np.random.rand(20, 10, 3),
                np.random.rand(10, 20, 3),
                np.random.rand(10, 20, 4),
            ],
            (3, 20, 20, 4),
        ],
    ],
)
def test_concatenate_varying_shape_arrays_concept(inp, shape):
    result = concatenate_varying_shape_arrays(inp, fillvalue=np.nan, dtype=np.float64)
    assert result.shape == shape


@pytest.mark.parametrize(
    "inp",
    [
        [np.random.rand(3), np.random.rand(2)],
        [np.random.rand(10, 3), np.random.rand(20, 3)],
        [np.random.rand(10, 10, 3), np.random.rand(20, 10, 3)],
        [np.random.rand(10, 10, 3), np.random.rand(10, 20, 3)],
        [np.random.rand(20, 10, 3), np.random.rand(10, 20, 3)],
        [
            np.random.rand(20, 10, 3),
            np.random.rand(10, 20, 3),
            np.random.rand(10, 20, 4),
        ],
    ],
)
@pytest.mark.parametrize("fillvalue", [np.nan, 0.0])
def test_decompose_varying_shape_arrays(inp, fillvalue):
    concatenated = concatenate_varying_shape_arrays(
        inp, fillvalue=fillvalue, dtype=np.float64
    )
    result = decompose_varying_shape_arrays(concatenated, fillvalue=fillvalue)
    assert len(result) == len(inp)
    for i, array in enumerate(result):
        assert array.shape == inp[i].shape
        npt.assert_array_equal(array, inp[i])
