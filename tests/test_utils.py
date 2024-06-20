import h5py
import numpy as np
import pytest

from znh5md.utils import (
    concatenate_varying_shape_arrays,
    fill_dataset,
    split_varying_shape_array,
)


@pytest.mark.parametrize(
    ["a", "b", "c", "result"],
    [
        [
            np.array([1]),
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([[1, np.nan, np.nan], [1, 2, np.nan], [1, 2, 3]]),
        ],
        [
            np.array([[1, 2]]),
            np.array([[1, 2], [1, 2]]),
            np.array([[1, 2], [1, 2], [1, 2]]),
            np.array(
                [
                    [[1, 2], [np.nan, np.nan], [np.nan, np.nan]],
                    [[1, 2], [1, 2], [np.nan, np.nan]],
                    [[1, 2], [1, 2], [1, 2]],
                ]
            ),
        ],
    ],
)
def test_concatenate_split_varying_shape_arrays(a, b, c, result):
    assert np.array_equal(
        concatenate_varying_shape_arrays([a, b, c]), result, equal_nan=True
    )

    a1, a2, a3 = split_varying_shape_array(result)

    assert np.array_equal(a1, a)
    assert np.array_equal(a2, b)
    assert np.array_equal(a3, c)


def test_concatenate_split_varying_shape_arrays_empty():
    a = np.array([1, 2])
    b = np.array([])
    c = np.array([4, 5, 6])

    result = np.array([[1, 2, np.nan], [np.nan, np.nan, np.nan], [4, 5, 6]])
    assert np.array_equal(
        concatenate_varying_shape_arrays([a, b, c]), result, equal_nan=True
    )

    a, b, c = split_varying_shape_array(result)
    assert np.array_equal(a, np.array([1, 2]))
    assert np.array_equal(b, np.array([]))
    assert np.array_equal(c, np.array([4, 5, 6]))


@pytest.mark.parametrize(
    ["a", "b", "result"],
    [
        [
            np.array([1, 2]),
            np.array([3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
        ],
        [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6, 7]]),
            np.array([[1, 2, np.nan], [3, 4, np.nan], [5, 6, 7]]),
        ],
        [
            np.array([[[1, 2, 3]]]),
            np.array([[[1, 2, 3], [1, 2, 3]]]),
            np.array([[[1, 2, 3], [np.nan, np.nan, np.nan]], [[1, 2, 3], [1, 2, 3]]]),
        ],
    ],
)
def test_fill_dataset(tmp_path, a, b, result):
    maxshape = tuple([None] * a.ndim)

    with h5py.File(tmp_path / "test.h5", "w") as f:
        f.create_dataset(
            "test", data=a, chunks=True, maxshape=maxshape, dtype=np.float32
        )

    with h5py.File(tmp_path / "test.h5", "a") as f:
        fill_dataset(f["test"], b)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        ds = f["test"][:]

    assert np.array_equal(ds, result, equal_nan=True)
