import h5py
import numpy as np
import pytest

from znh5md.utils import (
    concatenate_varying_shape_arrays,
    fill_dataset,
    remove_nan_rows,
)


def test_concatenate_split_varying_shape_arrays_empty():
    a = np.array([1, 2])
    b = np.array([])
    c = np.array([4, 5, 6])

    result = np.array([[1, 2, np.nan], [np.nan, np.nan, np.nan], [4, 5, 6]])
    assert np.array_equal(
        concatenate_varying_shape_arrays([a, b, c]), result, equal_nan=True
    )


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


def test_remove_nan_rows():
    a = np.array([1, 2, np.nan, 4, 5])
    assert np.array_equal(remove_nan_rows(a), np.array([1, 2, 4, 5]))

    b = np.array([[1, 2], [np.nan, np.nan], [4, 5]])
    assert np.array_equal(remove_nan_rows(b), np.array([[1, 2], [4, 5]]))

    c = np.array(5)

    assert np.array_equal(remove_nan_rows(c), 5)
    assert remove_nan_rows(np.nan) is None


def test_build_atoms():
    pass


def test_handle_info_special_cases():
    pass


def test_build_structures():
    pass
