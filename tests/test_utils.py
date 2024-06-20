import numpy as np
import pytest

from znh5md.utils import concatenate_varying_shape_arrays, split_varying_shape_array


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
