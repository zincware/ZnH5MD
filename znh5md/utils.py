import numpy as np


def concatenate_varying_shape_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate arrays of varying lengths into a numpy array.

    Gaps are filled with NaN values.

    Parameters
    ----------
        arrays (list of numpy.array): List of numpy arrays of varying lengths.

    Returns
    -------
        numpy.array: A numpy array where each row corresponds to an input array.

    Example:
    >>> concatenate_varying_shape_arrays([np.array([1, 2]), np.array([3, 4, 5])])
    array([[ 1.,  2., nan],
           [ 3.,  4.,  5.]])

    """
    max_n_particles = max(x.shape[0] for x in arrays)
    dimensions = arrays[0].shape[1:]

    result = np.full((len(arrays), max_n_particles, *dimensions), np.nan)
    for i, x in enumerate(arrays):
        result[i, : x.shape[0], ...] = x
    return result


def remove_nan_rows(array: np.ndarray) -> np.ndarray:
    """Remove rows with NaN values from a numpy array.

    Parameters
    ----------
        array (numpy.array): At least 2D numpy array.

    Returns
    -------
        numpy.array: A numpy array where rows with NaN values are removed.

    Example:
    >>> remove_nan_rows(np.array([[ 1.,  2., np.nan], [ 3.,  4.,  5.]]))
    array([[3., 4., 5.]])

    """
    return array[~np.isnan(array).all(axis=tuple(range(1, array.ndim)))]


def split_varying_shape_array(array: np.ndarray) -> list[np.ndarray]:
    """Split a numpy array into a list of 1D arrays.

    NaN values are removed to yield arrays of varying lengths.

    Parameters
    ----------
        array (numpy.array): At least 2D numpy array.

    Returns
    -------
        list of numpy.array: A list of numpy arrays where each array corresponds to a row in the input array.

    Example:
    >>> split_varying_shape_array(np.array([[ 1.,  2., np.nan], [ 3.,  4.,  5.]]))
    [array([1., 2.]), array([3., 4., 5.])]

    """
    arrays = []
    for row in array:
        valid_elements = remove_nan_rows(row)
        arrays.append(valid_elements)

    return arrays
