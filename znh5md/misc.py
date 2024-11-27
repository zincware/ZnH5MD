import numpy as np

def concatenate_varying_shape_arrays(values: list, fillvalue: str | int | float | bool) -> np.ndarray:
    """
    Concatenates arrays of varying shapes into a single array, 
    padding smaller arrays with a specified fillvalue.

    Args:
        values (list): List of numpy arrays with varying shapes.
        fillvalue (str | int | float | bool): Value used to fill missing entries.
            The fillvalue also determines the dtype of the output array.

    Returns:
        np.ndarray: A new array containing the input arrays, padded to match the maximum shape.
    """
    # Determine the dtype from the fillvalue
    dtype = np.array(fillvalue).dtype

    # Determine the maximum shape along all dimensions
    maxshape = list(values[0].shape)
    for value in values[1:]:
        maxshape = [max(a, b) for a, b in zip(maxshape, value.shape)]

    # Add the batch dimension
    maxshape = (len(values), *maxshape)

    # Create an array filled with the fillvalue
    dataset = np.full(maxshape, fillvalue, dtype=dtype)

    # Insert each value into the dataset
    for i, value in enumerate(values):
        # Create slices for each dimension of the current value
        slices = tuple(slice(0, dim) for dim in value.shape)
        dataset[(i,) + slices] = value

    return dataset


def decompose_varying_shape_arrays(dataset: np.ndarray, fillvalue: str | int | float | bool | np.ndarray) -> list:
    """
    Decomposes a concatenated array with padding into a list of original arrays.

    Args:
        dataset (np.ndarray): The concatenated array with padding.
        fillvalue (str | int | float | bool): Value used to fill missing entries in the original concatenation.

    Returns:
        list: List of numpy arrays with the padding removed.
    """
    decomposed = []

    for value in dataset:
        if np.isnan(fillvalue):
            decomposed.append(value[~np.isnan(value)])
        else:
            decomposed.append(value[value != fillvalue])

    return decomposed
