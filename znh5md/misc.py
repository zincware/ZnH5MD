import contextlib
import os
import typing as t

import h5py
import numpy as np


class _MISSING:
    """Sentinel value for missing entries."""

    pass


MISSING = _MISSING()


def concatenate_varying_shape_arrays(
    values: list, fillvalue: str | int | float | bool, dtype
) -> np.ndarray:
    """
    Concatenates arrays of varying shapes into a single array,
    padding smaller arrays with a specified fillvalue.

    Args:
        values (list): List of numpy arrays with varying shapes.
        fillvalue (str | int | float | bool): Value used to fill missing entries.
            The fillvalue also determines the dtype of the output array.

    Returns:
        np.ndarray: A new array containing the input arrays,
        padded to match the maximum shape.
    """

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


def decompose_varying_shape_arrays(
    dataset: np.ndarray, fillvalue: str | int | float | bool | np.ndarray
) -> list:
    """
    Decomposes a concatenated array with padding into a list of original arrays.

    Parameters
    ----------
        dataset: np.ndarray
            The concatenated array with padding.
        fillvalue: (str | int | float | bool | np.ndarray)
            Value used to fill missing entries in the original concatenation.

    Returns
    -------
        list: List of numpy arrays with the padding removed.
    """
    decomposed = []
    is_nan = np.isnan(fillvalue) if isinstance(fillvalue, float) else None

    for value in dataset:
        slices = []
        # Collapse all other dimensions to find non-fillvalue
        # regions along the current axis
        if is_nan:
            mask = ~np.isnan(value)
        else:
            mask = value != fillvalue
        if not mask.any():
            # There is no data in "value", so we append MISSING
            decomposed.append(MISSING)
        else:
            for axis in range(value.ndim):
                # Sum along all dimensions except the current one
                axis_sum = mask.any(
                    axis=tuple(i for i in range(value.ndim) if i != axis)
                )
                end = len(axis_sum) - np.argmax(
                    axis_sum[::-1]
                )  # Last non-fillvalue index + 1
                slices.append(slice(0, end))

            # Use slices to extract the non-fillvalue portion of the array
            decomposed.append(value[tuple(slices)])

    return decomposed


@contextlib.contextmanager
def open_file(
    filename: str | os.PathLike | None, file_handle: h5py.File | None, **kwargs
) -> t.Generator[h5py.File, None, None]:
    if file_handle is not None:
        yield file_handle
    else:
        with h5py.File(filename, **kwargs) as f:
            yield f


def fill_dataset(dataset, new_data, shift, fill_value):
    # shift is applied along axis 0:
    #  a dataset might not have been extenden in the last step
    #  because no data was added.
    #  with the shift we ensure that the missing data along axis 0 is filled with np.nan
    # Axis 0 is the configuration axis
    # Axis 1 is the number of particles axis
    # all following axis are optional, e.g 2 can be (x, y, z) or 1 can already be energy

    old_shape = dataset.shape
    new_shape = new_data.shape

    if len(old_shape) == 1 and len(new_shape) == 1:
        dataset.resize((old_shape[0] + new_shape[0] + shift,))
        dataset[old_shape[0] + shift :] = new_data
        return

    # Determine the new shape of the dataset
    max_shape = (
        old_shape[0] + new_shape[0] + shift,
        max(old_shape[1], new_shape[1]),
        *old_shape[2:],
    )

    # Resize the dataset to the new shape
    dataset.resize(max_shape)

    # Fill the new data rows with np.nan if necessary
    if new_shape[1] < max_shape[1]:
        padded_new_data = np.full(
            (new_shape[0], max_shape[1], *old_shape[2:]), fill_value
        )
        padded_new_data[:, : new_shape[1]] = new_data
    else:
        padded_new_data = new_data

    dataset[old_shape[0] + shift :] = padded_new_data
