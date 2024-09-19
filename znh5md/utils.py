import ase
import h5py
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from znh5md.config import NUMERIC_FILL_VALUE, STRING_FILL_VALUE

NUMPY_STRING_DTYPE = np.dtype("S512")


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
    if np.shape(arrays[0]) == ():
        return np.array(arrays).flatten()
    if len(np.shape(arrays[0])) == 0:
        return np.array(arrays)
    max_n_particles = max(x.shape[0] for x in arrays)
    dimensions = arrays[0].shape[1:]

    if arrays[0].dtype == NUMPY_STRING_DTYPE:
        result = np.full(
            (len(arrays), max_n_particles), STRING_FILL_VALUE, dtype=NUMPY_STRING_DTYPE
        )
        for i, x in enumerate(arrays):
            result[i, : x.shape[0]] = x
    else:
        result = np.full(
            (len(arrays), max_n_particles, *dimensions), NUMERIC_FILL_VALUE
        )
        for i, x in enumerate(arrays):
            result[i, : x.shape[0], ...] = x
    return result


def remove_nan_rows(array: np.ndarray) -> np.ndarray | object:
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
    >>> remove_nan_rows(np.nan)
    None
    >>> remove_nan_rows(np.array(1))
    1

    """        
    if isinstance(array, np.ndarray) and array.dtype == object:
        # TODO: test if this has been added in a second append!
        data = np.array([x.decode() for x in array if x != STRING_FILL_VALUE])
        if len(data) == 0:
            return None
        return data
    if np.isnan(array).all():
        return None
    if len(np.shape(array)) == 0:
        return array if not np.isnan(array) else None
    return array[~np.isnan(array).all(axis=tuple(range(1, array.ndim)))]


def fill_dataset(dataset, new_data, shift=0):
    # shift is applied along axis 0:
    #  a dataset might not have been extenden in the last step because no data was added.
    #  with the shfit we ensure that the missing data along axis 0 is filled with np.nan
    # Axis 0 is the configuration axis
    # Axis 1 is the number of particles axis
    # all following axis are optional, e.g 2 can be (x, y, z) or 1 can already be energy

    old_shape = dataset.shape
    new_shape = new_data.shape

    fill_value = get_h5py_fill_value(new_data)

    if len(old_shape) == 1 and len(new_shape) == 1:
        dataset.resize((old_shape[0] + new_shape[0] + shift,))
        if shift > 0:
            dataset[old_shape[0] :] = fill_value
        dataset[old_shape[0] + shift :] = new_data
        return

    # Determine the new shape of the dataset
    max_shape = (
        old_shape[0] + new_shape[0],
        max(old_shape[1], new_shape[1]),
        *old_shape[2:],
    )
    # raise ValueError(f"{old_shape=}, {new_shape=}, {max_shape=}")

    # Resize the dataset to the new shape
    dataset.resize(max_shape)

    # Fill the new columns of the existing data with np.nan
    if old_shape[1] < max_shape[1]:
        dataset[:, old_shape[1] :] = fill_value

    # Fill the new data rows with np.nan if necessary
    if new_shape[1] < max_shape[1]:
        padded_new_data = np.full((new_shape[0], max_shape[1], *old_shape[2:]), fill_value)
        padded_new_data[:, : new_shape[1]] = new_data
    else:
        padded_new_data = new_data

    # Append the new data to the dataset
    dataset[old_shape[0] :] = padded_new_data


def handle_info_special_cases(info_data: dict) -> dict:
    keys_to_remove = []
    for key, value in info_data.items():
        if isinstance(value, bytes):
            # string types
            if value == STRING_FILL_VALUE:
                keys_to_remove.append(key)
            else:
                info_data[key] = value.decode("utf-8")
        elif isinstance(value, dict):
            # json / dict types
            info_data[key] = value
        else:
            # float / int / bool types
            info_data[key] = remove_nan_rows(value)
    for key in keys_to_remove:
        info_data.pop(key)
    return info_data


def build_atoms(args) -> ase.Atoms:
    (
        atomic_numbers,
        positions,
        velocities,
        cell,
        pbc,
        calc_data,
        info_data,
        arrays_data,
    ) = args
    atomic_numbers = remove_nan_rows(atomic_numbers)
    if positions is not None:
        positions = remove_nan_rows(positions)
    if velocities is not None:
        velocities = remove_nan_rows(velocities)
    if calc_data is not None:
        calc_data = {key: remove_nan_rows(value) for key, value in calc_data.items()}
    if arrays_data is not None:
        arrays_data = {
            key: remove_nan_rows(value) for key, value in arrays_data.items()
        }

    if info_data is not None:  # we don't need this check?
        info_data = handle_info_special_cases(info_data)

    # TODO: remove non-existing values (using sentinels!)
    # TODO: write check to ensure None will not be removed!
    for key in list(info_data):
        if info_data[key] is None:
            info_data.pop(key)
    for key in list(arrays_data):
        if arrays_data[key] is None:
            arrays_data.pop(key)

    atoms = ase.Atoms(
        symbols=atomic_numbers,
        positions=positions,
        velocities=velocities,
        pbc=pbc,
        cell=cell,
    )
    atoms.arrays.update(arrays_data)
    atoms.info.update(info_data)

    if calc_data is not None:
        if len(calc_data) > 0:
            if not all(val is None for val in calc_data.values()):
                atoms.calc = SinglePointCalculator(atoms=atoms)
                atoms.calc.results = calc_data

    return atoms


def build_structures(
    cell,
    pbc,
    arrays_data,
    calc_data,
    info_data,
) -> list[ase.Atoms]:
    structures = []

    positions = arrays_data.pop("positions", None)
    velocities = arrays_data.pop("velocity", None)
    atomic_numbers = arrays_data.pop("species")
    if atomic_numbers is not None:
        # could use ThreadPoolExecutor here
        # but there is no performance gain
        for idx in range(len(atomic_numbers)):
            args = (
                atomic_numbers[idx],
                positions[idx] if positions is not None else None,
                velocities[idx] if velocities is not None else None,
                cell[idx] if cell is not None else None,
                pbc[idx] if isinstance(pbc[0], np.ndarray) else pbc,
                {
                    key: value[idx]
                    for key, value in calc_data.items()
                    if len(value) > idx
                },
                {key: value[idx] for key, value in info_data.items()},
                {key: value[idx] for key, value in arrays_data.items()},
            )
            structures.append(build_atoms(args))
    return structures


def get_h5py_dtype(data: np.ndarray):
    if data.dtype == NUMPY_STRING_DTYPE:
        return h5py.string_dtype(encoding="utf-8")
    else:
        return data.dtype


def get_h5py_fill_value(data: np.ndarray):
    if data.dtype == NUMPY_STRING_DTYPE:
        return STRING_FILL_VALUE
    else:
        return NUMERIC_FILL_VALUE
