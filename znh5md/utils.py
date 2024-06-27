import ase
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator


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
    if len(np.shape(arrays[0])) == 0:
        return np.array(arrays)
    max_n_particles = max(x.shape[0] for x in arrays)
    dimensions = arrays[0].shape[1:]

    result = np.full((len(arrays), max_n_particles, *dimensions), np.nan)
    for i, x in enumerate(arrays):
        result[i, : x.shape[0], ...] = x
    return result


def remove_nan_rows(array: np.ndarray) -> np.ndarray | None:
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
    if len(np.shape(array)) == 0:
        return array if not np.isnan(array) else None
    return array[~np.isnan(array).all(axis=tuple(range(1, array.ndim)))]


def split_varying_shape_array(array: np.ndarray) -> list[np.ndarray]:
    """Split a numpy array into a list of 1D arrays.

    NaN values are removed to yield arrays of varying lengths.

    Parameters
    ----------
        array (numpy.array): At least 2D numpy array.

    Returns
    -------
        list of numpy.array: A list of numpy arrays where
        each array corresponds to a row in the input array.

    Example:
    >>> split_varying_shape_array(np.array([[ 1.,  2., np.nan], [ 3.,  4.,  5.]]))
    [array([1., 2.]), array([3., 4., 5.])]

    """
    arrays = []
    for row in array:
        valid_elements = remove_nan_rows(row)
        arrays.append(valid_elements)

    return arrays


def fill_dataset(dataset, new_data):
    # Axis 0 is the configuration axis
    # Axis 1 is the number of particles axis
    # all following axis are optional, e.g 2 can be (x, y, z) or 1 can already be energy

    old_shape = dataset.shape
    new_shape = new_data.shape

    if len(old_shape) == 1 and len(new_shape) == 1:
        dataset.resize((old_shape[0] + new_shape[0],))
        dataset[old_shape[0] :] = new_data
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
        dataset[:, old_shape[1] :] = np.nan

    # Fill the new data rows with np.nan if necessary
    if new_shape[1] < max_shape[1]:
        padded_new_data = np.full((new_shape[0], max_shape[1], *old_shape[2:]), np.nan)
        padded_new_data[:, : new_shape[1]] = new_data
    else:
        padded_new_data = new_data

    # Append the new data to the dataset
    dataset[old_shape[0] :] = padded_new_data


def build_atoms(
    atomic_numbers, positions, velocities, cell, pbc, calc_data, info_data, arrays_data
) -> ase.Atoms:
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
        for idx in range(len(atomic_numbers)):
            # ruff thinks, this is less complex than doing it in place ... ??
            atoms = build_atoms(
                atomic_numbers=remove_nan_rows(atomic_numbers[idx]),
                positions=remove_nan_rows(positions[idx])
                if positions is not None
                else None,
                velocities=remove_nan_rows(velocities[idx])
                if velocities is not None
                else None,
                cell=cell[idx] if cell is not None else None,
                pbc=pbc[idx] if isinstance(pbc[0], np.ndarray) else pbc,
                arrays_data={
                    k: remove_nan_rows(v[idx]) for k, v in arrays_data.items()
                },
                calc_data={k: remove_nan_rows(v[idx]) for k, v in calc_data.items()}
                if calc_data is not None
                else None,
                info_data={k: remove_nan_rows(v[idx]) for k, v in info_data.items()},
            )
            structures.append(atoms)
    return structures
