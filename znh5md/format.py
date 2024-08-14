import dataclasses
import enum
import json
from typing import Any, Dict, List, Optional, TypedDict, Union

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_properties

from .utils import NUMPY_STRING_DTYPE, concatenate_varying_shape_arrays


class ASEKeyMetaData(TypedDict):
    unit: Optional[str]
    calc: Optional[bool]


class CustomINFOData(str, enum.Enum):
    """Custom INFO data that is not stored in the particles group."""

    h5md_step = "step"
    h5md_time = "time"


UNIT_MAPPING = {
    "energy": "eV",
    "force": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
    "velocity": "Angstrom/fs",
    "position": "Angstrom",
    "time": "fs",
}


@dataclasses.dataclass
class ASEData:
    """A dataclass for storing ASE Atoms data."""

    cell: Optional[np.ndarray]
    pbc: np.ndarray
    observables: Dict[str, np.ndarray]
    particles: Dict[str, np.ndarray]
    metadata: Optional[Dict[str, ASEKeyMetaData]] = None
    time: Optional[np.ndarray] = None
    step: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        for key in UNIT_MAPPING:
            if key not in self.metadata:
                self.metadata[key] = {"unit": UNIT_MAPPING[key], "calc": False}

    def __len__(self):
        return len(self.particles["species"])


def get_property(group, name: str, prop: str, index) -> Optional[np.ndarray]:
    """Retrieve a property from an HDF5 group."""
    try:
        return group[name][prop]["value"][index]
    except KeyError:
        return None
    except OSError:
        # h5py raises an OSError if index == len(field)
        # but we need an IndexError
        raise IndexError(f"Index {index} out of bounds for property {prop}")


def get_atomic_numbers(group, name: str, index) -> Optional[np.ndarray]:
    """Retrieve atomic numbers from an HDF5 group."""
    return get_property(group, name, "species", index)


def get_positions(group, name: str, index) -> Optional[np.ndarray]:
    """Retrieve positions from an HDF5 group."""
    return get_property(group, name, "position", index)


def get_box(group, name: str, index) -> Optional[np.ndarray]:
    """Retrieve cell box dimensions from an HDF5 group."""
    return get_property(group, name, "box/edges", index)


def get_velocities(group, name: str, index) -> Optional[np.ndarray]:
    """Retrieve velocities from an HDF5 group."""
    return get_property(group, name, "velocity", index)


def get_pbc(group, name: str, index) -> np.ndarray:
    """Retrieve periodic boundary conditions from an HDF5 group."""
    if "pbc" in group[name]["box"]:
        return group[name]["box"]["pbc"]["value"][index]
    else:
        try:
            boundary = group[name]["box"].attrs["boundary"]
            return np.array([True if x == b"periodic" else False for x in boundary])
        except KeyError:
            return np.array([False, False, False])


def get_species_aux_data(
    group: h5py.Group, name: str, field: str, index: Union[int, list[int], slice]
) -> np.ndarray:
    """Helper function to retrieve data from an HDF5 group."""
    try:
        # Attempt to retrieve the data using the provided index
        data = group[name]["species"][field][index]
    except ValueError:
        # Backwards compatibility: Handle case where the data is stored as a scalar
        scalar_data = group[name]["species"][field][()]

        # Calculate the length of the indexed data without loading the actual data
        value_shape = group[name]["species"]["value"].shape
        if isinstance(index, slice):
            value_length = len(range(*index.indices(value_shape[0])))
            value_start = index.start if index.start is not None else 0
        elif isinstance(index, list):
            if not sorted(index) == index:
                raise ValueError("Indices must be sorted")
            value_length = len(index)
            value_start = index[0]
        elif isinstance(index, int):
            value_length = 1
            value_start = index
        else:
            raise TypeError("Unsupported index type")

        # data should be (start * scalar_data, end * scalar_data, step * scalar_data)
        data = np.arange(value_start, value_start + value_length) * scalar_data

    return data


def get_time(
    group: h5py.Group, name: str, index: Union[int, list[int], slice]
) -> np.ndarray:
    """Retrieve time from an HDF5 group."""
    return get_species_aux_data(group, name, "time", index)


def get_step(
    group: h5py.Group, name: str, index: Union[int, list[int], slice]
) -> np.ndarray:
    """Retrieve step from an HDF5 group."""
    return get_species_aux_data(group, name, "step", index)


ASE_TO_H5MD = {
    "numbers": "species",  # remove
    "positions": "position",  # remove
    "cell": "box",
}


def extract_atoms_data(atoms: Atoms, use_ase_calc: bool = True) -> ASEData:  # noqa: C901
    """
    Extract data from an ASE Atoms object and return an ASEData object.

    Args:
        atoms: ase.Atoms
            An ASE Atoms object containing the atomic structure.
        use_ase_calc: bool, optional
            Whether to include data from the ASE calculator.
            Defaults to True.

    Returns:
        ASEData:
            An object containing the extracted data, including particles'
            information, observables, metadata, time, and step.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    velocities = atoms.get_velocities() if "momenta" in atoms.arrays else None

    particles: Dict[str, Any] = {
        "species": atomic_numbers,
        "position": positions,
    }
    if velocities is not None:
        particles["velocity"] = velocities

    info_data: Dict[str, Any] = {}
    uses_calc: List[str] = []
    metadata = {}

    if use_ase_calc and atoms.calc is not None:
        for key, value in atoms.calc.results.items():
            key = "force" if key == "forces" else key
            uses_calc.append(key)
            value = np.array(value) if isinstance(value, (int, float, list)) else value

            if value.ndim > 1 and value.shape[0] == len(atomic_numbers):
                particles[key] = value
            else:
                info_data[key] = value

    for key, value in atoms.info.items():
        if use_ase_calc and key in all_properties:
            raise ValueError(f"Key {key} is reserved for ASE calculator results.")
        if key not in ASE_TO_H5MD and key not in CustomINFOData.__members__:
            if isinstance(value, str):
                if len(value) > NUMPY_STRING_DTYPE.itemsize:
                    raise ValueError(f"String {key} is too long to be stored.")
                info_data[key] = np.array(value, dtype=NUMPY_STRING_DTYPE)
            elif isinstance(value, dict):
                info_data[key] = np.array(json.dumps(value), dtype=NUMPY_STRING_DTYPE)
                metadata[key] = {"unit": None, "calc": False, "type": "json"}
            else:
                info_data[key] = value

    for key, value in atoms.arrays.items():
        if use_ase_calc and key in all_properties:
            raise ValueError(f"Key {key} is reserved for ASE calculator results.")
        if key not in ASE_TO_H5MD:
            particles[key] = value

    time: Optional[float] = atoms.info.get(CustomINFOData.h5md_time.name, None)
    step: Optional[int] = atoms.info.get(CustomINFOData.h5md_step.name, None)

    metadata.update(
        {key: {"unit": UNIT_MAPPING.get(key), "calc": True} for key in uses_calc}
    )

    return ASEData(
        cell=cell,
        pbc=pbc,
        observables=info_data,
        particles=particles,
        metadata=metadata,
        time=time,
        step=step,
    )


# TODO highlight that an additional dimension is added to ASEData here
def combine_asedata(data: List[ASEData]) -> ASEData:
    """Combine multiple ASEData objects into one."""
    cell = _combine_property([x.cell for x in data])
    pbc = np.array(
        [x.pbc if x.pbc is not None else [False, False, False] for x in data]
    )

    observables = _combine_dicts([x.observables for x in data])
    particles = _combine_dicts([x.particles for x in data])

    time_occurrences = sum([x.time is not None for x in data])
    step_occurrences = sum([x.step is not None for x in data])
    if time_occurrences == len(data):
        time = np.array([x.time for x in data])
    elif time_occurrences == 0:
        time = None
    else:
        raise ValueError("Time is not consistent across data objects")

    if step_occurrences == len(data):
        step = np.array([x.step for x in data])
    elif step_occurrences == 0:
        step = None
    else:
        raise ValueError("Step is not consistent across data objects")

    return ASEData(
        cell=cell,
        pbc=pbc,
        observables=observables,
        particles=particles,
        metadata=data[0].metadata,  # we assume they are all equal
        time=time,
        step=step,
    )


def _combine_property(properties: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Helper function to combine varying shape arrays."""
    if all(x is None for x in properties):
        return None
    return concatenate_varying_shape_arrays(
        [x if x is not None else np.array([]) for x in properties]
    )


def _combine_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Helper function to combine dictionaries containing numpy arrays."""
    combined = {}
    for key in dicts[0]:
        data = []
        for d in dicts:
            if key in d:
                data.append(d[key])
            else:
                dims = dicts[0][key].ndim
                # Create an array with the appropriate number of dimensions.
                if dims == 0:
                    # Handle the case where the number of dimensions is 0
                    data.append(np.NaN)
                else:
                    data.append(np.full_like(dicts[0][key], np.NaN))
        if data:
            combined[key] = concatenate_varying_shape_arrays(data)
        else:
            raise ValueError(f"Key {key} is missing in one of the data objects.")
    return combined
