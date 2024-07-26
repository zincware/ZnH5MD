import dataclasses
import enum
from typing import Dict, List, Optional, TypedDict, Union

import ase
import h5py
import numpy as np
from ase.calculators.calculator import all_properties

from .utils import concatenate_varying_shape_arrays


class ASEKeyMetaData(TypedDict):
    unit: Optional[str]
    calc: Optional[bool]


class CustomINFOData(str, enum.Enum):
    """Custom INFO data that is not stored in the particles group."""

    h5md_step = "step"
    h5md_time = "time"


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

        for name in all_properties:
            self.metadata[name] = {"unit": None, "calc": True}

        self.metadata.pop("forces")  # is called 'force' in h5md
        self.metadata["force"] = {"unit": "eV/Angstrom", "calc": True}

        self.metadata["energy"]["unit"] = "eV"
        self.metadata["velocity"] = {"unit": "Angstrom/fs", "calc": False}
        self.metadata["position"] = {"unit": "angstrom", "calc": False}

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


def extract_atoms_data(atoms: ase.Atoms) -> ASEData:
    """Extract data from an ASE Atoms object into an ASEData object."""
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    velocities = atoms.get_velocities() if "momenta" in atoms.arrays else None

    info_data = {}
    particles = {"species": atomic_numbers, "position": positions}
    if velocities is not None:
        particles["velocity"] = velocities
    # save keys gathered from the calculator
    uses_calc: list[str] = []

    if atoms.calc is not None:
        for key, result in atoms.calc.results.items():
            if key not in all_properties:
                uses_calc.append(key)
            value = (
                np.array(result) if isinstance(result, (int, float, list)) else result
            )
            if value.ndim > 1 and value.shape[0] == len(atomic_numbers):
                particles[key if key != "forces" else "force"] = value
            else:
                info_data[key] = value

    for key, value in atoms.info.items():
        if (
            key not in all_properties
            and key not in ASE_TO_H5MD
            and key not in CustomINFOData.__members__
        ):
            info_data[key] = value

    for key, value in atoms.arrays.items():
        if key not in all_properties and key not in ASE_TO_H5MD:
            particles[key] = value

    time = atoms.info.get(CustomINFOData.h5md_time.name, None)
    step = atoms.info.get(CustomINFOData.h5md_step.name, None)

    return ASEData(
        cell=cell,
        pbc=pbc,
        observables=info_data,
        particles=particles,
        metadata={key: {"unit": None, "calc": True} for key in uses_calc},
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
        combined[key] = concatenate_varying_shape_arrays(
            [
                d[key] if isinstance(d[key], np.ndarray) else np.array([d[key]])
                for d in dicts
            ]
        )
    return combined
