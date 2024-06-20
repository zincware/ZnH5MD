import dataclasses

import ase
import numpy as np
from ase.calculators.calculator import all_properties
from bidict import bidict

from .utils import concatenate_varying_shape_arrays


@dataclasses.dataclass
class ASEData:
    atomic_numbers: np.ndarray | None
    positions: np.ndarray | None
    cell: np.ndarray | None
    pbc: np.ndarray
    velocities: np.ndarray | None
    info_data: dict[str, np.ndarray]
    arrays_data: dict[str, np.ndarray]


def get_property(group, name, prop, index) -> np.ndarray | None:
    try:
        return group[name][prop]["value"][index]
    except KeyError:
        return None


def get_atomic_numbers(group, name, index) -> np.ndarray | None:
    return get_property(group, name, "species", index)


def get_positions(group, name, index) -> np.ndarray | None:
    return get_property(group, name, "position", index)


def get_box(group, name, index) -> np.ndarray | None:
    return get_property(group, name, "box/edges", index)


def get_velocities(group, name, index) -> np.ndarray | None:
    return get_property(group, name, "velocity", index)


def get_pbc(group, name, index) -> np.ndarray:
    if "pbc" in group[name]["box"]:
        return group[name]["box"]["pbc"]["value"][index]
    else:
        try:
            boundary = group[name]["box"].attrs["boundary"]
            return np.array([True if x == b"periodic" else False for x in boundary])
        except KeyError:
            return np.array([False, False, False])


# TODO: mapping from ASE to H5MD property names
# TODO: this is currently used in two different ways:
# - exclude properties with custom readers
# - map ASE to H5MD property names and vice versa
ASE_TO_H5MD = bidict(
    {
        "numbers": "species",
        "positions": "position",
        "cell": "box",
    }
)


def extract_atoms_data(atoms: ase.Atoms) -> ASEData:
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    if "momenta" in atoms.arrays:
        velocities = atoms.get_velocities()
    else:
        velocities = None
    info_data = {}
    arrays_data = {}
    if atoms.calc is not None:
        for key in atoms.calc.results:
            if isinstance(atoms.calc.results[key], (int, float)):
                value = np.array([atoms.calc.results[key]])
            else:
                value = atoms.calc.results[key]
            # We check for all properties, because shape[0] can be
            # equal to the number of atoms so this makes it a bit safer.
            # if you encout any issues here, make sure that #atoms != len(property)
            if value.shape[0] == len(atomic_numbers) or key in all_properties:
                arrays_data[key if key != "forces" else "force"] = value
            else:
                info_data[key] = value
    for key in atoms.info:
        if key not in all_properties and key not in ASE_TO_H5MD:
            info_data[key] = atoms.info[key]

    for key in atoms.arrays:
        if key not in all_properties and key not in ASE_TO_H5MD:
            arrays_data[key] = atoms.arrays[key]

    return ASEData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
        velocities=velocities,
        info_data=info_data,
        arrays_data=arrays_data,
    )


def combine_asedata(data: list[ASEData]) -> ASEData:
    atomic_numbers = concatenate_varying_shape_arrays([x.atomic_numbers for x in data])
    if all(x.positions is None for x in data):
        positions = None
    else:
        positions = concatenate_varying_shape_arrays(
            [x.positions if x.positions is not None else np.array([]) for x in data]
        )
    if all(x.cell is None for x in data):
        cell = None
    else:
        cell = concatenate_varying_shape_arrays(
            [x.cell if x.cell is not None else np.array([]) for x in data]
        )
    pbc = np.array(
        [x.pbc if x.pbc is not None else [False, False, False] for x in data]
    )
    if all(x.velocities is None for x in data):
        velocities = None
    else:
        velocities = concatenate_varying_shape_arrays(
            [x.velocities if x.velocities is not None else np.array([]) for x in data]
        )
    info_data = {
        key: concatenate_varying_shape_arrays(
            [
                x.info_data[key]
                if isinstance(x.info_data[key], np.ndarray)
                else np.array([x.info_data[key]])
                for x in data
            ]
        )
        for key in data[0].info_data
    }
    arrays_data = {
        key: concatenate_varying_shape_arrays(
            [
                x.arrays_data[key]
                if isinstance(x.arrays_data[key], np.ndarray)
                else np.array([x.arrays_data[key]])
                for x in data
            ]
        )
        for key in data[0].arrays_data
    }
    return ASEData(
        atomic_numbers, positions, cell, pbc, velocities, info_data, arrays_data
    )
