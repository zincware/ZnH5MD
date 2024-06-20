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
    pbc: list[bool]
    momenta: np.ndarray | None
    calc_data: dict[str, np.ndarray]
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


def get_momenta(group, name, index) -> np.ndarray | None:
    return get_property(group, name, "momentum", index)


def get_pbc(group, name, index) -> list[bool]:
    # TODO: support pbc on/off
    try:
        boundary = group[name]["box"].attrs["boundary"]
        return [True if x == b"periodic" else False for x in boundary]
    except KeyError:
        return [False, False, False]


# TODO: mapping from ASE to H5MD property names
ASE_TO_H5MD = bidict(
    {
        "numbers": "species",
        "positions": "position",
        "cell": "box",
        "momenta": "momentum",
    }
)


def extract_atoms_data(atoms: ase.Atoms) -> ASEData:
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    momenta = atoms.get_momenta()
    calc_data = {}
    info_data = {}
    arrays_data = {}
    if atoms.calc is not None:
        for key in atoms.calc.results:
            if key in all_properties:
                calc_data[key] = (
                    atoms.calc.results[key]
                    if isinstance(atoms.calc.results[key], np.ndarray)
                    else np.array([atoms.calc.results[key]])
                )
            elif key not in ASE_TO_H5MD:
                if len(atoms.calc.results[key]) == len(atomic_numbers):
                    arrays_data[key] = np.array([atoms.calc.results[key]])
                else:
                    info_data[key] = np.array([atoms.calc.results[key]])
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
        momenta=momenta,
        calc_data=calc_data,
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
    pbc = [x.pbc if x.pbc is not None else [False, False, False] for x in data]
    if all(x.momenta is None for x in data):
        momenta = None
    else:
        momenta = concatenate_varying_shape_arrays(
            [x.momenta if x.momenta is not None else np.array([]) for x in data]
        )
    calc_data = {
        key: concatenate_varying_shape_arrays([x.calc_data[key] for x in data])
        for key in data[0].calc_data
    }
    info_data = {
        key: concatenate_varying_shape_arrays([x.info_data[key] for x in data])
        for key in data[0].info_data
    }
    arrays_data = {
        key: concatenate_varying_shape_arrays([x.arrays_data[key] for x in data])
        for key in data[0].arrays_data
    }
    return ASEData(
        atomic_numbers, positions, cell, pbc, momenta, calc_data, info_data, arrays_data
    )
