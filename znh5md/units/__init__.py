from enum import Enum


class H5MDUnits(str, Enum):
    energy = "eV"
    forces = "eV/Angstrom"
    stress = "eV/Angstrom^3"
    velocities = "Angstrom/fs"
    positions = "Angstrom"
    time = "fs"
    cell = "Angstrom"


def get_unit(key: str) -> str | None:
    try:
        return H5MDUnits[key].value
    except KeyError:
        return None
