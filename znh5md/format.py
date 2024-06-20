import numpy as np
from bidict import bidict


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
