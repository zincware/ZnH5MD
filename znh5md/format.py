import numpy as np


def get_atomic_numbers(group, name, index) -> np.ndarray:
    return group[name]["species"]["value"][index]


def get_positions(group, name, index) -> np.ndarray:
    return group[name]["position"]["value"][index]


def get_box(group, name, index) -> np.ndarray:
    return group[name]["box"]["edges"]["value"][index]


def get_momenta(group, name, index) -> np.ndarray:
    return group[name]["momentum"]["value"][index]


def get_property(group, name, prop, index) -> np.ndarray:
    return group[name][prop]["value"][index]


def get_pbc(group, name, index) -> list[bool]:
    # TODO: support pbc on/off
    boundary = group[name]["box"].attrs["boundary"]
    return [True if x == b"periodic" else False for x in boundary]


CUSTOM_READER = {
    "species",
    "position",
    "box",
    "momentum",
}
