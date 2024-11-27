import ase
import numpy as np

from znh5md.serialization import MISSING, Frames


def encode(frames: list[ase.Atoms]) -> Frames:
    obj = Frames()
    obj.extend(frames)
    return obj
