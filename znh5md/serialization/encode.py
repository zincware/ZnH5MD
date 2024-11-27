import ase

from znh5md.serialization import Frames


def encode(frames: list[ase.Atoms]) -> Frames:
    obj = Frames()
    obj.extend(frames)
    return obj
