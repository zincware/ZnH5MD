import typing as t

import ase

from znh5md.serialization import Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def extend(self: "IO", data: list[ase.Atoms]) -> None:
    frames = Frames.from_ase(data)
