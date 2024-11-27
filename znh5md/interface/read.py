import typing as t

import ase
import numpy as np

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def getitem(
    self: "IO", index: int | np.int_ | slice | np.ndarray
) -> ase.Atoms | list[ase.Atoms]: ...
