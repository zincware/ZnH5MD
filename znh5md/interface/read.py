import numpy as np
import ase

import typing as t

if t.TYPE_CHECKING:
  from znh5md.interface.io import IO

def getitem(self: "IO", index: int | np.int_| slice | np.ndarray) -> ase.Atoms| list[ase.Atoms]:
    ...