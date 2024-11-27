import dataclasses

import numpy as np
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import typing as t

# NumPy does not yet support type hints with shapes
# https://github.com/numpy/numpy/issues/16544


# TODO: convert this into a mutable sequence?

@dataclasses.dataclass
class Frames:
    """Dataclass for Atoms object serialization."""

    positions: np.ndarray  # (N, )
    numbers: np.ndarray  # (N, )
    pbc: np.ndarray  # (N, 3)
    cell: np.ndarray  # (N, 3, 3)
    arrays: dict[str, np.ndarray]  # (N, ...), where ... is the shape of the array
    info: dict[str, np.ndarray]  # (N, ...), where ... is the shape of the array
    calc: dict[str, np.ndarray]  # (N, ...), where ... is the shape of the array


    def __iter__(self) -> t.Iterator[ase.Atoms]:
        """Iterate over the frames."""
        for idx in range(len(self.positions)):
            atoms = ase.Atoms(numbers=self.numbers[idx], positions=self.positions[idx], cell=self.cell[idx], pbc=self.pbc[idx])
            # TODO: remove MISSING values!
            for key in self.arrays:
                if self.arrays[key][idx] is not MISSING:
                    atoms.arrays[key] = self.arrays[key][idx]
            for key in self.info:
                if self.info[key][idx] is not MISSING:
                    atoms.info[key] = self.info[key][idx]
            for key in self.calc:
                if self.calc[key][idx] is not MISSING:
                    if atoms.calc is None:
                        atoms.calc = SinglePointCalculator(atoms)
                    atoms.calc.results[key] = self.calc[key][idx]
            yield atoms


class _MISSING:
    pass


MISSING = _MISSING()
