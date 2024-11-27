import dataclasses
import numpy as np
import numpy.typing as nptype
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import typing as t

CONTENT_TYPE = dict[str, np.ndarray | dict | float | int | str | bool]

def concatenate(a: np.ndarray, b: np.ndarray | dict | float | int | str | bool) -> np.ndarray:
    # edge case for float
    if isinstance(b, (float, int)):
        return np.array(list(a) + [b])


    if len(a.shape) != 1 or len(b.shape) != 1:
        return np.array(list(a) + list(b), dtype=object)
    else:
        return np.concatenate([a, b], axis=0, dtype=object)
    
# TODO TEST
# print(concatenate(np.random.rand(1, 100, 3), np.random.rand(1, 100, 3)).shape)
# print(concatenate(np.random.rand(1, 100, 3), np.random.rand(1, 90, 3)).shape)
# print(concatenate(np.random.rand(1, 100, 3), np.random.rand(2, 90, 3)).shape)
# print(concatenate(np.random.rand(2, 100, 3), np.random.rand(1, 100, 3)).shape)
# print(concatenate(np.random.rand(2, 100, 3), np.random.rand(1, 90, 3)).shape)
# print(concatenate(np.random.rand(2, 100, 3), np.random.rand(2, 90, 3)).shape)

# a  = concatenate(np.random.rand(1, 100, 3), np.random.rand(1, 100, 3))
# concatenate(a, np.random.rand(1, 80, 3)).shape


def process_category(target: dict[str, nptype.NDArray[np.object_]], content: CONTENT_TYPE, index: int) -> None:
    """
    Process a category (arrays, info, calc) for a single atoms object, ensuring
    that keys are added and missing values are backfilled.

    Parameters:
        target (dict): The main dictionary storing data for the category.
        content (dict): The data from the current atoms object (arrays, info, or calc).
        index (int): The index of the current atoms object in the trajectory used for backfilling.
    """
    seen = set(content.keys())
    unseen = set(target.keys()) - seen

    for key in content:
        if key not in target:
            # Backfill existing entries with MISSING for the new key
            if index > 0:
                target[key] = np.array([MISSING] * index + [content[key]], dtype=object)
            else:    
                target[key] = np.array([content[key]])
        else:
            # Add the new data to the existing key
            target[key] = concatenate(target[key], np.array([content[key]]))

    for key in unseen:
        # Backfill missing entries with MISSING for the unseen key
        target[key] = concatenate(target[key], np.array([MISSING], dtype=object))


@dataclasses.dataclass
class Frames:
    """Dataclass for Atoms object serialization."""

    positions: nptype.NDArray[np.object_] | None = None  # (N, )
    numbers: nptype.NDArray[np.object_] | None = None  # (N, )
    pbc: nptype.NDArray[np.bool_] | None = None  # (N, 3)
    cell: nptype.NDArray[np.float_] | None = None  # (N, 3, 3)
    arrays: dict[str, nptype.NDArray[np.object_]] = dataclasses.field(
        default_factory=dict
    )  # (N, m, ...) where m is the number of atoms in the frame
    info: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )  # (N, ...) can either be object or if consistent shape the respective dtype
    calc: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )  # (N, ...) can either be object or if consistent shape the respective dtype

    @classmethod
    def from_ase(cls, frames: t.Iterable[ase.Atoms]) -> "Frames":
        """Create a Frames object from a sequence of ASE Atoms objects."""
        obj = cls()
        obj.extend(frames)
        return obj

    def __iter__(self) -> t.Iterator[ase.Atoms]:
        """Iterate over the frames."""
        if self.positions is None:
            return iter([])
        for idx in range(len(self.positions)):
            try:
                yield self[idx]
            except IndexError:
                return
            
    def __len__(self) -> int:
        """Return the number of frames."""
        if self.positions is None:
            return 0
        return len(self.positions)

    def __getitem__(self, idx: int) -> ase.Atoms:
        """Return a single frame."""
        if self.positions is None:
            raise IndexError("No frames in Frames object.")
        atoms = ase.Atoms(
            numbers=self.numbers[idx],
            positions=self.positions[idx],
            cell=self.cell[idx],
            pbc=self.pbc[idx],
        )
        for key in self.arrays:
            if not isinstance(self.arrays[key][idx], _MISSING):
                atoms.arrays[key] = self.arrays[key][idx]
        for key in self.info:
            if not isinstance(self.info[key][idx], _MISSING):
                atoms.info[key] = self.info[key][idx]
        for key in self.calc:
            if not isinstance(self.calc[key][idx], _MISSING):
                if atoms.calc is None:
                    atoms.calc = SinglePointCalculator(atoms)
                atoms.calc.results[key] = self.calc[key][idx]

        return atoms

    def append(self, atoms: ase.Atoms) -> None:
        """Append a frame to the frames."""
        self.extend([atoms])

    def extend(self, frames: t.Iterable[ase.Atoms]) -> None:
        """Extend the frames with a sequence of frames."""
        positions = np.array([atoms.positions for atoms in frames], dtype=object)
        numbers = np.array([atoms.numbers for atoms in frames], dtype=object)
        pbc = np.array([atoms.pbc for atoms in frames], dtype=object)
        cell = np.array([atoms.cell.array for atoms in frames], dtype=object)

        if self.positions is None:
            # Initialize arrays
            self.positions = positions
            self.numbers = numbers
            self.pbc = pbc
            self.cell = cell
            start_idx = 0
        else:
            # Concatenate existing data with new data
            start_idx = len(self.positions)
            self.positions = concatenate(self.positions, positions)
            self.numbers = concatenate(self.numbers, numbers)
            self.pbc = np.concatenate([self.pbc, pbc])
            self.cell = np.concatenate([self.cell, cell])

        for idx, atoms in enumerate(frames, start=start_idx):
            # Process arrays
            atoms_arrays = {
                key: value
                for key, value in atoms.arrays.items()
                if key not in ["positions", "numbers"]
            }
            process_category(self.arrays, atoms_arrays, idx)

            # Process info
            process_category(self.info, atoms.info, idx)

            # Process calc
            if atoms.calc is not None:
                process_category(self.calc, atoms.calc.results, idx)
            elif len(self.calc) != 0:
                process_category(self.calc, {}, idx)

class _MISSING:
    """Sentinel value for missing entries."""

    pass


MISSING = _MISSING()