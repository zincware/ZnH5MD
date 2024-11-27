import dataclasses
import typing as t

import ase
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

CONTENT_TYPE = dict[str, np.ndarray | dict | float | int | str | bool]

def process_category(
    target: dict[str, list], content: CONTENT_TYPE, index: int
) -> None:
    """
    Process a category (arrays, info, calc) for a single atoms object, ensuring
    that keys are added and missing values are backfilled.

    Parameters
    ----------
    target : dict
        The target dictionary to update.
    content : dict
        The content to add to the target dictionary.
    index : int
        The index of the current frame.
    """
    seen = set(content.keys())
    unseen = set(target.keys()) - seen

    for key in content:
        if key not in target:
            # Backfill existing entries with MISSING for the new key
            target[key] = [MISSING] * index + [content[key]]
        else:
            # Add the new data to the existing key
            target[key].append(content[key])

    for key in unseen:
        # Backfill missing entries with MISSING for the unseen key
        target[key].append(MISSING)

# TODO: provide a reference for each, for the later padding to write to h5

@dataclasses.dataclass(repr=False)
class Frames:
    """Dataclass for Atoms object serialization."""

    positions: list = dataclasses.field(default_factory=list)
    numbers: list = dataclasses.field(default_factory=list)
    pbc: list = dataclasses.field(default_factory=list)
    cell: list = dataclasses.field(default_factory=list)
    arrays: dict[str, list] = dataclasses.field(
        default_factory=dict
    )
    info: dict[str, list] = dataclasses.field(
        default_factory=dict
    )
    calc: dict[str, list] = dataclasses.field(
        default_factory=dict
    )

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

        start_idx = len(self.positions)

        self.positions.extend([atoms.positions for atoms in frames])
        self.numbers.extend([atoms.numbers for atoms in frames])
        self.pbc.extend([atoms.pbc for atoms in frames])
        self.cell.extend([atoms.cell.array for atoms in frames])

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
