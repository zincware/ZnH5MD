import dataclasses
import numpy as np
import numpy.typing as nptype
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import typing as t

CONTENT_TYPE = dict[str, np.ndarray | dict | float | int | str | bool]


def process_category(target: dict[str, list], content: CONTENT_TYPE, index) -> None:
    """
    Process a category (arrays, info, calc) for a single atoms object, ensuring
    that keys are added and missing values are backfilled.

    Parameters:
        target (dict): The main dictionary storing data for the category.
        atoms_data (dict): The data from the current atoms object (arrays, info, or calc).
        index (int): The index of the current atoms object in the trajectory used for backfilling.
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
    info: dict[str, nptype.NDArray[np.object_]] = dataclasses.field(
        default_factory=dict
    )  # (N, ...)
    calc: dict[str, nptype.NDArray[np.object_]] = dataclasses.field(
        default_factory=dict
    )  # (N, ...)

    def __iter__(self) -> t.Iterator[ase.Atoms]:
        """Iterate over the frames."""
        if self.positions is None:
            return iter([])
        for idx in range(len(self.positions)):
            yield self[idx]

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
        pbc = np.array([atoms.pbc for atoms in frames])
        cell = np.array([atoms.cell.array for atoms in frames])

        if self.positions is None:
            self.positions = positions
            self.numbers = numbers
            self.pbc = pbc
            self.cell = cell
        else:
            raise NotImplementedError("Can only extend Frames once.")
            self.positions = np.concatenate([self.positions, positions])
            self.numbers = np.concatenate([self.numbers, numbers])
            self.pbc = np.concatenate([self.pbc, pbc])
            self.cell = np.concatenate([self.cell, cell])

        for idx, atoms in enumerate(frames):
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

        for key in self.arrays:
            self.arrays[key] = np.array(self.arrays[key], dtype=object)
        for key in self.info:
            self.info[key] = np.array(self.info[key], dtype=object)
        for key in self.calc:
            self.calc[key] = np.array(self.calc[key], dtype=object)


class _MISSING:
    """Sentinel value for missing entries."""

    pass


MISSING = _MISSING()
