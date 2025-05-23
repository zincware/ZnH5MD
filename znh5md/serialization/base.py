import contextlib
import dataclasses
import functools
import json
import typing as t

import ase
import h5py
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from znh5md.misc import MISSING, concatenate_varying_shape_arrays
from znh5md.units import get_unit

# Define allowed types
ALLOWED_TYPES = t.Union[np.ndarray, dict, float, int, str, bool, list, type(MISSING)]

# Define content type
CONTENT_TYPE = dict[str, ALLOWED_TYPES]

# Define origin type
ORIGIN_TYPE = t.Literal["calc", "info", "arrays", "atoms"]


@dataclasses.dataclass
class Entry:
    value: list[ALLOWED_TYPES] | np.ndarray = dataclasses.field(repr=False)
    origin: ORIGIN_TYPE | None
    name: str
    unit: str | None = None

    def __post_init__(self):
        if self.unit is None:
            self.unit = get_unit(self.name)

        if isinstance(self.value, list):
            if isinstance(self.value[0], list):
                try:
                    self.value = [np.array(v, dtype=np.float64) for v in self.value]
                except ValueError:
                    pass

    @functools.cached_property
    def ref(self) -> t.Any:
        for v in self.value:
            if v is not MISSING:
                return v
        else:
            raise ValueError("All values are MISSING")

    @functools.cached_property
    def dtype(self) -> t.Any:
        if isinstance(self.ref, np.ndarray):
            if self.ref.dtype.kind in ["O", "S", "U"]:
                return h5py.string_dtype()
            else:
                return np.float64
        elif isinstance(self.ref, str):
            return h5py.string_dtype()
        elif isinstance(self.ref, dict):
            return h5py.string_dtype()
        elif isinstance(self.ref, list):
            if any(not isinstance(v, (int, float, bool)) for v in self.ref):
                return h5py.string_dtype()
            return np.float64
        else:
            return np.float64

    @property
    def fillvalue(self) -> t.Any:
        if self.dtype == h5py.string_dtype():
            return ""
        else:
            return np.nan

    def dump(self) -> t.Tuple[np.ndarray | list, t.Any]:
        data = self.value
        try:
            if self.dtype == h5py.string_dtype():
                # Handle string data
                serialized_data = [
                    json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)
                    if v is not MISSING
                    else ""
                    for v in data
                ]
                return serialized_data, h5py.string_dtype()
            else:
                # Handle non-string data
                processed_data = [
                    np.array(v, dtype=self.dtype)
                    if v is not MISSING
                    else np.full_like(self.ref, self.fillvalue, dtype=self.dtype)
                    for v in data
                ]
                return (
                    concatenate_varying_shape_arrays(
                        processed_data, self.fillvalue, dtype=np.float64
                    ),
                    np.float64,
                )
        except:
            print(f"Error in dump for {self}")
            print(self.value)
            print(type(self.value))
            raise


def process_momenta(target: dict[str, list], atoms: ase.Atoms, index: int) -> None:
    if "velocities" not in target:
        if "momenta" in atoms.arrays:
            target["velocities"] = [MISSING] * index + [atoms.get_velocities()]
    else:
        if "momenta" in atoms.arrays:
            target["velocities"].append(atoms.get_velocities())
        else:
            target["velocities"].append(MISSING)


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
    seen = set(content.keys()) | {"momenta", "velocities"}
    unseen = set(target.keys()) - seen

    for key in content:
        if key == "momenta":
            # Momenta are handled separately via velocities
            continue
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
    arrays: dict[str, list] = dataclasses.field(default_factory=dict)
    info: dict[str, list] = dataclasses.field(default_factory=dict)
    calc: dict[str, list] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_ase(
        cls, frames: t.Iterable[ase.Atoms], include: list[str] | None = None
    ) -> "Frames":
        """Create a Frames object from a sequence of ASE Atoms objects."""
        obj = cls()
        obj.extend(frames, include=include)
        return obj

    def keys(self) -> t.Iterator[str]:
        """Iterate over the keys."""
        yield "numbers"
        yield "positions"
        yield "pbc"
        yield "cell"
        for key in self.arrays:
            yield key
        for key in self.info:
            yield key
        for key in self.calc:
            yield key

    def items(self) -> t.Iterator[Entry]:
        """Iterate over the items."""
        yield Entry(self.numbers, "atoms", name="numbers")  # numbers has to be first!
        if len(self.positions) > 0:
            yield Entry(self.positions, "atoms", name="positions")
        if len(self.pbc) > 0:
            yield Entry(self.pbc, "atoms", name="pbc")
        if len(self.cell) > 0:
            yield Entry(self.cell, "atoms", name="cell")
        for key in self.arrays:
            yield Entry(self.arrays[key], "arrays", name=key)
        for key in self.info:
            yield Entry(self.info[key], "info", name=key)
        for key in self.calc:
            yield Entry(self.calc[key], "calc", name=key)

    def check(self) -> None:
        if len(list(self.keys())) != len(set(self.keys())):
            # find duplicates
            duplicates = []
            seen = set()
            for key in self.keys():
                if key in seen:
                    duplicates.append(key)
                else:
                    seen.add(key)
            raise ValueError(f"Duplicate keys found in Frames object: {duplicates}")

    def __iter__(self) -> t.Iterator[ase.Atoms]:
        """Iterate over the frames."""
        if self.numbers is None:
            return iter([])
        for idx in range(len(self.numbers)):
            try:
                yield self[idx]
            except IndexError:
                return

    def __len__(self) -> int:
        """Return the number of frames."""
        if self.numbers is None:
            return 0
        return len(self.numbers)

    def __getitem__(self, idx: int) -> ase.Atoms:
        """Return a single frame."""
        # this raises the IndexError to determine the length of the Frames object
        atoms = ase.Atoms(
            numbers=self.numbers[idx],
            # positions=self.positions[idx],
            # cell=self.cell[idx],
            # pbc=self.pbc[idx],
        )
        if len(self.positions) > 0:
            atoms.set_positions(self.positions[idx])
        if len(self.cell) > 0:
            atoms.set_cell(self.cell[idx])
        if len(self.pbc) > 0:
            atoms.set_pbc(self.pbc[idx])
        # all data following here can be missing
        for key in self.arrays:
            with contextlib.suppress(IndexError):
                if self.arrays[key][idx] is MISSING:
                    continue
                if key == "velocities":
                    atoms.set_velocities(self.arrays[key][idx])
                else:
                    atoms.arrays[key] = self.arrays[key][idx]

        for key in self.info:
            with contextlib.suppress(IndexError):
                if self.info[key][idx] is not MISSING:
                    atoms.info[key] = self.info[key][idx]
        for key in self.calc:
            with contextlib.suppress(IndexError):
                if self.calc[key][idx] is not MISSING:
                    if atoms.calc is None:
                        atoms.calc = SinglePointCalculator(atoms)
                    atoms.calc.results[key] = self.calc[key][idx]

        return atoms

    def append(self, atoms: ase.Atoms) -> None:
        """Append a frame to the frames."""
        self.extend([atoms])

    def extend(
        self, frames: t.Iterable[ase.Atoms], include: list[str] | None = None
    ) -> None:
        """Extend the frames with a sequence of frames."""

        if include is not None:
            # Convert include list to a set for faster lookups
            include_set = set(include)
        else:
            include_set = None

        start_idx = len(self.positions)
        self.numbers.extend([atoms.numbers for atoms in frames])

        if include_set is None or "position" in include_set:
            self.positions.extend([atoms.positions for atoms in frames])
        if include_set is None or "box" in include_set:
            self.pbc.extend([atoms.pbc for atoms in frames])
            self.cell.extend([atoms.cell.array for atoms in frames])

        for idx, atoms in enumerate(frames, start=start_idx):
            # Process arrays
            if include_set is None:
                atoms_arrays = {
                    key: value
                    for key, value in atoms.arrays.items()
                    if key not in ["positions", "numbers"]
                }
            else:
                atoms_arrays = {
                    key: value
                    for key, value in atoms.arrays.items()
                    if key not in ["positions", "numbers"] and key in include
                }
            process_category(self.arrays, atoms_arrays, idx)
            if include_set is None or "velocities" in include_set:
                process_momenta(self.arrays, atoms, idx)

            # Process info
            if include_set is None:
                atoms_info = atoms.info
            else:
                atoms_info = {
                    key: value
                    for key, value in atoms.info.items()
                    if key in include_set
                }
            process_category(self.info, atoms_info, idx)

            # Process calc
            if atoms.calc is not None:
                if include_set is None:
                    atoms_calc = atoms.calc.results
                else:
                    atoms_calc = {
                        key: value
                        for key, value in atoms.calc.results.items()
                        if key in include_set
                    }
                process_category(self.calc, atoms_calc, idx)
            elif len(self.calc) != 0:
                process_category(self.calc, {}, idx)
