from collections.abc import MutableSequence
import dataclasses
import os
import h5py
import pathlib
import znh5md.format as fmt
import ase


MutableSequence = object

# TODO: use pint to convert the units in the h5md file to ase units


@dataclasses.dataclass
class IO(MutableSequence):
    filename: os.PathLike
    particle_group: str | None = None

    def __post_init__(self):
        if self.particle_group is None:
            if pathlib.Path(self.filename).exists():
                with h5py.File(self.filename, "r") as f:
                    self.particle_group = next(iter(f["particles"].keys()))
            else:
                self.particle_group = "atoms"

    def __getitem__(self, index) -> ase.Atoms | list[ase.Atoms]:
        single_item = isinstance(index, int)
        index = [index] if single_item else index

        with h5py.File(self.filename, "r") as f:
            atomic_numbers = fmt.get_atomic_numbers(
                f["particles"], self.particle_group, index
            )
            positions = fmt.get_positions(f["particles"], self.particle_group, index)
            cell = fmt.get_box(f["particles"], self.particle_group, index)
            pbc = fmt.get_pbc(f["particles"], self.particle_group, index)
            momenta = fmt.get_momenta(f["particles"], self.particle_group, index)

        structures = []
        for idx in range(len(atomic_numbers)):
            atoms = ase.Atoms(
                symbols=atomic_numbers[idx],
                positions=positions[idx],
                cell=cell[idx],
                momenta=momenta[idx],
            )
            if isinstance(pbc[0], bool):
                atoms.pbc = pbc
            else:
                atoms.pbc = pbc[idx]
            structures.append(atoms)

        return structures[0] if single_item else structures
