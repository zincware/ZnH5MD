import dataclasses
import os
import pathlib
from collections.abc import MutableSequence

import ase
import h5py
import numpy as np
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md.format as fmt
from znh5md import utils

MutableSequence = object

# TODO: use pint to convert the units in the h5md file to ase units
# TODO: allow to keep the file open when extending / appending to the file
# TODO: allow external file handles instead of providing filename


@dataclasses.dataclass
class IO(MutableSequence):
    filename: os.PathLike

    author: str = "'N/A"
    author_email: str = "N/A"
    creator: str = "N/A"
    creator_version: str = "N/A"
    particle_group: str | None = None

    def __post_init__(self):
        self.filename = pathlib.Path(self.filename)
        if self.particle_group and self.filename.exists():
            with h5py.File(self.filename, "r") as f:
                self.particle_group = next(iter(f["particles"].keys()))
        else:
            self.particle_group = "atoms"

    def create_file(self):
        with h5py.File(self.filename, "w") as f:
            g_h5md = f.create_group("h5md")
            g_h5md.attrs["version"] = np.array([1, 1])
            g_author = g_h5md.create_group("author")
            g_author.attrs["name"] = self.author
            g_author.attrs["email"] = self.author_email
            g_creator = g_h5md.create_group("creator")
            g_creator.attrs["name"] = self.creator
            g_creator.attrs["version"] = self.creator_version

            g_particles = f.create_group("particles")

    def __len__(self) -> int:
        with h5py.File(self.filename, "r") as f:
            return len(f["particles"][self.particle_group]["species"]["value"])

    def __getitem__(self, index) -> ase.Atoms | list[ase.Atoms]:
        single_item = isinstance(index, int)
        index = [index] if single_item else index

        arrays_data = {}
        calc_data = {}
        info_data = {}

        with h5py.File(self.filename, "r") as f:
            atomic_numbers = fmt.get_atomic_numbers(
                f["particles"], self.particle_group, index
            )
            positions = fmt.get_positions(f["particles"], self.particle_group, index)
            cell = fmt.get_box(f["particles"], self.particle_group, index)
            pbc = fmt.get_pbc(f["particles"], self.particle_group, index)
            momenta = fmt.get_momenta(f["particles"], self.particle_group, index)
            for key in f["particles"][self.particle_group].keys():
                if key not in fmt.ASE_TO_H5MD.inverse:
                    if key in all_properties:
                        calc_data[key] = fmt.get_property(
                            f["particles"], self.particle_group, key, index
                        )
                    else:
                        arrays_data[key] = fmt.get_property(
                            f["particles"], self.particle_group, key, index
                        )
            if f"observables/{self.particle_group}" in f:
                for key in f[f"observables/{self.particle_group}"].keys():
                    if key in all_properties:
                        calc_data[key] = fmt.get_property(
                            f["observables"],
                            self.particle_group,
                            key,
                            index,
                        )
                    else:
                        info_data[key] = fmt.get_property(
                            f["observables"],
                            self.particle_group,
                            key,
                            index,
                        )

        structures = []
        if atomic_numbers is not None:
            for idx in range(len(atomic_numbers)):
                atoms = ase.Atoms(symbols=utils.remove_nan_rows(atomic_numbers[idx]))
                if positions is not None:
                    atoms.positions = utils.remove_nan_rows(positions[idx])
                if cell is not None:
                    atoms.cell = cell[idx]
                if momenta is not None:
                    atoms.set_momenta(utils.remove_nan_rows(momenta[idx]))
                if pbc is not None:
                    atoms.pbc = pbc
                for key, value in arrays_data.items():
                    atoms.new_array(key, utils.remove_nan_rows(value[idx]))

                if len(calc_data) > 0:
                    atoms.calc = SinglePointCalculator(atoms)
                for key, value in calc_data.items():
                    atoms.calc.results[key] = utils.remove_nan_rows(value[idx])

                for key, value in info_data.items():
                    atoms.info[key] = utils.remove_nan_rows(value[idx])

                structures.append(atoms)

        return structures[0] if single_item else structures

    def extend(self, images: list[ase.Atoms]):
        if not self.filename.exists():
            self.create_file()

        species = []
        positions = []

        for atoms in images:
            species.append(atoms.get_atomic_numbers())
            positions.append(atoms.get_positions())

        species = utils.concatenate_varying_shape_arrays(species)
        positions = utils.concatenate_varying_shape_arrays(positions)

        with h5py.File(self.filename, "a") as f:
            if self.particle_group not in f["particles"]:
                g_particle_grp = f["particles"].create_group(self.particle_group)
                g_particle_grp = f["particles"][self.particle_group]
                # add g_particle_grp.attrs["species"]["value"] = species
                g_species = g_particle_grp.create_group("species")
                ds_value = g_species.create_dataset(
                    "value",
                    data=species,
                    dtype=np.float32,
                    chunks=True,
                    maxshape=(None, None),
                )

                g_positions = g_particle_grp.create_group("position")
                ds_value = g_positions.create_dataset(
                    "value",
                    data=positions,
                    chunks=True,
                    maxshape=(None, None, None),
                    dtype=np.float32,
                )
            else:
                # we assume every key exists and won't create new datasets.
                # if there is suddenly new data we would have to fill
                # everything with NaNs before that value, which is
                # currently not implemented
                g_particle_grp = f["particles"][self.particle_group]

                g_species = g_particle_grp["species"]
                utils.fill_dataset(g_species["value"], species)

                g_positions = g_particle_grp["position"]
                utils.fill_dataset(g_positions["value"], positions)

    def append(self, atoms: ase.Atoms):
        self.extend([atoms])
