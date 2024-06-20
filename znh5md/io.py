import dataclasses
import os
import pathlib
from collections.abc import MutableSequence
from typing import List, Optional, Union

import ase
import h5py
import numpy as np
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md.format as fmt
from znh5md import utils

# TODO: use pint to convert the units in the h5md file to ase units
# TODO: allow to keep the file open when extending / appending to the file
# TODO: allow external file handles instead of providing filename


@dataclasses.dataclass
class IO(MutableSequence):
    """A class for handling H5MD files for ASE Atoms objects."""

    filename: Union[str, os.PathLike]
    pbc_group: bool = True  # Specify PBC per step (Not H5MD conform)
    save_units: bool = True  # Export ASE units into the H5MD file
    author: str = "N/A"
    author_email: str = "N/A"
    creator: str = "N/A"
    creator_version: str = "N/A"
    particle_group: Optional[str] = None

    def __post_init__(self):
        self.filename = pathlib.Path(self.filename)
        self._set_particle_group()

    def _set_particle_group(self):
        if self.particle_group and self.filename.exists():
            with h5py.File(self.filename, "r") as f:
                self.particle_group = next(iter(f["particles"].keys()))
        elif not self.particle_group:
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
            f.create_group("particles")

    def __len__(self) -> int:
        with h5py.File(self.filename, "r") as f:
            return len(f["particles"][self.particle_group]["species"]["value"])

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ase.Atoms, List[ase.Atoms]]:
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
            velocities = fmt.get_velocities(f["particles"], self.particle_group, index)

            self._extract_additional_data(f, index, arrays_data, calc_data, info_data)

        structures = self._build_structures(
            atomic_numbers,
            positions,
            cell,
            pbc,
            velocities,
            arrays_data,
            calc_data,
            info_data,
        )

        return structures[0] if single_item else structures

    def _extract_additional_data(self, f, index, arrays_data, calc_data, info_data):
        for key in f["particles"][self.particle_group].keys():
            if key not in fmt.ASE_TO_H5MD.inverse:
                if key in all_properties or key == "force":
                    calc_data[key if key != "force" else "forces"] = fmt.get_property(
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
                        f["observables"], self.particle_group, key, index
                    )
                else:
                    info_data[key] = fmt.get_property(
                        f["observables"], self.particle_group, key, index
                    )

    def _build_structures(
        self,
        atomic_numbers,
        positions,
        cell,
        pbc,
        velocities,
        arrays_data,
        calc_data,
        info_data,
    ):
        structures = []
        if atomic_numbers is not None:
            for idx in range(len(atomic_numbers)):
                atoms = ase.Atoms(symbols=utils.remove_nan_rows(atomic_numbers[idx]))
                if positions is not None:
                    atoms.positions = utils.remove_nan_rows(positions[idx])
                if cell is not None:
                    atoms.cell = cell[idx]
                if velocities is not None:
                    atoms.set_velocities(utils.remove_nan_rows(velocities[idx]))
                if pbc is not None:
                    atoms.pbc = pbc[idx] if isinstance(pbc[idx], np.ndarray) else pbc

                for key, value in arrays_data.items():
                    atoms.new_array(key, utils.remove_nan_rows(value[idx]))

                if calc_data:
                    atoms.calc = SinglePointCalculator(atoms)
                    for key, value in calc_data.items():
                        atoms.calc.results[key] = utils.remove_nan_rows(value[idx])

                for key, value in info_data.items():
                    atoms.info[key] = utils.remove_nan_rows(value[idx])

                structures.append(atoms)
        return structures

    def extend(self, images: List[ase.Atoms]):
        if not self.filename.exists():
            self.create_file()

        data = [fmt.extract_atoms_data(atoms) for atoms in images]
        combined_data = fmt.combine_asedata(data)

        with h5py.File(self.filename, "a") as f:
            if self.particle_group not in f["particles"]:
                self._create_particle_group(f, combined_data)
            else:
                self._extend_existing_data(f, combined_data)

    def _create_particle_group(self, f, data):
        g_particle_grp = f["particles"].create_group(self.particle_group)
        self._create_group(g_particle_grp, "species", data.atomic_numbers)
        self._create_group(g_particle_grp, "position", data.positions, "Angstrom")
        self._create_group(g_particle_grp, "box/edges", data.cell)
        g_particle_grp["box"].attrs["dimension"] = 3

        if self.pbc_group and data.pbc is not None:
            self._create_group(g_particle_grp, "box/pbc", data.pbc)
        self._create_group(g_particle_grp, "velocity", data.velocities, "Angstrom/fs")
        for key, value in data.arrays_data.items():
            self._create_group(
                g_particle_grp, key, value, "eV/Angstrom" if key == "force" else None
            )
        self._create_observables(f, data.info_data)

    def _create_group(self, parent_grp, name, data, unit=None):
        if data is not None:
            g_grp = parent_grp.create_group(name)
            ds_value = g_grp.create_dataset(
                "value",
                data=data,
                dtype=np.float64,
                chunks=True,
                maxshape=([None] * data.ndim),
            )
            if unit and self.save_units:
                ds_value.attrs["unit"] = unit
            self._add_time_and_step(g_grp, len(data))

    def _add_time_and_step(self, grp, length):
        ds_time = grp.create_dataset("time", dtype=int, data=np.arange(length))
        ds_time.attrs["unit"] = "fs"
        grp.create_dataset("step", dtype=int, data=np.arange(length))

    def _create_observables(self, f, info_data):
        if info_data:
            g_observables = f.require_group("observables")
            g_info = g_observables.require_group(self.particle_group)
            for key, value in info_data.items():
                g_observable = g_info.create_group(key)
                ds_value = g_observable.create_dataset(
                    "value",
                    data=value,
                    dtype=np.float64,
                    chunks=True,
                    maxshape=([None] * value.ndim),
                )
                self._add_time_and_step(g_observable, len(value))

    def _extend_existing_data(self, f, data):
        g_particle_grp = f["particles"][self.particle_group]
        self._extend_group(g_particle_grp, "species", data.atomic_numbers)
        self._extend_group(g_particle_grp, "position", data.positions)
        self._extend_group(g_particle_grp, "box/edges", data.cell)
        if self.pbc_group and data.pbc is not None:
            self._extend_group(g_particle_grp, "box/pbc", data.pbc)
        self._extend_group(g_particle_grp, "velocity", data.velocities)
        for key, value in data.arrays_data.items():
            self._extend_group(g_particle_grp, key, value)
        self._extend_observables(f, data.info_data)

    def _extend_group(self, parent_grp, name, data):
        if data is not None and name in parent_grp:
            g_grp = parent_grp[name]
            utils.fill_dataset(g_grp["value"], data)

    def _extend_observables(self, f, info_data):
        if f"observables/{self.particle_group}" in f:
            g_observables = f[f"observables/{self.particle_group}"]
            for key, value in info_data.items():
                if key in g_observables:
                    g_val = g_observables[key]
                    utils.fill_dataset(g_val["value"], value)

    def append(self, atoms: ase.Atoms):
        self.extend([atoms])

    def __delitem__(self, index):
        raise NotImplementedError("Deleting items is not supported")

    def __setitem__(self, index, value):
        raise NotImplementedError("Setting items is not supported")

    def insert(self, index, value):
        raise NotImplementedError("Inserting items is not supported")
