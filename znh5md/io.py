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

    # NOT H5MD conform, specify pbc per step
    pbc_group: bool = True

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
                    if isinstance(pbc[idx], np.ndarray):
                        atoms.pbc = pbc[idx]
                    else:
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

        data = []

        for atoms in images:
            data.append(fmt.extract_atoms_data(atoms))

        data = fmt.combine_asedata(data)

        with h5py.File(self.filename, "a") as f:
            if self.particle_group not in f["particles"]:
                g_particle_grp = f["particles"].create_group(self.particle_group)
                g_species = g_particle_grp.create_group("species")
                ds_value = g_species.create_dataset(
                    "value",
                    data=data.atomic_numbers,
                    dtype=np.float64,
                    chunks=True,
                    maxshape=([None] * data.atomic_numbers.ndim),
                )
                ds_time = g_species.create_dataset(
                    "time",
                    dtype=int,
                    data=np.arange(len(data.atomic_numbers)),
                )
                ds_time.attrs["unit"] = "fs"
                ds_frame = g_species.create_dataset(
                    "step",
                    dtype=int,
                    data=np.arange(len(data.atomic_numbers)),
                )
                if data.positions is not None:
                    g_positions = g_particle_grp.create_group("position")
                    ds_value = g_positions.create_dataset(
                        "value",
                        data=data.positions,
                        chunks=True,
                        maxshape=([None] * data.positions.ndim),
                        dtype=np.float64,
                    )
                    ds_time = g_positions.create_dataset(
                        "time",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                    ds_time.attrs["unit"] = "fs"
                    ds_frame = g_positions.create_dataset(
                        "step",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                if data.cell is not None:
                    g_cell = g_particle_grp.create_group("box")
                    g_cell.attrs["dimension"] = 3  # hard coded for now
                    g_edges = g_cell.create_group("edges")
                    ds_value = g_edges.create_dataset(
                        "value",
                        data=data.cell,
                        chunks=True,
                        maxshape=([None] * data.cell.ndim),
                        dtype=np.float64,
                    )
                    ds_time = g_edges.create_dataset(
                        "time",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                    ds_time.attrs["unit"] = "fs"
                    ds_frame = g_edges.create_dataset(
                        "step",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                if self.pbc_group:
                    if data.pbc is not None:
                        if "box" not in g_particle_grp:
                            g_cell = g_particle_grp.create_group("box")
                        else:
                            g_cell = g_particle_grp["box"]
                        g_pbc = g_cell.create_group("pbc")
                        ds_value = g_pbc.create_dataset(
                            "value",
                            data=data.pbc,
                            chunks=True,
                            maxshape=(None, 3),
                            dtype=np.float64,
                        )
                        ds_time = g_pbc.create_dataset(
                            "time",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )
                        ds_time.attrs["unit"] = "fs"
                        ds_frame = g_pbc.create_dataset(
                            "step",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )

                if data.momenta is not None:
                    g_momenta = g_particle_grp.create_group("momentum")
                    ds_value = g_momenta.create_dataset(
                        "value",
                        data=data.momenta,
                        chunks=True,
                        maxshape=([None] * data.momenta.ndim),
                        dtype=np.float64,
                    )
                    ds_time = g_momenta.create_dataset(
                        "time",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                    ds_time.attrs["unit"] = "fs"
                    ds_frame = g_momenta.create_dataset(
                        "step",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                for key, value in data.arrays_data.items():
                    g_array = g_particle_grp.create_group(key)
                    ds_value = g_array.create_dataset(
                        "value",
                        data=value,
                        chunks=True,
                        maxshape=([None] * value.ndim),
                        dtype=np.float64,
                    )
                    ds_time = g_array.create_dataset(
                        "time",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )
                    ds_time.attrs["unit"] = "fs"
                    ds_frame = g_array.create_dataset(
                        "step",
                        dtype=int,
                        data=np.arange(len(data.atomic_numbers)),
                    )

                if len(data.calc_data) > 0:
                    if "observables" not in f:
                        g_observables = f.create_group("observables")
                    if self.particle_group not in f["observables"]:
                        g_calc = f["observables"].create_group(self.particle_group)
                    else:
                        g_calc = f["observables"][self.particle_group]
                    for key, value in data.calc_data.items():
                        g_observable = g_calc.create_group(key)
                        ds_value = g_observable.create_dataset(
                            "value",
                            data=value,
                            chunks=True,
                            maxshape=([None] * value.ndim),
                            dtype=np.float64,
                        )
                        ds_time = g_observable.create_dataset(
                            "time",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )
                        ds_time.attrs["unit"] = "fs"
                        ds_frame = g_observable.create_dataset(
                            "step",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )

                if len(data.info_data) > 0:
                    if "observables" not in f:
                        g_observables = f.create_group("observables")
                    if self.particle_group not in f["observables"]:
                        g_info = f["observables"].create_group(self.particle_group)
                    else:
                        g_info = f["observables"][self.particle_group]
                    for key, value in data.info_data.items():
                        g_observable = g_info.create_group(key)
                        ds_value = g_observable.create_dataset(
                            "value",
                            data=value,
                            chunks=True,
                            maxshape=([None] * value.ndim),
                            dtype=np.float64,
                        )
                        ds_time = g_observable.create_dataset(
                            "time",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )
                        ds_time.attrs["unit"] = "fs"
                        ds_frame = g_observable.create_dataset(
                            "step",
                            dtype=int,
                            data=np.arange(len(data.atomic_numbers)),
                        )
            else:
                # we assume every key exists and won't create new datasets.
                # if there is suddenly new data we would have to fill
                # everything with NaNs before that value, which is
                # currently not implemented
                g_particle_grp = f["particles"][self.particle_group]
                g_species = g_particle_grp["species"]
                utils.fill_dataset(g_species["value"], data.atomic_numbers)
                # TODO: should check if there are groups that are not in the new data.
                # They also must be extended then!
                if data.positions is not None:
                    g_positions = g_particle_grp["position"]
                    utils.fill_dataset(g_positions["value"], data.positions)
                if data.cell is not None:
                    g_cell = g_particle_grp["box"]
                    g_edges = g_cell["edges"]
                    utils.fill_dataset(g_edges["value"], data.cell)
                if self.pbc_group:
                    if data.pbc is not None:
                        g_cell = g_particle_grp["box"]
                        g_pbc = g_cell["pbc"]
                        utils.fill_dataset(g_pbc["value"], data.pbc)

                if data.momenta is not None:
                    g_momenta = g_particle_grp["momentum"]
                    utils.fill_dataset(g_momenta["value"], data.momenta)

                for key, value in data.arrays_data.items():
                    g_array = g_particle_grp[key]
                    utils.fill_dataset(g_array["value"], value)

                if f"observables/{self.particle_group}" in f:
                    g_observables = f["observables"][self.particle_group]
                    for key, value in data.calc_data.items():
                        g_val = g_observables[key]
                        utils.fill_dataset(g_val["value"], value)

                    for key, value in data.info_data.items():
                        g_val = g_observables[key]
                        utils.fill_dataset(g_val["value"], value)

    def append(self, atoms: ase.Atoms):
        self.extend([atoms])
