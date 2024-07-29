import contextlib
import dataclasses
import os
import pathlib
import typing as t
import warnings
from collections.abc import MutableSequence
from typing import List, Optional, Union

import ase
import h5py
import numpy as np
from ase.calculators.calculator import all_properties
from tqdm import tqdm

import znh5md.format as fmt
from znh5md import utils

# TODO: use pint to convert the units in the h5md file to ase units


@contextlib.contextmanager
def _open_file(
    filename: str | os.PathLike | None, file_handle: h5py.File | None, **kwargs
) -> t.Generator[h5py.File, None, None]:
    if file_handle is not None:
        yield file_handle
    else:
        with h5py.File(filename, **kwargs) as f:
            yield f


@dataclasses.dataclass
class IO(MutableSequence):
    """A class for handling H5MD files for ASE Atoms objects."""

    filename: Optional[str | os.PathLike] = None
    file_handle: Optional[h5py.File] = None
    pbc_group: bool = True  # Specify PBC per step (Not H5MD conform)
    save_units: bool = True  # Export ASE units into the H5MD file
    author: str = "N/A"
    author_email: str = "N/A"
    creator: str = "N/A"
    creator_version: str = "N/A"
    particle_group: Optional[str] = None
    compression: Optional[str] = "gzip"
    compression_opts: Optional[int] = None
    timestep: float = 1.0
    store: t.Literal["time", "linear"] = "linear"
    tqdm_limit: int = 100

    def __post_init__(self):
        if self.filename is None and self.file_handle is None:
            raise ValueError("Either filename or file_handle must be provided")
        if self.filename is not None and self.file_handle is not None:
            raise ValueError("Only one of filename or file_handle can be provided")
        if self.filename is not None:
            self.filename = pathlib.Path(self.filename)
        self._set_particle_group()
        self._read_author_creator()

    def _set_particle_group(self):
        if self.particle_group is not None:
            pass
        elif self.filename is not None and self.filename.exists():
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.particle_group = next(iter(f["particles"].keys()))
        elif (
            self.file_handle is not None
            and pathlib.Path(self.file_handle.filename).exists()
        ):
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.particle_group = next(iter(f["particles"].keys()))
        else:
            self.particle_group = "atoms"

    def _read_author_creator(self):
        with contextlib.suppress(FileNotFoundError, KeyError):
            # FileNotFoundError if the filename does not exist
            # KeyError if the file has not yet been initialized as H5MD
            #   or the keys are not provided, which is officially
            #   not allowed in H5MD.
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.author = f["h5md"]["author"].attrs["name"]
                self.author_email = f["h5md"]["author"].attrs["email"]
                self.creator = f["h5md"]["creator"].attrs["name"]
                self.creator_version = f["h5md"]["creator"].attrs["version"]

    def create_file(self):
        with _open_file(self.filename, self.file_handle, mode="w") as f:
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
        with _open_file(self.filename, self.file_handle, mode="r") as f:
            return len(f["particles"][self.particle_group]["species"]["value"])

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ase.Atoms, List[ase.Atoms]]:
        single_item = isinstance(index, int)
        index = [index] if single_item else index

        arrays_data = {}
        calc_data = {}
        info_data = {}

        with _open_file(self.filename, self.file_handle, mode="r") as f:
            arrays_data["species"] = fmt.get_atomic_numbers(
                f["particles"], self.particle_group, index
            )
            positions = fmt.get_positions(f["particles"], self.particle_group, index)
            if positions is not None:
                arrays_data["positions"] = positions
            cell = fmt.get_box(f["particles"], self.particle_group, index)
            pbc = fmt.get_pbc(f["particles"], self.particle_group, index)

            self._extract_additional_data(f, index, arrays_data, calc_data, info_data)
            self._update_info_data_with_time_and_step(info_data, f, index)

        structures = utils.build_structures(
            cell,
            pbc,
            arrays_data,
            calc_data,
            info_data,
        )

        return structures[0] if single_item else structures

    def _update_info_data_with_time_and_step(self, info_data, f, index):
        # we use the time value from `species` because it is guaranteed to exist
        time = fmt.get_time(f["particles"], self.particle_group, index)
        step = fmt.get_step(f["particles"], self.particle_group, index)
        if any(key in info_data for key in fmt.CustomINFOData.__members__):
            raise ValueError("key is already present in the info data.")
        info_data[fmt.CustomINFOData.h5md_time.name] = time.tolist()
        info_data[fmt.CustomINFOData.h5md_step.name] = step.tolist()

    def _extract_additional_data(self, f, index, arrays_data, calc_data, info_data):
        for key in f["particles"][self.particle_group].keys():
            if key not in list(fmt.ASE_TO_H5MD.values()):
                if (
                    f["particles"][self.particle_group][key]["value"].attrs.get(
                        "ASE_CALCULATOR_RESULT"
                    )
                    or key in all_properties
                    or key == "force"
                ):
                    calc_data[key if key != "force" else "forces"] = fmt.get_property(
                        f["particles"], self.particle_group, key, index
                    )
                else:
                    arrays_data[key] = fmt.get_property(
                        f["particles"], self.particle_group, key, index
                    )
        if f"observables/{self.particle_group}" in f:
            for key in f[f"observables/{self.particle_group}"].keys():
                if (
                    f["observables"][self.particle_group][key]["value"].attrs.get(
                        "ASE_CALCULATOR_RESULT"
                    )
                    or key in all_properties
                ):
                    calc_data[key] = fmt.get_property(
                        f["observables"], self.particle_group, key, index
                    )
                else:
                    info_data[key] = fmt.get_property(
                        f["observables"], self.particle_group, key, index
                    )

    def extend(self, images: List[ase.Atoms]):
        if len(images) == 0:
            warnings.warn("No data provided")
            return
        if self.filename is not None and not self.filename.exists():
            self.create_file()
        if self.file_handle is not None:
            needs_creation = False
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                needs_creation = "h5md" not in f
            if needs_creation:
                self.create_file()

        data = []
        _disable_tqdm = len(images) < self.tqdm_limit if self.tqdm_limit > 0 else True
        for atoms in tqdm(
            images, ncols=120, desc="Preparing data", disable=_disable_tqdm
        ):
            data.append(fmt.extract_atoms_data(atoms))
        combined_data = fmt.combine_asedata(data)

        with _open_file(self.filename, self.file_handle, mode="a") as f:
            if self.particle_group not in f["particles"]:
                self._create_particle_group(f, combined_data)
            else:
                self._extend_existing_data(f, combined_data)

    def _create_particle_group(self, f, data: fmt.ASEData):
        g_particle_grp = f["particles"].create_group(self.particle_group)
        self._create_group(
            g_particle_grp, "box/edges", data.cell, time=data.time, step=data.step
        )
        g_particle_grp["box"].attrs["dimension"] = 3
        g_particle_grp["box"].attrs["boundary"] = [
            "periodic" if y else "none" for y in data.pbc[0]
        ]

        if self.pbc_group and data.pbc is not None:
            self._create_group(
                g_particle_grp, "box/pbc", data.pbc, time=data.time, step=data.step
            )

        _disable_tqdm = len(data) < self.tqdm_limit if self.tqdm_limit > 0 else True
        for key, value in tqdm(
            data.particles.items(),
            ncols=120,
            desc="Creating groups",
            disable=_disable_tqdm,
        ):
            self._create_group(
                g_particle_grp,
                key,
                value,
                data.metadata.get(key, {}).get("unit"),
                calc=data.metadata.get(key, {}).get("calc"),
                time=data.time,
                step=data.step,
            )
        self._create_observables(
            f,
            data.observables,
            data.metadata,
            time=data.time,
            step=data.step,
        )

    def _create_group(
        self,
        parent_grp,
        name,
        data,
        unit: Optional[str] = None,
        calc: Optional[bool] = None,
        time: np.ndarray | None = None,
        step: np.ndarray | None = None,
    ):
        if data is not None:
            g_grp = parent_grp.create_group(name)
            ds_value = g_grp.create_dataset(
                "value",
                data=data,
                dtype=np.float64,
                chunks=True,
                maxshape=([None] * data.ndim),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            if calc is not None:
                ds_value.attrs["ASE_CALCULATOR_RESULT"] = calc
            if unit and self.save_units:
                ds_value.attrs["unit"] = unit
            if time is None:
                time = np.arange(len(data)) * self.timestep
            elif self.store == "linear":
                warnings.warn("time is ignored in 'linear' storage mode")
            if step is None:
                step = np.arange(len(data))
            elif self.store == "linear":
                warnings.warn("step is ignored in 'linear' storage mode")
            self._add_time_and_step(g_grp, step, time)

    def _add_time_and_step(self, grp, step: np.ndarray, time: np.ndarray):
        if self.store == "time":
            ds_time = grp.create_dataset(
                "time",
                dtype=np.float64,
                data=time,
                compression=self.compression,
                compression_opts=self.compression_opts,
                maxshape=(None,),
            )
            ds_time.attrs["unit"] = "fs"
            ds_step = grp.create_dataset(
                "step",
                dtype=int,
                data=step,
                compression=self.compression,
                compression_opts=self.compression_opts,
                maxshape=(None,),
            )
        elif self.store == "linear":
            ds_time = grp.create_dataset(
                "time",
                dtype=np.float64,
                data=self.timestep,
            )
            ds_time.attrs["unit"] = "fs"
            ds_step = grp.create_dataset(
                "step",
                dtype=int,
                data=1,
            )
            ds_time[()] = self.timestep
            ds_step[()] = 1

    def _create_observables(
        self,
        f,
        info_data,
        metadata: dict,
        time: np.ndarray | None = None,
        step: np.ndarray | None = None,
    ):
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
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                if metadata.get(key, {}).get("calc") is not None:
                    ds_value.attrs["ASE_CALCULATOR_RESULT"] = metadata[key]["calc"]
                if metadata.get(key, {}).get("unit") and self.save_units:
                    ds_value.attrs["unit"] = metadata[key]["unit"]
                if time is None:
                    time = np.arange(len(value)) * self.timestep
                elif self.store == "linear":
                    warnings.warn("time is ignored in 'linear' storage mode")
                if step is None:
                    step = np.arange(len(value))
                elif self.store == "linear":
                    warnings.warn("step is ignored in 'linear' storage mode")
                self._add_time_and_step(g_observable, step, time)

    def _extend_existing_data(self, f, data: fmt.ASEData):
        g_particle_grp = f["particles"][self.particle_group]
        self._extend_group(
            g_particle_grp, "box/edges", data.cell, step=data.step, time=data.time
        )
        if self.pbc_group and data.pbc is not None:
            self._extend_group(
                g_particle_grp, "box/pbc", data.pbc, step=data.step, time=data.time
            )
        _disable_tqdm = len(data) < self.tqdm_limit if self.tqdm_limit > 0 else True
        for key, value in tqdm(
            data.particles.items(),
            ncols=120,
            desc="Extending groups",
            disable=_disable_tqdm,
        ):
            self._extend_group(
                g_particle_grp, key, value, step=data.step, time=data.time
            )
        self._extend_observables(f, data.observables, step=data.step, time=data.time)

    def _extend_group(
        self,
        parent_grp,
        name,
        data,
        step: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ):
        if data is not None and name in parent_grp:
            g_grp = parent_grp[name]
            utils.fill_dataset(g_grp["value"], data)
            if self.store == "time":
                if time is None:
                    last_time = g_grp["time"][-1]
                    time = np.arange(len(data)) * self.timestep + last_time
                if step is None:
                    last_step = g_grp["step"][-1]
                    step = np.arange(len(data)) + last_step
                utils.fill_dataset(
                    g_grp["time"],
                    time,
                )
                utils.fill_dataset(g_grp["step"], step)

    def _extend_observables(
        self,
        f,
        info_data,
        step: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ):
        if f"observables/{self.particle_group}" in f:
            g_observables = f[f"observables/{self.particle_group}"]
            for key, value in info_data.items():
                if key in g_observables:
                    g_val = g_observables[key]
                    utils.fill_dataset(g_val["value"], value)
                    if self.store == "time":
                        if time is None:
                            last_time = g_val["time"][-1]
                            time = np.arange(len(value)) * self.timestep + last_time
                        if step is None:
                            last_step = g_val["step"][-1]
                            step = np.arange(len(value)) + last_step
                        utils.fill_dataset(
                            g_val["time"],
                            time,
                        )
                        utils.fill_dataset(g_val["step"], step)

    def append(self, atoms: ase.Atoms):
        self.extend([atoms])

    def __delitem__(self, index):
        raise NotImplementedError("Deleting items is not supported")

    def __setitem__(self, index, value):
        raise NotImplementedError("Setting items is not supported")

    def insert(self, index, value):
        raise NotImplementedError("Inserting items is not supported")
