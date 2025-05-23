import typing as t

import ase
import h5py
import numpy as np

from znh5md.misc import fill_dataset, open_file
from znh5md.path import AttributePath, get_h5md_path
from znh5md.serialization import Entry, Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def create_group(
    f,
    path,
    entry: Entry,
    ref_length: int,
    particles_goup: str,
    pbc_grp: bool,
    store: t.Literal["time", "linear"],
    save_units: bool,
    timestep: float,
    compression: str | None,
    compression_opts: int | None,
    chunk_size: int | None,
) -> None:
    if path in f:
        raise ValueError(f"Group {path} already exists")
    data, dtype = entry.dump()

    if entry.name == "pbc":
        box_grp = f[f"/particles/{particles_goup}"].create_group("box")
        box_grp.attrs.create(
            AttributePath.boundary.value,
            ["periodic" if x else "none" for x in entry.ref],
        )
        box_grp.attrs.create(AttributePath.dimension.value, 3)
        if not pbc_grp:
            return

    grp = f.create_group(path)

    if dtype == h5py.string_dtype():
        chunks = True if chunk_size is None else (chunk_size,)
        ds = grp.create_dataset(
            "value",
            shape=(ref_length + len(data),),
            maxshape=(None,),
            fillvalue=entry.fillvalue,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )
        ds[ref_length:] = data
    else:
        maxshape = tuple(None for _ in data.shape)
        shape = (ref_length + len(data),) + data.shape[1:]
        chunks = (
            True if chunk_size is None else tuple([chunk_size] + list(data.shape[1:]))
        )

        ds = grp.create_dataset(
            "value",
            shape=shape,
            maxshape=maxshape,
            fillvalue=entry.fillvalue,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )
        ds[ref_length:] = data

    if entry.origin is not None:
        grp.attrs.create(AttributePath.origin.value, entry.origin)
    if entry.unit is not None and save_units:
        ds.attrs.create(AttributePath.unit.value, entry.unit)

    # We use linear time and step for now
    # because most of the time we don't have an step / offset ...
    if store == "time":
        grp.create_dataset("step", data=np.arange(1, len(ds) + 1), maxshape=(None,))
    else:
        grp.create_dataset("step", data=1)
    # step_ds.attrs.create(AttributePath.unit.value, "fs")

    if store == "time":
        time_ds = grp.create_dataset(
            "time", data=np.arange(1, len(ds) + 1) * timestep, maxshape=(None,)
        )
    else:
        time_ds = grp.create_dataset("time", data=timestep)
    time_ds.attrs.create(AttributePath.unit.value, "fs")


def extend_group(
    f,
    path,
    entry: Entry,
    ref_length: int,
    store: t.Literal["time", "linear"],
    timestep: float,
) -> None:
    if path not in f:
        raise ValueError(f"Group {path} not found exists")

    grp = f[path]
    data, dtype = entry.dump()

    shift = ref_length - len(grp["value"])
    if dtype == h5py.string_dtype():
        grp["value"].resize((len(grp["value"]) + len(data) + shift,))
        grp["value"][len(grp["value"]) - len(data) :] = data
    else:
        fill_dataset(grp["value"], data, shift, entry.fillvalue)

    if store == "time":
        step_ds = grp["step"]
        step_ds.resize((ref_length,))
        step_ds[:] = np.arange(1, ref_length + 1)
        time_ds = grp["time"]
        time_ds.resize((ref_length,))
        time_ds[:] = np.arange(1, ref_length + 1) * timestep


def extend(self: "IO", data: list[ase.Atoms]) -> None:
    if self.particles_group is None:
        raise ValueError("Particles group not set")

    # TODO: flag to save with origin=None to test against files no coming from znh5md

    frames = Frames.from_ase(data, include=self.include)
    frames.check()

    species_path = get_h5md_path("numbers", self.particles_group, frames)

    with open_file(self.filename, self.file_handle, mode="a") as f:
        if species_path in f:
            ref_length = len(f[species_path]["value"])
        else:
            ref_length = 0

        for entry in frames.items():
            path = get_h5md_path(entry.name, self.particles_group, frames)
            if path in f:
                extend_group(
                    f, path, entry, ref_length, store=self.store, timestep=self.timestep
                )
            else:
                if not self._store_ase_origin:
                    entry.origin = None
                # TODO: what about extending with pbc group false?
                create_group(
                    f,
                    path,
                    entry,
                    ref_length,
                    self.particles_group,
                    pbc_grp=self.pbc_group,
                    store=self.store,
                    save_units=self.save_units,
                    timestep=self.timestep,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    chunk_size=self.chunk_size,
                )
