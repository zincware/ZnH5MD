import typing as t

import ase
import h5py

from znh5md.misc import fill_dataset, open_file
from znh5md.path import AttributePath, get_h5md_path
from znh5md.serialization import Entry, Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def create_group(f, path, entry: Entry, ref_length: int) -> None:
    if path in f:
        raise ValueError(f"Group {path} already exists")
    grp = f.create_group(path)
    data, dtype = entry.dump()
    # TODO: needs shift as well!
    if dtype == h5py.string_dtype():
        ds = grp.create_dataset(
            "value",
            shape=(ref_length + len(data),),
            maxshape=(None,),
            fillvalue=entry.fillvalue,
            dtype=dtype,
        )
        ds[ref_length:] = data
    else:
        maxshape = tuple(None for _ in data.shape)
        shape = (ref_length + len(data),) + data.shape[1:]
        ds = grp.create_dataset(
            "value",
            shape=shape,
            maxshape=maxshape,
            fillvalue=entry.fillvalue,
            dtype=dtype,
        )
        ds[ref_length:] = data
    if entry.origin is not None:
        grp.attrs.create(AttributePath.origin.value, entry.origin)
    if entry.unit is not None:
        grp.attrs.create(AttributePath.unit.value, entry.unit)

    # We use linear time and step for now
    # because most of the time we don't have an step / offset ...
    step_ds = grp.create_dataset("step", data=1)
    time_ds = grp.create_dataset("time", data=1)


def extend_group(f, path, entry: Entry, ref_length: int) -> None:
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


def extend(self: "IO", data: list[ase.Atoms]) -> None:
    if self.particles_group is None:
        raise ValueError("Particles group not set")

    # TODO: flag to save with origin=None to test against files no coming from znh5md

    frames = Frames.from_ase(data)
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
                extend_group(f, path, entry, ref_length)
            else:
                if not self._store_ase_origin:
                    entry.origin = None
                create_group(f, path, entry, ref_length)
