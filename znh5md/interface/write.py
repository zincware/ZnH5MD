import typing as t

import ase
import h5py

from znh5md.misc import open_file
from znh5md.path import AttributePath, get_h5md_path
from znh5md.serialization import Entry, Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def create_group(f, path, entry: Entry) -> None:
    if path in f:
        raise ValueError(f"Group {path} already exists")
    grp = f.create_group(path)
    data, dtype = entry.dump()
    if dtype == h5py.string_dtype():
        grp.create_dataset("value", data=data, maxshape=(None,))
    else:
        grp.create_dataset("value", data=data, maxshape=(None, *data.shape[1:]))
    if entry.origin is not None:
        grp.attrs.create(AttributePath.origin.value, entry.origin)
    if entry.unit is not None:
        grp.attrs.create(AttributePath.unit.value, entry.unit)

    # We use linear time and step for now
    # because most of the time we don't have an step / offset ...
    step_ds = grp.create_dataset("step", data=1)
    time_ds = grp.create_dataset("time", data=1)


def extend_group(self: "IO", path, data) -> None:
    raise NotImplementedError("extend existing groups not implemented yet")


def extend(self: "IO", data: list[ase.Atoms]) -> None:
    if self.particles_group is None:
        raise ValueError("Particles group not set")

    # TODO: flag to save with origin=None to test against files no coming from znh5md

    frames = Frames.from_ase(data)
    frames.check()
    with open_file(self.filename, self.file_handle, mode="a") as f:
        for entry in frames.items():
            path = get_h5md_path(entry.name, self.particles_group, frames)
            if path in f:
                extend_group(self, path, entry)
            else:
                if not self._store_ase_origin:
                    entry.origin = None
                create_group(f, path, entry)
