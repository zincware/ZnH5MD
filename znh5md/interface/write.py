import json
import typing as t

import ase
import h5py
import numpy as np

from znh5md.misc import concatenate_varying_shape_arrays, open_file
from znh5md.path import AttributePath, get_h5md_path
from znh5md.serialization import Entry, Frames, MISSING

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def process_calc_info_arrays(data: list) -> t.Tuple[t.Union[np.ndarray, list], t.Any]:
    # TODO: handle MISSING
    ref = None
    for v in data:
        if v is not MISSING:
            ref = v
            break
    else:
        raise ValueError("All values are MISSING")

    if isinstance(ref, np.ndarray):
        if ref.dtype.kind in ["O", "S", "U"]:
            return [json.dumps(v.tolist()) if v is not MISSING else "" for v in data], h5py.string_dtype()
        else:
            data = [np.array(v) if v is not MISSING else np.full_like(ref, np.nan) for v in data]
            return concatenate_varying_shape_arrays(data, np.nan), np.float64
    elif isinstance(ref, str):
        return [json.dumps(v) if v is not MISSING else "" for v in data], h5py.string_dtype()
    elif isinstance(ref, (int, float)):
        data_ = np.array([v if v is not MISSING else np.nan for v in data])
        return data_, data_.dtype
    elif isinstance(ref, dict):
        return [json.dumps(v) if v is not MISSING else "" for v in data], h5py.string_dtype()
    elif isinstance(ref, list):
        return [json.dumps(v) if v is not MISSING else "" for v in data], h5py.string_dtype()
    else:
        raise ValueError(f"Unknown data type: {type(data[0])}")


def create_group(f, path, entry: Entry) -> None:
    if path in f:
        raise ValueError(f"Group {path} already exists")
    grp = f.create_group(path)
    data, dtype = process_calc_info_arrays(entry.value)
    if dtype == h5py.string_dtype():
        grp.create_dataset("value", data=data, maxshape=(None,))
    else:
        grp.create_dataset("value", data=data, maxshape=(None, *data.shape[1:]))
    if entry.origin is not None:
        grp.attrs.create(AttributePath.origin.value, entry.origin)
    if entry.unit is not None:
        grp.attrs.create(AttributePath.unit.value, entry.unit)


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
