import json
import typing as t

import ase
import h5py
import numpy as np

from znh5md.misc import concatenate_varying_shape_arrays, open_file
from znh5md.path import get_h5md_path
from znh5md.serialization import Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def process_calc_info_arrays(data: list) -> t.Tuple[t.Union[np.ndarray, list], t.Any]:
    if isinstance(data[0], np.ndarray):
        try:
            return concatenate_varying_shape_arrays(data, np.nan), np.float64
        except ValueError:  # could be an unsupported format
            return [json.dumps(v.tolist()) for v in data], h5py.string_dtype()
    elif isinstance(data[0], str):
        return [json.dumps(v) for v in data], h5py.string_dtype()
    elif isinstance(data[0], (int, float)):
        data_ = np.array(data)
        return data_, data_.dtype
    elif isinstance(data[0], dict):
        return [json.dumps(v) for v in data], h5py.string_dtype()
    elif isinstance(data[0], list):
        return [json.dumps(v) for v in data], h5py.string_dtype()
    else:
        raise ValueError(f"Unknown data type: {type(data[0])}")


def create_group(self: "IO", path, data) -> None:
    with open_file(self.filename, self.file_handle, mode="a") as f:
        if path in f:
            raise ValueError(f"Group {path} already exists")
        grp = f.create_group(path)
        data, dtype = process_calc_info_arrays(data)
        # TODO: dtype?
        if dtype == h5py.string_dtype():
            ds = grp.create_dataset("value", data=data, maxshape=(None,))
        else:
            ds = grp.create_dataset(
                "value", data=data, maxshape=(None, *data.shape[1:])
            )


def extend_group(self: "IO", path, data) -> None:
    pass


def extend(self: "IO", data: list[ase.Atoms]) -> None:
    if self.particles_group is None:
        raise ValueError("Particles group not set")

    frames = Frames.from_ase(data)
    frames.check()

    for key, value in frames.items():
        path = get_h5md_path(key, self.particles_group, frames)
        create_group(self, path, value)
