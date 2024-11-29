import json
import typing as t

import ase
import numpy as np
from ase.calculators.calculator import all_properties

from znh5md.misc import decompose_varying_shape_arrays, open_file
from znh5md.path import AttributePath, H5MDToASEMapping
from znh5md.serialization import MISSING, ORIGIN_TYPE, Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def update_frames(
    self, name: str, value: np.ndarray, origin: ORIGIN_TYPE, use_ase_calc: bool
) -> None:
    if name in ["positions", "numbers", "pbc", "cell"]:
        setattr(self, name, decompose_varying_shape_arrays(value, np.nan))
    else:
        if value.dtype.kind in ["O", "S", "U"]:
            data = [json.loads(v) if v != b"" else MISSING for v in value]
        else:
            data = decompose_varying_shape_arrays(value, np.nan)
            data = [x if not np.all(np.isnan(x)) else MISSING for x in data]

        if origin is not None and use_ase_calc:
            if origin == "calc":
                self.calc[name] = data
            elif origin == "info":
                self.info[name] = data
            elif origin == "arrays":
                self.arrays[name] = data
            elif origin == "atoms":
                raise ValueError(f"Origin 'atoms' is not allowed for {name}")
            else:
                raise ValueError(f"Unknown origin: {origin}")
        else:
            if name in all_properties:
                if use_ase_calc:
                    self.calc[name] = data
                else:
                    if not isinstance(data[0], (float, int, bool, dict, str)) and len(
                        data[0]
                    ) == len(self.numbers[0]):
                        self.arrays[name] = data
                    else:
                        self.info[name] = data
            else:
                if not isinstance(data[0], (float, int, bool, dict, str)) and len(
                    data[0]
                ) == len(self.numbers[0]):
                    self.arrays[name] = data
                else:
                    self.info[name] = data


def getitem(
    self: "IO", index: int | np.int_ | slice | np.ndarray | list[int]
) -> ase.Atoms | list[ase.Atoms]:
    frames = Frames()
    is_single_item = False
    if isinstance(index, int):
        is_single_item = True
        index = [index]

    with open_file(self.filename, self.file_handle, mode="r") as f:
        particles = f[f"/particles/{self.particles_group}"]
        # first do species then the rest so we know the length of the arrays
        #  for sorting into arrays, info, calc
        grp = particles["species/value"]
        update_frames(
            frames, H5MDToASEMapping.species.value, grp[index], None, self.use_ase_calc
        )

        for grp_name in particles:
            if grp_name == "species":
                continue
            grp = particles[grp_name]  # Access the subgroup or dataset
            origin = grp.attrs.get(AttributePath.origin.value, None)
            if grp_name == "box":
                update_frames(
                    frames, "cell", grp["edges/value"][index], origin, self.use_ase_calc
                )
                try:
                    update_frames(
                        frames,
                        "pbc",
                        grp["pbc/value"][index],
                        origin,
                        self.use_ase_calc,
                    )
                except KeyError:
                    pbc = grp.attrs.get(AttributePath.boundary.value, ["none"] * 3)
                    pbc = np.array([b != "none" for b in pbc], dtype=bool)
                    frames.pbc = np.array([pbc] * len(frames))
            else:
                try:
                    try:
                        update_frames(
                            frames,
                            H5MDToASEMapping[grp_name].value,
                            grp["value"][index],
                            origin,
                            self.use_ase_calc,
                        )
                    except KeyError:
                        update_frames(
                            frames,
                            grp_name,
                            grp["value"][index],
                            origin,
                            self.use_ase_calc,
                        )
                    except (OSError, IndexError):
                        pass  # values must not be backfilled to the length of the species
                except KeyError:
                    raise KeyError(
                        f"Key '{grp_name}' does not seem to be a valid H5MD group - missing 'value' dataset."
                    )

        if f"/observables/{self.particles_group}" in f:
            observables = f[f"/observables/{self.particles_group}"]
            for grp_name in observables:
                grp = observables[grp_name]
                origin = grp.attrs.get(AttributePath.origin.value, None)
                try:
                    try:
                        update_frames(
                            frames,
                            H5MDToASEMapping[grp_name].value,
                            grp["value"][index],
                            origin,
                            self.use_ase_calc,
                        )
                    except KeyError:
                        update_frames(
                            frames,
                            grp_name,
                            grp["value"][index],
                            origin,
                            self.use_ase_calc,
                        )
                    except (OSError, IndexError):
                        pass  # values must not be backfilled to the length of the species
                except KeyError:
                    raise KeyError(
                        f"Key '{grp_name}' does not seem to be a valid H5MD group - missing 'value' dataset."
                    )

    return list(frames) if not is_single_item else frames[0]
