import typing as t

import ase
import numpy as np

from znh5md.misc import open_file
from znh5md.path import H5MDToASEMapping
from znh5md.serialization import Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def getitem(
    self: "IO", index: int | np.int_ | slice | np.ndarray
) -> ase.Atoms | list[ase.Atoms]:
    frames = Frames()

    with open_file(self.filename, self.file_handle, mode="r") as f:
        particles = f[f"/particles/{self.particles_group}"]
        # first do species then the rest so we know the length of the arrays
        #  for sorting into arrays, info, calc
        grp = particles["species/value"]
        frames.set(H5MDToASEMapping.species.value, grp[index])

        for grp_name in particles:
            if grp_name == "species":
                continue
            grp = particles[grp_name]  # Access the subgroup or dataset
            if grp_name == "box":
                frames.set("cell", grp["edges/value"][index])
                frames.set("pbc", grp["pbc/value"][index])
            else:
                try:
                    try:
                        frames.set(
                            H5MDToASEMapping[grp_name].value, grp["value"][index]
                        )
                    except KeyError:
                        frames.set(grp_name, grp["value"][index])
                except KeyError:
                    raise KeyError(
                        f"Key '{grp_name}' does not seem to be a valid H5MD group - missing 'value' dataset."
                    )

        observables = f[f"/observables/{self.particles_group}"]
        for grp_name in observables:
            grp = observables[grp_name]
            try:
                try:
                    frames.set(H5MDToASEMapping[grp_name].value, grp["value"][index])
                except KeyError:
                    frames.set(grp_name, grp["value"][index])
            except KeyError:
                raise KeyError(
                    f"Key '{grp_name}' does not seem to be a valid H5MD group - missing 'value' dataset."
                )

    return frames
