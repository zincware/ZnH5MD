"""Format handler for h5md files."""
import dataclasses
import typing

import h5py


@dataclasses.dataclass
class FormatHandler:
    filename: str

    file_handle: typing.Callable = h5py.File

    def __post_init__(self):
        self.particle_key = None
        with self.file as file:
            for particle_key in ["all", "atoms"]:
                if particle_key in file["particles"]:
                    # TODO what if all and atoms appear?
                    self.particle_key = particle_key
        if self.particle_key is None:
            raise ValueError("Could not determine required key '/particles/<...>'")

    @property
    def file(self) -> h5py.File:
        """The 'h5py.File' from filename."""
        return self.file_handle(self.filename)

    @property
    def time_dependent_groups(self) -> typing.List[str]:
        """All time dependent groups.

        References
        ----------
        https://www.nongnu.org/h5md/h5md.html#time-dependent-data

        """
        return list(self.file[f"particles/{self.particle_key}"])

    def __getattr__(self, item):
        group = f"particles/{self.particle_key}/{item}"
        try:
            return self.file[group]
        except KeyError as err:
            raise AttributeError(
                f"Could not load group '{group}' from '{self.filename}'"
            ) from err

    @property
    def box(self):
        group = f"particles/{self.particle_key}/box/edges"
        return self.file[group]
