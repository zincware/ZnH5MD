import dataclasses
import os
import typing
import logging

import ase
import h5py
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ChunkData:
    value: np.ndarray
    step: np.ndarray
    time: np.ndarray

    @property
    def shape(self):
        return tuple(None for _ in range(len(self.value.shape)))

    def __len__(self):
        return len(self.step)


CHUNK_DICT = typing.Dict[str, ChunkData]


@dataclasses.dataclass
class DatabaseWriter:
    filename: str
    atoms_path: str = os.path.join("particles", "atoms")

    def initialize_database_groups(self):
        """Create all groups that are required."""
        with h5py.File(self.filename, "w") as file:
            particles = file.create_group("particles")
            _ = particles.create_group("atoms")

    def create_particles_group_from_chunk_data(self, **kwargs: CHUNK_DICT):
        for group_name, chunk_data in kwargs.items():
            log.debug(f"creating particle groups {group_name}")
            with h5py.File(self.filename, "r+") as file:
                atoms = file[self.atoms_path]
                dataset_group = atoms.create_group(group_name)
                dataset_group.create_dataset(
                    "value", maxshape=chunk_data.shape, data=chunk_data.value, chunks=True
                )
                dataset_group.create_dataset(
                    "time", maxshape=(None,), data=chunk_data.time, chunks=True
                )
                dataset_group.create_dataset(
                    "step", maxshape=(None,), data=chunk_data.step, chunks=True
                )

    def add_chunk_data_to_particles_group(self, **kwargs: CHUNK_DICT):
        for group_name, chunk_data in kwargs.items():
            with h5py.File(self.filename, "r+") as file:
                atoms = file[self.atoms_path]
                dataset_group = atoms[group_name]
                n_current_frames = dataset_group["value"].shape[0]
                n_new_frames = len(chunk_data)

                dataset_group["value"].resize(n_current_frames + n_new_frames, axis=0)
                dataset_group["time"].resize(n_current_frames + n_new_frames, axis=0)
                dataset_group["step"].resize(n_current_frames + n_new_frames, axis=0)
                log.debug(
                    f"Resizing from {n_current_frames} to {n_current_frames+n_new_frames}"
                )

                log.debug(f"appending to particle groups {group_name}")
                dataset_group["value"][:] = np.concatenate(
                    [dataset_group["value"][:n_current_frames], chunk_data.value]
                )
                dataset_group["time"][:] = np.concatenate(
                    [dataset_group["time"][:n_current_frames], chunk_data.time]
                )
                dataset_group["step"][:] = np.concatenate(
                    [dataset_group["step"][:n_current_frames], chunk_data.step]
                )

    def add_chunk_data(self, **kwargs: CHUNK_DICT) -> None:
        """Write Chunks to the database.

        Create a new group, if it does not exist yet.
        Add to existing groups otherwise.

        Parameters
        ----------
        kwargs: dict[str, ChunkData]
            The chunk data to write to the database. The key is the name of the group.
        """
        for group_name, chunk_data in kwargs.items():
            try:
                self.add_chunk_data_to_particles_group(**{group_name: chunk_data})
            except KeyError:
                self.create_particles_group_from_chunk_data(**{group_name: chunk_data})


@dataclasses.dataclass
class MockAtomsReader:
    atoms: list[ase.Atoms]
    frames_per_chunk: int

    def yield_chunks(
        self, group_name: list = None
    ) -> typing.Iterator[typing.Dict[str, ChunkData]]:
        start_index = 0
        stop_index = 0
        if group_name is None:
            group_name = ["position"]

        while stop_index < len(self.atoms):
            stop_index = start_index + self.frames_per_chunk
            data = {}
            for name in group_name:
                if name == "position":
                    value = np.array(
                        [x.get_positions() for x in self.atoms[start_index:stop_index]]
                    )
                elif name == "species":
                    value = np.array(
                        [
                            x.get_atomic_numbers()
                            for x in self.atoms[start_index:stop_index]
                        ]
                    )
                else:
                    raise ValueError(f"Value {name} not supported")
                data[name] = ChunkData(
                    value=value,
                    step=np.arange(start_index, start_index + len(value)),
                    time=np.arange(start_index, start_index + len(value)),
                )
            yield data
            start_index = stop_index
