import abc
import dataclasses
import logging
import os
import typing

import h5py
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ExplicitStepTimeChunk:
    """Time-dependent data for a single group.

    References
    ----------
    https://h5md.nongnu.org/h5md.html#time-dependent-data
    """

    value: np.ndarray
    step: np.ndarray
    time: np.ndarray

    @property
    def shape(self) -> tuple:
        """The shape of the value array."""
        return tuple(None for _ in range(len(self.value.shape)))

    def __len__(self):
        """Get the number of frames in the chunk.

        The number of frames is the same as the length of the step / time array
        or the first dimension of the value array.
        """
        return len(self.step)


CHUNK_DICT = typing.Dict[str, ExplicitStepTimeChunk]


class DataReader(abc.ABC):
    """Abstract base class for reading data and yielding chunks."""

    @abc.abstractmethod
    def yield_chunks(
        self, *args, **kwargs
    ) -> typing.Iterator[typing.Dict[str, ExplicitStepTimeChunk]]:
        """Yield chunks of data.

        This method will yield chunks of data to be written to the HDF5 File.
        It should implement a generator pattern that e.g. reads from files.

        Returns
        -------
        typing.Iterator[typing.Dict[str, ExplicitStepTimeChunk]]
            A dictionary of chunks. The key is the name of the group.
            Each chunk containing the data for one group.
        """
        raise NotImplementedError()


def _create_dataset(dataset_group: h5py.Group, chunk_data: ExplicitStepTimeChunk):
    dataset_group.create_dataset(
        "value", maxshape=chunk_data.shape, data=chunk_data.value, chunks=True
    )
    dataset_group.create_dataset(
        "time", maxshape=(None,), data=chunk_data.time, chunks=True
    )
    dataset_group.create_dataset(
        "step", maxshape=(None,), data=chunk_data.step, chunks=True
    )


def _append_dataset(dataset_group: h5py.Group, chunk_data: ExplicitStepTimeChunk):
    n_current_frames = dataset_group["value"].shape[0]
    n_new_frames = len(chunk_data)

    dataset_group["value"].resize(n_current_frames + n_new_frames, axis=0)
    dataset_group["time"].resize(n_current_frames + n_new_frames, axis=0)
    dataset_group["step"].resize(n_current_frames + n_new_frames, axis=0)
    log.debug(f"Resizing from {n_current_frames} to {n_current_frames+n_new_frames}")
    # We also have to reshape value, if the the shape changed in axis=1 (e.g. number of atoms)
    if len(chunk_data.value.shape) > 1:
        if chunk_data.value.shape[1] > dataset_group["value"].shape[1]:
            # we resize the group
            # we fill the new values with Nan
            old_size = dataset_group["value"].shape[1]
            dataset_group["value"].resize(chunk_data.value.shape[1], axis=1)
            dataset_group["value"][:, old_size:] = np.nan
        elif chunk_data.value.shape[1] < dataset_group["value"].shape[1]:
            # we add Nan to the chunk data, because it is smaller than the group
            n_new_particles = dataset_group["value"].shape[1] - chunk_data.value.shape[1]
            fill_shape = list(chunk_data.value.shape)
            fill_shape[1] = n_new_particles
            chunk_data.value = np.concatenate(
                [chunk_data.value, np.full(fill_shape, np.nan)], axis=1
            )

    log.debug(f"appending to particle groups {dataset_group.name}")
    dataset_group["value"][:] = np.concatenate(
        [dataset_group["value"][:n_current_frames], chunk_data.value]
    )
    dataset_group["time"][:] = np.concatenate(
        [dataset_group["time"][:n_current_frames], chunk_data.time]
    )
    dataset_group["step"][:] = np.concatenate(
        [dataset_group["step"][:n_current_frames], chunk_data.step]
    )


@dataclasses.dataclass
class DataWriter:
    filename: str
    atoms_path: str = os.path.join("particles", "atoms")

    def initialize_database_groups(self):
        """Create all groups that are required.

        We create the following groups:
        - particles/atoms
        """
        with h5py.File(self.filename, "w") as file:
            particles = file.create_group("particles")
            _ = particles.create_group("atoms")

    def _handle_special_cases_group_names(self, groupname: str) -> str:
        """Update group name in special cases.

        Some groups, especially the box group, are nested differently.
        """
        if groupname in ["boundary", "edges"]:
            return f"box/{groupname}"

        return groupname

    def create_particles_group_from_chunk_data(self, **kwargs: CHUNK_DICT):
        """Create a new group for the given elements.

        This method will create the following datasets for each group in kwargs.
        - particles/atoms/<group_name>/value
        - particles/atoms/<group_name>/time
        - particles/atoms/<group_name>/step

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.
        """
        for group_name, chunk_data in kwargs.items():
            log.debug(f"creating particle groups {group_name}")
            with h5py.File(self.filename, "r+") as file:
                atoms = file[self.atoms_path]
                group_name = self._handle_special_cases_group_names(group_name)
                dataset_group = atoms.create_group(group_name)
                _create_dataset(dataset_group, chunk_data)

    def add_chunk_data_to_particles_group(self, **kwargs: CHUNK_DICT):
        """Add data to an existing group.

        For each group in kwargs, the following datasets are resized and appended to:
        - particles/atoms/<group_name>/value
        - particles/atoms/<group_name>/time
        - particles/atoms/<group_name>/step

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.
            The group must already exist.
        """
        for group_name, chunk_data in kwargs.items():
            with h5py.File(self.filename, "r+") as file:
                atoms = file[self.atoms_path]
                group_name = self._handle_special_cases_group_names(group_name)
                dataset_group = atoms[group_name]

                _append_dataset(dataset_group, chunk_data)

    def add_chunk_data(self, **kwargs: CHUNK_DICT) -> None:
        """Write Chunks to the database.

        Create a new group, if it does not exist yet.
        Add to existing groups otherwise.

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.
        """
        for group_name, chunk_data in kwargs.items():
            try:
                self.add_chunk_data_to_particles_group(**{group_name: chunk_data})
            except KeyError:
                self.create_particles_group_from_chunk_data(**{group_name: chunk_data})

    def add(self, reader: DataReader):
        """Add data from a reader to the HDF5 file."""
        for chunk in reader.yield_chunks():
            self.add_chunk_data(**chunk)
