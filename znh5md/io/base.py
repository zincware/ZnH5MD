import abc
import dataclasses
import logging
import pathlib
import typing

import h5py
import numpy as np

log = logging.getLogger(__name__)

import typing_extensions as te

from znh5md.format import GRP, PARTICLES_GRP


@dataclasses.dataclass
class StepTimeChunk:
    """Abstract class for time-dependent data for a single group.

    Parameters
    ----------
    value : np.ndarray
        The value, to be stored in the value dataset.
    step : np.ndarray
        The step, to be stored in the step dataset.
    time : np.ndarray
        The time, to be stored in the time dataset.

    References
    ----------
    https://h5md.nongnu.org/h5md.html#time-dependent-data

    """

    value: np.ndarray
    step: np.ndarray
    time: np.ndarray

    value_units: str = None
    time_units: str = None

    @property
    def shape(self) -> tuple:
        """The shape of the value array."""
        return tuple(None for _ in range(len(self.value.shape)))

    def __len__(self):
        """Get the number of frames in the chunk."""
        return len(self.value)

    def create_dataset(self, dataset_group: h5py.Group):
        """Create the datasets for the chunk."""
        raise NotImplementedError

    def append_to_dataset(self, dataset_group: h5py.Group):
        """Append the data to the dataset."""
        raise NotImplementedError

    def _resize_dataset_group(self, dataset_group: h5py.Group, fill_value):
        """Resize the dataset_group."""
        old_size = dataset_group["value"].shape[1]
        dataset_group["value"].resize(self.value.shape[1], axis=1)
        dataset_group["value"][:, old_size:] = fill_value

    def _resize_value(self, n_new_particles: int, fill_value):
        """Resize the value array."""
        fill_shape = list(self.value.shape)
        fill_shape[1] = n_new_particles
        self.value = np.concatenate(
            [self.value, np.full(fill_shape, fill_value)], axis=1
        )

    def resize_by_particle_count(self, dataset_group: h5py.Group, fill_value=np.nan):
        # We also have to reshape value, if the the shape
        #  changed in axis=1 (e.g. number of atoms)
        if len(self.value.shape) > 1:
            if self.value.shape[1] > dataset_group["value"].shape[1]:
                # we resize the group
                # we fill the new values with Nan
                self._resize_dataset_group(dataset_group, fill_value)

            elif self.value.shape[1] < dataset_group["value"].shape[1]:
                # we add Nan to the chunk data, because it is smaller than the group
                n_new_particles = dataset_group["value"].shape[1] - self.value.shape[1]
                self._resize_value(n_new_particles, fill_value)


@dataclasses.dataclass
class ExplicitStepTimeChunk(StepTimeChunk):
    """Same as StepTimeChunk, but with explicit step and time."""

    def create_dataset(self, dataset_group: h5py.Group):
        """Create the datasets for the chunk."""
        value_ds = dataset_group.create_dataset(
            "value", maxshape=self.shape, data=self.value, chunks=True
        )
        time_ds = dataset_group.create_dataset(
            "time", maxshape=(None,), data=self.time, chunks=True
        )
        dataset_group.create_dataset(
            "step", maxshape=(None,), data=self.step, chunks=True
        )

        if self.value_units is not None:
            value_ds.attrs["unit"] = self.value_units
        if self.time_units is not None:
            time_ds.attrs["unit"] = self.time_units

    def append_to_dataset(self, dataset_group: h5py.Group):
        n_current_frames = dataset_group["value"].shape[0]

        self.resize_by_particle_count(dataset_group)
        for key in ("value", "time", "step"):
            dataset_group[key].resize(n_current_frames + len(self), axis=0)
            dataset_group[key][:] = np.concatenate(
                [dataset_group[key][:n_current_frames], self.value]
            )


@dataclasses.dataclass
class FixedStepTimeChunk(StepTimeChunk):
    """Same as StepTimeChunk, but with fixed step and time."""

    step: int
    time: float

    def create_dataset(self, dataset_group: h5py.Group):
        """Create the datasets for the chunk."""
        value_ds = dataset_group.create_dataset(
            "value", maxshape=self.shape, data=self.value, chunks=True
        )
        time_ds = dataset_group.create_dataset(
            "time",
            data=np.arange(len(self.value)) * self.time,
            chunks=True,
            maxshape=(None,),
        )
        dataset_group.create_dataset(
            "step", data=np.arange(len(self.value)), chunks=True, maxshape=(None,)
        )

        if self.value_units is not None:
            value_ds.attrs["unit"] = self.value_units
        if self.time_units is not None:
            time_ds.attrs["unit"] = self.time_units

    def append_to_dataset(self, dataset_group: h5py.Group):
        n_current_frames = dataset_group["value"].shape[0]

        self.resize_by_particle_count(dataset_group)
        dataset_group["value"].resize(n_current_frames + len(self), axis=0)
        dataset_group["value"][:] = np.concatenate(
            [dataset_group["value"][:n_current_frames], self.value]
        )
        # append to time and step as well
        dataset_group["time"].resize(n_current_frames + len(self), axis=0)
        dataset_group["time"][:] = np.concatenate(
            [
                dataset_group["time"][:n_current_frames],
                np.arange(len(self.value)) * self.time + n_current_frames * self.time,
            ]
        )
        dataset_group["step"].resize(n_current_frames + len(self), axis=0)
        dataset_group["step"][:] = np.concatenate(
            [
                dataset_group["step"][:n_current_frames],
                np.arange(len(self.value)) + n_current_frames,
            ]
        )


CHUNK_DICT = typing.Dict[str, ExplicitStepTimeChunk]


class DataReader(abc.ABC):
    """Abstract base class for reading data and yielding chunks."""

    @abc.abstractmethod
    def yield_chunks(
        self, *args, **kwargs
    ) -> typing.Iterator[typing.Dict[str, StepTimeChunk]]:
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

    def _fill_with_nan(self, data: list) -> np.ndarray:
        max_n_particles = max(x.shape[0] for x in data)
        dimensions = data[0].shape[1:]

        result = np.full((len(data), max_n_particles, *dimensions), np.nan)
        for i, x in enumerate(data):
            result[i, : x.shape[0], ...] = x
        return result


@dataclasses.dataclass
class DataWriter:
    filename: str
    particles_path: str = "particles/atoms"
    observables_path: str = "observables/atoms"

    @te.deprecated("DB will be initialized automatically when adding data.")
    def initialize_database_groups(self):
        """Create all groups that are required.

        We create the following groups:
        - particles/atoms
        """
        pass

    def _handle_special_cases_group_names(self, groupname: str) -> str:
        """Update group name in special cases.

        Some groups, especially the box group, are nested differently.
        """
        if groupname in [GRP.boundary, GRP.edges, GRP.pbc]:
            return f"box/{groupname}"

        return groupname

    def create_group(self, db_path, group_name, chunk_data):
        """Create a new group for the given elements.

        This method will create the following datasets for each group in kwargs.
        - <db_path>/<group_name>/value
        - <db_path>/<group_name>/time
        - <db_path>/<group_name>/step

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.

        """
        group_name = self._handle_special_cases_group_names(group_name)
        dataset_group = db_path.create_group(group_name)
        chunk_data.create_dataset(dataset_group)

    def add_data_to_group(self, db_path, group_name, chunk_data: StepTimeChunk):
        """Add data to an existing group.

        For each group in kwargs, the following datasets are resized and appended to:
        - <db_path>/<group_name>/value
        - <db_path>/<group_name>/time
        - <db_path>/<group_name>/step

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.
            The group must already exist.

        """
        group_name = self._handle_special_cases_group_names(group_name)
        dataset_group = db_path[group_name]
        chunk_data.append_to_dataset(dataset_group)

    def handle_boundary(self, file, chunk_data):
        """Special case for the box boundary.

        This requires a special case, because the boundary is inside
        the box group.
        """
        if GRP.boundary not in file[f"{self.particles_path}/box"]:
            atoms = file[self.particles_path]
            # we create the box group
            atoms.create_dataset(f"box/{GRP.boundary}", data=chunk_data.value)
            # dimension group is required by H5MD
            atoms.create_dataset(f"box/{GRP.dimension}", data=len(chunk_data.value))

            atoms["box"].attrs["boundary"] = ["periodic", "periodic", "periodic"]
            atoms["box"].attrs["dimension"] = 3

    def add_chunk_data(self, **kwargs: CHUNK_DICT) -> None:
        """Write Chunks to the database.

        Create a new group, if it does not exist yet.
        Add to existing groups otherwise.

        Parameters
        ----------
        kwargs: dict[str, ExplicitStepTimeChunk]
            The chunk data to write to the database. The key is the name of the group.

        """
        if not pathlib.Path(self.filename).exists():
            with h5py.File(self.filename, "w") as f:
                # create the file
                h5md = f.create_group("h5md")
                h5md.attrs["version"] = np.array([1, 1])
                author = h5md.create_group("author")
                author.attrs["name"] = "N/A"  # TODO: make configurable
                creator = h5md.create_group("creator")
                creator.attrs["name"] = "ZnH5MD"

        with h5py.File(self.filename, "r+") as file:
            for group_name, chunk_data in kwargs.items():
                if group_name == GRP.boundary:
                    self.handle_boundary(file, chunk_data)
                else:
                    if group_name in PARTICLES_GRP:
                        try:
                            group_path = file[self.particles_path]
                        except KeyError:
                            log.debug(f"creating particle groups {group_name}")
                            group_path = file.create_group(self.particles_path)
                    else:
                        try:
                            group_path = file[self.observables_path]
                        except KeyError:
                            log.debug(f"creating observable groups {group_name}")
                            group_path = file.create_group(self.observables_path)

                    try:
                        self.add_data_to_group(group_path, group_name, chunk_data)
                    except KeyError:
                        log.debug(f"creating particle groups {group_name}")

                        self.create_group(group_path, group_name, chunk_data)

    def add(self, reader: DataReader):
        """Add data from a reader to the HDF5 file."""
        for chunk in reader.yield_chunks():
            self.add_chunk_data(**chunk)
