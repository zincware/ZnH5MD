import dataclasses
import functools

import dask.array
import h5py

from znh5md.znh5md.base import H5MDBase


@functools.singledispatch
def get_mask(species: int, data):
    return species == data


@get_mask.register
def _(species: list, data):
    mask = get_mask(species[0], data)
    for x in species[1:]:
        mask += get_mask(x, data)
    return mask


@dataclasses.dataclass
class DaskDataSet:
    value: dask.array
    time: dask.array
    step: dask.array
    species: dask.array

    @classmethod
    def from_file(cls, item: h5py.Dataset, species, value_chunks, time_chunks):
        value = dask.array.from_array(item["value"], chunks=value_chunks)
        time = dask.array.from_array(item["time"], chunks=time_chunks)
        step = dask.array.from_array(item["step"], chunks=time_chunks)
        species = dask.array.from_array(species["value"], chunks=time_chunks)

        if time.shape == ():
            time = dask.array.arange(0, len(value) * time, time)

        if step.shape == ():
            step = dask.array.arange(0, len(value) * step, step)

        return cls(value=value, time=time, step=step, species=species)

    @classmethod
    def from_values(cls, value, time, step, species):
        return cls(value=value, time=time, step=step, species=species)

    def slice_by_species(self, species) -> "DaskDataSet":
        species_flattened = self.species.reshape(-1)
        # species_mask = sum(species_flattened == x for x in species)
        value_flat = self.value.reshape((-1,) + self.value.shape[2:])
        sliced = value_flat[get_mask(species, species_flattened)]
        value = sliced.compute_chunk_sizes().reshape(
            (self.value.shape[0],) + (-1,) + self.value.shape[2:]
        )

        species = (
            species_flattened[get_mask(species, species_flattened)]
            .compute_chunk_sizes()
            .reshape((self.species.shape[0], -1))
        )

        return self.from_values(
            value=value, time=self.time, step=self.step, species=species
        )

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)) and isinstance(item[0], slice):
            return self.from_values(
                value=self.value[item],
                time=self.time[item[0]],
                step=self.step[item[0]],
                species=self.species[item[:2]],
            )
        return self.from_values(
            value=self.value[item],
            time=self.time[item],
            step=self.step[item],
            species=self.species[item],
        )

    def __len__(self) -> int:
        return len(self.value)

    def batch(self, size, axis):
        start = 0
        while start < self.value.shape[axis]:
            if axis == 0:
                yield self[start : start + size]
            elif axis == 1:
                yield self[:, start : start + size]
            elif axis == 2:
                yield self[:, :, start : start + size]
            else:
                raise ValueError(
                    f"axis must be in (0, 1, 2). 'axis={axis}' is currently not"
                    " supported."
                )
            start += size

    @property
    def shape(self):
        return self.value.shape


@dataclasses.dataclass
class DaskDataSet:
    value: dask.array
    time: dask.array
    step: dask.array
    species: dask.array

    @classmethod
    def from_file(cls, item: h5py.Dataset, species, value_chunks, time_chunks):
        value = dask.array.from_array(item["value"], chunks=value_chunks)
        time = dask.array.from_array(item["time"], chunks=time_chunks)
        step = dask.array.from_array(item["step"], chunks=time_chunks)
        species = dask.array.from_array(species["value"], chunks=time_chunks)

        if time.shape == ():
            time = dask.array.arange(0, len(value) * time, time)

        if step.shape == ():
            step = dask.array.arange(0, len(value) * step, step)

        return cls(value=value, time=time, step=step, species=species)

    @classmethod
    def from_values(cls, value, time, step, species):
        return cls(value=value, time=time, step=step, species=species)

    def slice_by_species(self, species) -> "DaskDataSet":
        species_flattened = self.species.reshape(-1)
        # species_mask = sum(species_flattened == x for x in species)
        value_flat = self.value.reshape((-1,) + self.value.shape[2:])
        sliced = value_flat[get_mask(species, species_flattened)]
        value = sliced.compute_chunk_sizes().reshape(
            (self.value.shape[0],) + (-1,) + self.value.shape[2:]
        )

        species = (
            species_flattened[get_mask(species, species_flattened)]
            .compute_chunk_sizes()
            .reshape((self.species.shape[0], -1))
        )

        return self.from_values(
            value=value, time=self.time, step=self.step, species=species
        )

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)) and isinstance(item[0], slice):
            return self.from_values(
                value=self.value[item],
                time=self.time[item[0]],
                step=self.step[item[0]],
                species=self.species[item[:2]],
            )
        return self.from_values(
            value=self.value[item],
            time=self.time[item],
            step=self.step[item],
            species=self.species[item],
        )

    def __len__(self) -> int:
        return len(self.value)

    def batch(self, size, axis):
        start = 0
        while start < self.value.shape[axis]:
            if axis == 0:
                yield self[start : start + size]
            elif axis == 1:
                yield self[:, start : start + size]
            elif axis == 2:
                yield self[:, :, start : start + size]
            else:
                raise ValueError(
                    f"axis must be in (0, 1, 2). 'axis={axis}' is currently not"
                    " supported."
                )
            start += size

    @property
    def shape(self):
        return self.value.shape


@dataclasses.dataclass
class DaskH5MD(H5MDBase):
    """Dask interface for H5MD files.

    Attributes
    ----------
    fixed_species_index: bool, default=False
        if the species indices are fixed in time, a faster way of slicing
        by species is available and can be selected.

    """

    time_chunk_size: int = 10
    species_chunk_size: int = 10
    fixed_species_index: bool = False

    @property
    def position(self) -> DaskDataSet:
        return DaskDataSet.from_file(
            item=self.format_handler.position,
            species=self.format_handler.species,
            value_chunks=(self.time_chunk_size, self.species_chunk_size, 3),
            time_chunks=self.time_chunk_size,
        )

    @property
    def species(self) -> DaskDataSet:
        return DaskDataSet.from_file(
            item=self.format_handler.species,
            species=self.format_handler.species,
            value_chunks=(self.time_chunk_size, self.species_chunk_size),
            time_chunks=self.time_chunk_size,
        )

    def __getattr__(self, item):
        return DaskDataSet.from_file(
            item=getattr(self.format_handler, item),
            species=self.format_handler.species,
            value_chunks="auto",
            time_chunks="auto",
        )
