import logging
import typing

import h5py
import tensorflow as tf

from znh5md.core.generators import BatchGenerator
from znh5md.templates.base import H5MDTemplate

log = logging.getLogger(__name__)


class H5MDGroup:
    def __init__(self, file, group):
        self._file = file
        self._group = group

    def __repr__(self):
        return f"H5MD Group <{self._group}>"

    def __getitem__(self, item):
        with h5py.File(self._file) as f:
            return f[self._group][item]

    def __len__(self):
        with h5py.File(self._file) as f:
            return len(f[self._group])

    @property
    def shape(self):
        with h5py.File(self._file) as f:
            return f[self._group].shape

    @property
    def dtype(self):
        with h5py.File(self._file) as f:
            return f[self._group].dtype

    def get_dataset(
        self,
        axis: typing.Union[typing.List, int] = 0,
        selection=None,
        loop_indices=None,
        prefetch: int = None,
        batch_size: int = 1,
    ) -> tf.data.Dataset:
        """Generate a TensorFlow DataSet for the given Property

        Parameters
        ----------
        axis: int | list[int], default = 0
            The main axis for the generator to iterate over.
            The default axis = 0, means that the dataset will iterate over configurations
            and returns shape (n_atoms, 3).
            For axis = 1 it would return (n_configurations, 3)
        selection: list | slice
            Slice along the (first) remaining axis. For selection=[0, 1, 2, 3] the dataset
            will be of shape (4, 3) or in other word will be sliced as x[[0, 1, 2, 3]].
        loop_indices: list
            If given, the iterator will only return those loop indices and skip all others
        prefetch: int
            Prefetch data for faster loading. Without prefetching, the dataset will be
            gathered from the file in sizes of 1 along the dimension to iterate over.
            With prefetching it will load the first #prefetch elements. This will
            only affect performance and memory but not the shape of the generator.
            Prefetch defaults to the batch_size.
        batch_size: int
            The size of the batch to return. For axis=0 this would be (batch, n_atoms, 3)
            where batch is over the dimension of configurations.

        Returns
        -------

        tf.data.DataSet:
            A dataset for the given property / group with shape
            (n_configurations, n_atoms, 3) for most values, e.g. positions/value
            or shape (n_configurations) for time / step
        """
        if prefetch is None:
            prefetch = batch_size

        if isinstance(axis, int):
            generator = BatchGenerator(
                obj=self,
                shape=self.shape,
                axis=axis,
                loop_indices=loop_indices,
                prefetch=prefetch,
                selection=selection,
            )

            dataset = tf.data.Dataset.from_generator(
                generator.loop,
                output_signature=tf.TensorSpec(
                    shape=generator.loop_shape, dtype=self.dtype
                ),
            )

            if prefetch != batch_size:
                if axis == 0:
                    dataset = dataset.unbatch().batch(batch_size)
                elif axis == 1:
                    dataset = dataset.map(lambda x: tf.transpose(x, [1, 0, 2]))
                    dataset = dataset.unbatch().batch(batch_size)
                    dataset = dataset.map(lambda x: tf.transpose(x, [1, 0, 2]))
                elif axis == 2:
                    dataset = dataset.map(lambda x: tf.transpose(x, [2, 0, 1]))
                    dataset = dataset.unbatch().batch(batch_size)
                    dataset = dataset.map(lambda x: tf.transpose(x, [2, 0, 1]))

            return dataset

        elif tuple(axis) == (0, 1):
            log.warning(f"Iterating over {axis} is experimental and can be very slow!")

            def generator():
                for config_index in range(self.shape[0]):
                    for species_index in range(self.shape[1]):
                        yield self[config_index, species_index]

            return tf.data.Dataset.from_generator(
                generator,
                output_signature=tf.TensorSpec(shape=self.shape[2:], dtype=self.dtype),
            ).batch(batch_size)

        raise ValueError(f"axis {axis} is not supported.")


class H5MDProperty:
    def __init__(self, group):
        self._database = None
        self._group = group

    def __set__(self, instance, value):
        """Can not write to H5MDProperty"""
        raise AttributeError("can't set attribute")

    def __get__(self, instance: H5MDTemplate, owner):
        # This is called before getitem / accessing properties, but I'm not sure
        #  if this is the best way to handle that.

        self._database = instance.database
        return self

    def __repr__(self):
        return f"H5MD Property <{self._group}>"

    def __getitem__(self, item):
        return self.value[item]

    def __len__(self):
        return len(self.value)

    def get_dataset(self, **kwargs) -> tf.data.Dataset:
        return tf.data.Dataset.zip(
            {
                "step": self.step.get_dataset(**kwargs),
                "time": self.time.get_dataset(**kwargs),
                "value": self.value.get_dataset(**kwargs),
            }
        )

    @property
    def shape(self) -> tuple:
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def step(self) -> H5MDGroup:
        return H5MDGroup(self._database, self._group + "/step")

    @property
    def time(self) -> H5MDGroup:
        return H5MDGroup(self._database, self._group + "/time")

    @property
    def value(self) -> H5MDGroup:
        return H5MDGroup(self._database, self._group + "/value")
