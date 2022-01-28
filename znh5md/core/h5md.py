import logging
import typing

import h5py
import tensorflow as tf

from znh5md.core.generators import BatchGenerator, BatchSelectionGenerator

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
            only affect performance and memory but not the shape of the generator

        Returns
        -------

        tf.data.DataSet:
            A dataset for the given property / group that e.g. can be batched via
            ds.batch(16) to convert (n_atoms, 3) -> (16, n_atoms, 3) for the first 16
            configurations.
        """
        if isinstance(axis, int):
            if selection is None:
                generator = BatchGenerator(
                    obj=self,
                    shape=self.shape,
                    axis=axis,
                    loop_indices=loop_indices,
                    prefetch=prefetch,
                )
            else:
                generator = BatchSelectionGenerator(
                    obj=self,
                    shape=self.shape,
                    axis=axis,
                    loop_indices=loop_indices,
                    selection=selection,
                    prefetch=prefetch,
                )

            dataset = tf.data.Dataset.from_generator(
                generator.loop,
                output_signature=tf.TensorSpec(
                    shape=generator.loop_shape, dtype=self.dtype
                ),
            )
            if prefetch is not None:
                return dataset.unbatch()
            return dataset
        elif tuple(axis) == (0, 1):
            # WARNING: this is currently very slow and could use some prefetching
            # we could add prefetching to the last dimension
            def generator():
                for config_index in range(self.shape[0]):
                    for species_index in range(self.shape[1]):
                        yield self[config_index, species_index]

            return tf.data.Dataset.from_generator(
                generator,
                output_signature=tf.TensorSpec(shape=self.shape[2:], dtype=self.dtype),
            )

        raise ValueError(f"axis {axis} is not supported.")


class H5MDProperty:
    def __init__(self, *, attribute, group):
        self._attribute = attribute
        self._file = None
        self._group = group

    def __set__(self, instance, value):
        """Can not write to H5MDProperty"""
        raise AttributeError("can't set attribute")

    def __get__(self, instance, owner):
        # This is called before getitem / accessing properties but I'm not sure
        #  if this is the best way to handle that.
        self._file = getattr(instance, self._attribute)
        return self

    def __repr__(self):
        return f"H5MD Property <{self._group}>"

    def __getitem__(self, item):
        # print(item)
        return self.value[item]

    def __len__(self):
        return len(self.value)

    def get_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.zip(
            {
                "step": self.step.get_dataset(),
                "time": self.time.get_dataset(),
                "value": self.value.get_dataset(),
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
        return H5MDGroup(self._file, self._group + "/step")

    @property
    def time(self) -> H5MDGroup:
        return H5MDGroup(self._file, self._group + "/time")

    @property
    def value(self) -> H5MDGroup:
        return H5MDGroup(self._file, self._group + "/value")
