"""ZnH5MD: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
"""


def chunk_gen(lst, chunk_size):
    """Yield successive chunk_sized chunks from lst.

    References
    ----------
    https://stackoverflow.com/a/312464/10504481
    """
    # the zero is important!
    for idx in range(0, len(lst), chunk_size):
        yield lst[idx : idx + chunk_size]


class BatchGenerator:
    """Generator Object for tf.data.Dataset

    This Generator provides access to the loop method, axis and selection methods
    as well as the resulting shape
    """

    def __init__(
        self, obj, shape: tuple, axis, prefetch, loop_indices=None, selection=None
    ):
        """Constructor for the BatchGenerator

        Parameters
        ----------
        obj: any object that implements __getitem__.
            Designed for the H5MDGroup
        shape: tuple
            Shape of the obj array
        axis: int
            The axis of the obj to loop over
        prefetch: int
            The size along axis to load. This is the batch_size of the generator object
        loop_indices: list, optional
            A list of indices to restrict the generator to. If provided the generator will
            only loop over this list along the axis dimension.
        selection: list|slice, optional
            The yielded object x will be further sliced to x[selection] if provided.
        """
        self.obj = obj
        self.shape = shape
        self.axis = axis
        self.loop_indices = loop_indices
        self.prefetch = prefetch
        self.selection = selection
        if self.selection is None:
            self.selection = slice(None)

        if self.loop_indices is None:
            self.loop_indices = list(range(self.shape[self.axis]))

        if prefetch > self.shape[self.axis]:
            raise ValueError(
                f"Can't prefetch ({prefetch}) more items "
                f"than available {self.shape[self.axis]}."
            )

    def loop(self):
        """Method to loop over in tf.data.Dataset"""
        axis_selection = [slice(None) for _ in range(self.axis)]
        for chunk in chunk_gen(self.loop_indices, chunk_size=self.prefetch):
            # TODO slices are faster than lists to index - support both!
            slice_ = tuple(axis_selection + [chunk])
            # TODO Do not gather the full object and then apply selection but add the
            #  selection to slice_ for performance and memory efficiency
            if self.axis == 0 and len(self.shape) > 1:
                yield self.obj[slice_][:, self.selection]
            else:
                yield self.obj[slice_][self.selection]

    @property
    def loop_shape(self) -> tuple:
        """Provide the shape of the array yielded by the loop method.

        If the size of a dimension is unknown it will be set to None. The axis-dimension
        will be set to None.
        """
        loop_shape = list(self.shape)
        loop_shape[self.axis] = None
        if self.selection == slice(None):
            return tuple(loop_shape)
        for idx, val in enumerate(loop_shape):
            if val is not None:
                try:
                    loop_shape[idx] = len(self.selection)
                except TypeError:
                    loop_shape[idx] = None
                break
        # # TODO consider using shape of selection and support 2d slicing [slice, slice]
        return tuple(loop_shape)
