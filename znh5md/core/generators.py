def chunk_gen(lst, chunk_size):
    """Yield successive chunk_sized chunks from lst.

    References
    ----------
    https://stackoverflow.com/a/312464/10504481
    """
    # the zero is important!
    for idx in range(0, len(lst), chunk_size):
        yield lst[idx : idx + chunk_size]


class GeneratorBase:
    def __init__(self, obj, shape, axis, loop_indices, prefetch):
        self.obj = obj
        self.shape = shape
        self.axis = axis
        self.loop_indices = loop_indices
        self.prefetch = prefetch

        if loop_indices is not None and prefetch is not None:
            raise ValueError("Can not use batched loading with loop_indices")

    def loop(self):
        raise NotImplementedError

    @property
    def loop_shape(self):
        loop_shape = [x for i, x in enumerate(self.shape) if i != self.axis]
        if self.prefetch is None:
            return loop_shape
        return [None] + loop_shape


class BatchGenerator(GeneratorBase):
    def loop(self):
        axis_selection = [slice(None) for _ in range(self.axis)]
        if self.prefetch is not None:
            for chunk in chunk_gen(
                list(range(self.shape[self.axis])), chunk_size=self.prefetch
            ):
                # TODO slices are faster than lists to index - support both!
                slice_ = tuple(axis_selection + [chunk])
                yield self.obj[slice_]

        elif self.loop_indices is not None:
            for index in range(self.shape[self.axis]):
                if index not in self.loop_indices:
                    continue
                slice_ = tuple(axis_selection + [index])
                yield self.obj[slice_]
        else:
            # TODO remove in favour of using prefetch=1
            for index in range(self.shape[self.axis]):
                slice_ = tuple(axis_selection + [index])
                yield self.obj[slice_]


class BatchSelectionGenerator(BatchGenerator):
    def __init__(self, obj, shape, axis, loop_indices, prefetch, selection):
        super().__init__(obj, shape, axis, loop_indices, prefetch)
        self.selection = selection

    def loop(self):
        if self.prefetch is not None:
            for obj in super().loop():
                yield obj[:, self.selection]
        else:
            for obj in super().loop():
                yield obj[self.selection]

    @property
    def loop_shape(self):
        loop_shape = super().loop_shape
        has_zero_axis = int(self.prefetch is not None)
        # when batching we need to account for an extra axis before unbatching
        # TODO consider using shape of selection and support 2d slicing [slice, slice]
        try:
            loop_shape[has_zero_axis] = len(self.selection)
        except TypeError:
            # for example when selection is a slice
            loop_shape[has_zero_axis] = None
        return loop_shape
