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
    def __init__(self, obj, shape, axis, loop_indices, prefetch, selection):
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
        axis_selection = [slice(None) for _ in range(self.axis)]
        for chunk in chunk_gen(self.loop_indices, chunk_size=self.prefetch):
            # TODO slices are faster than lists to index - support both!
            slice_ = tuple(axis_selection + [chunk])
            if self.axis == 0 and len(self.shape) > 1:
                yield self.obj[slice_][:, self.selection]
            else:
                yield self.obj[slice_][self.selection]

    @property
    def loop_shape(self) -> tuple:
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
