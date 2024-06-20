import ase

from .io import IO


def read(filename) -> ase.Atoms:
    io = IO(filename)
    return io[:]


def write(filename, images) -> None:
    io = IO(filename)
    io.extend(images)
