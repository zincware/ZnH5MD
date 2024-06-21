import pathlib
import typing as t

import ase

from .io import IO


def read(filename, index: int | slice | list[int] = -1) -> ase.Atoms:
    io = IO(filename)
    return io[index]


def write(filename, images, append: bool = True) -> None:
    if not append and pathlib.Path(filename).exists():
        raise FileExistsError(
            f"{filename} already exists. Remove it or set append=True."
        )
    io = IO(filename)
    io.extend(images)


def iread(filename) -> t.Iterable[ase.Atoms]:
    io = IO(filename)
    for atoms in io:
        yield atoms