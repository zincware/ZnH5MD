import pathlib
import typing as t

import ase

from znh5md.interface import IO


def read(filename, index: int | slice | list[int] = -1) -> ase.Atoms | list[ase.Atoms]:
    io = IO(filename)
    return io[index]


def write(
    filename, images: ase.Atoms | list[ase.Atoms], append: bool = True, **kwargs
) -> None:
    if not append and pathlib.Path(filename).exists():
        raise FileExistsError(
            f"{filename} already exists. Remove it or set append=True."
        )
    if isinstance(images, ase.Atoms):
        images = [images]
    io = IO(filename, **kwargs)
    io.extend(images)


def iread(filename) -> t.Iterable[ase.Atoms]:
    io = IO(filename)
    for atoms in io:
        yield atoms
