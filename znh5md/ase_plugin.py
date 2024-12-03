from pathlib import Path
from typing import Generator

from ase.atoms import Atoms
from ase.utils.plugins import ExternalIOFormat

from znh5md.interface import IO

znh5md_format = ExternalIOFormat(
    desc="ZnH5MD h5 file",
    code="+S",  # multiple atoms objects, accepts a file name string
    module="znh5md.ase_plugin",
    ext="h5",
)


def read_znh5md(filename: str, index: slice) -> Generator[Atoms, None, None]:
    yield from IO(filename=filename, tqdm_limit=0)[index]


def write_znh5md(filename: str, images: list[Atoms], append: bool = False):
    if not append and Path(filename).exists():
        Path(filename).unlink()
    IO(filename=filename, tqdm_limit=0).extend(images)
