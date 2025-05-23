import contextlib
import dataclasses
import importlib.metadata
import os
import pathlib
import typing as t
import warnings
from collections.abc import MutableSequence

import ase
import h5py
import numpy as np

from znh5md.interface.read import getitem
from znh5md.interface.write import extend
from znh5md.misc import open_file

__version__ = importlib.metadata.version("znh5md")


@dataclasses.dataclass
class IO(MutableSequence):
    """A class for handling H5MD files for ASE Atoms objects."""

    filename: str | os.PathLike | None = None
    file_handle: h5py.File | None = None
    pbc_group: bool = True  # Specify PBC per step (Not H5MD conform)
    save_units: bool = True  # Export ASE units into the H5MD file
    author: str = "N/A"
    author_email: str = "N/A"
    creator: str = "znh5md"
    creator_version: str = __version__
    particles_group: str | None = None
    compression: str | None = "gzip"
    compression_opts: int | None = None
    timestep: float = 1.0
    store: t.Literal["time", "linear"] = "linear"
    tqdm_limit: int = 100
    chunk_size: int | None = 64
    use_ase_calc: bool = True
    variable_shape: bool = True
    include: list[str] | None = None

    _store_ase_origin: bool = True  # for testing purposes only

    def __post_init__(self):
        if self.filename is None and self.file_handle is None:
            raise ValueError("Either filename or file_handle must be provided")
        if self.filename is not None and self.file_handle is not None:
            raise ValueError("Only one of filename or file_handle can be provided")
        if self.filename is not None:
            self.filename = pathlib.Path(self.filename)
        self._set_particle_group()
        self._read_author_creator()
        if self.include is not None:
            if "position" not in self.include:
                raise ValueError("'position' must be in keys")

    def _set_particle_group(self):
        if self.particles_group is not None:
            pass
        elif self.filename is not None and self.filename.exists():
            with open_file(self.filename, self.file_handle, mode="r") as f:
                self.particles_group = next(iter(f["particles"].keys()))
        elif (
            self.file_handle is not None
            and pathlib.Path(self.file_handle.filename).exists()
        ):
            with open_file(self.filename, self.file_handle, mode="r") as f:
                self.particles_group = next(iter(f["particles"].keys()))
        else:
            self.particles_group = "atoms"  # Default group name

    def _read_author_creator(self):
        with contextlib.suppress(FileNotFoundError, KeyError):
            # FileNotFoundError if the filename does not exist
            # KeyError if the file has not yet been initialized as H5MD
            #   or the keys are not provided, which is officially
            #   not allowed in H5MD.
            with open_file(self.filename, self.file_handle, mode="r") as f:
                self.author = f["h5md"]["author"].attrs["name"]
                self.author_email = f["h5md"]["author"].attrs["email"]
                self.creator = f["h5md"]["creator"].attrs["name"]
                self.creator_version = f["h5md"]["creator"].attrs["version"]

    def create_file(self):
        with open_file(self.filename, self.file_handle, mode="w") as f:
            g_h5md = f.create_group("h5md")
            g_h5md.attrs["version"] = np.array([1, 1])
            g_author = g_h5md.create_group("author")
            g_author.attrs["name"] = self.author
            g_author.attrs["email"] = self.author_email
            g_creator = g_h5md.create_group("creator")
            g_creator.attrs["name"] = self.creator
            g_creator.attrs["version"] = self.creator_version
            f.create_group("particles")

    def __len__(self) -> int:
        with open_file(self.filename, self.file_handle, mode="r") as f:
            return len(f["particles"][self.particles_group]["species"]["value"])

    def __getitem__(
        self, index: int | np.int_ | slice | np.ndarray
    ) -> ase.Atoms | list[ase.Atoms]:
        try:
            return getitem(self, index)
        except FileNotFoundError:
            # FileNotFoundError is an OSError, but we want to handle it
            #  separately from the OSError h5py raises
            raise
        except OSError:
            raise IndexError("Index out of range")

    def extend(self, frames: list[ase.Atoms]) -> None:
        if not isinstance(frames, list):
            raise ValueError("images must be a list of ASE Atoms objects")
        if len(frames) == 0:
            warnings.warn("No data provided")
            return
        if self.filename is not None and not self.filename.exists():
            self.create_file()
        if self.file_handle is not None:
            needs_creation = False
            with open_file(self.filename, self.file_handle, mode="r") as f:
                needs_creation = "h5md" not in f
            if needs_creation:
                self.create_file()

        extend(self, frames)

    def append(self, atoms: ase.Atoms):
        if not isinstance(atoms, ase.Atoms):
            raise ValueError("atoms must be an ASE Atoms object")
        self.extend([atoms])

    def __delitem__(self, index):
        raise NotImplementedError("Deleting items is not supported")

    def __setitem__(self, index, value):
        raise NotImplementedError("Setting items is not supported")

    def insert(self, index, value):
        raise NotImplementedError("Inserting items is not supported")
