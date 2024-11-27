import dataclasses
import os
from collections.abc import MutableSequence
import typing as t
import h5py
import importlib.metadata
import contextlib
import pathlib
import numpy as np
import ase
from znh5md.interface.read import getitem
from znh5md.interface.write import extend

__version__ = importlib.metadata.version("znh5md")


@contextlib.contextmanager
def _open_file(
    filename: str | os.PathLike | None, file_handle: h5py.File | None, **kwargs
) -> t.Generator[h5py.File, None, None]:
    if file_handle is not None:
        yield file_handle
    else:
        with h5py.File(filename, **kwargs) as f:
            yield f

@dataclasses.dataclass
class IO(MutableSequence):
    """A class for handling H5MD files for ASE Atoms objects."""

    filename: str | os.PathLike |None = None
    file_handle: h5py.File|None = None
    pbc_group: bool = True  # Specify PBC per step (Not H5MD conform)
    save_units: bool = True  # Export ASE units into the H5MD file
    author: str = "N/A"
    author_email: str = "N/A"
    creator: str = "znh5md"
    creator_version: str = __version__
    particles_group: str|None = None
    compression: str|None = "gzip"
    compression_opts: int|None = None
    timestep: float = 1.0
    store: t.Literal["time", "linear"] = "linear"
    tqdm_limit: int = 100
    chunk_size: int|None = None
    use_ase_calc: bool = True

    def __post_init__(self):
        if self.filename is None and self.file_handle is None:
            raise ValueError("Either filename or file_handle must be provided")
        if self.filename is not None and self.file_handle is not None:
            raise ValueError("Only one of filename or file_handle can be provided")
        if self.filename is not None:
            self.filename = pathlib.Path(self.filename)
        self._set_particle_group()
        self._read_author_creator()

    def _set_particle_group(self):
        if self.particles_group is not None:
            pass
        elif self.filename is not None and self.filename.exists():
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.particles_group = next(iter(f["particles"].keys()))
        elif (
            self.file_handle is not None
            and pathlib.Path(self.file_handle.filename).exists()
        ):
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.particles_group = next(iter(f["particles"].keys()))
        else:
            self.particles_group = "atoms" # Default group name

    def _read_author_creator(self):
        with contextlib.suppress(FileNotFoundError, KeyError):
            # FileNotFoundError if the filename does not exist
            # KeyError if the file has not yet been initialized as H5MD
            #   or the keys are not provided, which is officially
            #   not allowed in H5MD.
            with _open_file(self.filename, self.file_handle, mode="r") as f:
                self.author = f["h5md"]["author"].attrs["name"]
                self.author_email = f["h5md"]["author"].attrs["email"]
                self.creator = f["h5md"]["creator"].attrs["name"]
                self.creator_version = f["h5md"]["creator"].attrs["version"]

    def create_file(self):
        with _open_file(self.filename, self.file_handle, mode="w") as f:
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
        with _open_file(self.filename, self.file_handle, mode="r") as f:
            return len(f["particles"][self.particles_group]["species"]["value"])
        
    def __getitem__(
        self, index: int | np.int_| slice | np.ndarray
    ) -> ase.Atoms| list[ase.Atoms]:
        return getitem(self, index)

    def extend(self, frames: list[ase.Atoms]) -> None:
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