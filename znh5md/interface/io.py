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
    """
    A class for handling H5MD files for ASE Atoms objects.

    This class provides an interface to read from and write to H5MD files,
    treating them as a sequence of ASE Atoms objects. It supports various
    configurations for file handling, data compression, and metadata.

    Parameters
    ----------
    filename : str | os.PathLike | None, optional
        Path to the H5MD file. If `file_handle` is not provided, this must be specified.
        Defaults to None.
    file_handle : h5py.File | None, optional
        An existing h5py file handle. If `filename` is not provided, this must
        be specified.
        Defaults to None.
    file_factory : Callable[[], ContextManager[h5py.File]] | None, optional
        A factory function that returns a context manager for an h5py file.
        File factory is restricted to read-only data access.
    pbc_group : bool, optional
        Specify if Periodic Boundary Conditions (PBC) should be stored per step.
        This is not H5MD conformant, but it allows for more flexibility in data storage.
        Defaults to True.
    save_units : bool, optional
        Export ASE units into the H5MD file. Defaults to True.
    author : str, optional
        Author's name for H5MD metadata. Defaults to "N/A".
    author_email : str, optional
        Author's email for H5MD metadata. Defaults to "N/A".
    creator : str, optional
        Name of the creating software. Defaults to "znh5md".
    creator_version : str, optional
        Version of the creating software. Defaults to the znh5md package version.
    particles_group : str | None, optional
        Name of the particles group within the H5MD file (e.g., "atoms").
        If None, it tries to infer from an existing file or defaults to "atoms".
        Defaults to None.
    compression : str | None, optional
        Compression filter to use for datasets (e.g., "gzip", "lzf").
        Defaults to "gzip".
    compression_opts : int | None, optional
        Compression options (e.g., compression level for gzip). Defaults to None.
    timestep : float, optional
        Time step between frames in the H5MD file. Defaults to 1.0.
    store : t.Literal["time", "linear"], optional
        Method for storing time and step information.
        "time" stores a time series, "linear" stores a single value.
        Defaults to "linear".
    tqdm_limit : int, optional
        Threshold for using tqdm progress bar. Defaults to 100.
    chunk_size : int | None | list[int] | tuple[int, ...], optional
        Chunk shape for HDF5 datasets. Can be an integer, list, or tuple.
        Defaults to (64, 64).
    use_ase_calc : bool, optional
        Whether to use ASE calculator for storing and retrieving data.
        Defaults to True.
    variable_shape : bool, optional
        Whether the data has a variable shape across frames. Defaults to True.
        This is not H5MD conformant, but allows for more flexibility in data storage.
    include : list[str] | None, optional
        List of attributes to include when reading/writing. If specified,
        "position" must be included. Defaults to None, which means all
        data is included.
    mask : list[int] | slice | None, optional
        Mask to apply when reading data, e.g., to select specific atoms.
        Not supported with `variable_shape=True`. Defaults to None.

    Raises
    ------
    ValueError
        If both `filename` and `file_handle` are provided or neither is provided.
        If "position" is not in `include` list when `include` is specified.
        If `mask` is used with `variable_shape=True`.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import bulk
    >>> from znh5md import IO
    >>>
    >>> # Create some dummy atoms objects
    >>> atoms1 = bulk("Cu", "fcc", a=3.6)
    >>> atoms2 = atoms1.copy()
    >>> atoms2.positions[0, 0] += 0.1
    >>>
    >>> # Write to an H5MD file
    >>> io = IO(filename="test.h5")
    >>> io.extend([atoms1, atoms2])
    >>>
    >>> # Read from the H5MD file
    >>> io_read = IO(filename="test.h5")
    >>> print(len(io_read))
    2
    >>> atoms_read = io_read[0]
    >>> print(atoms_read.positions[0])
    [0. 0. 0.]
    >>>
    >>> # Clean up
    >>> import os
    >>> os.remove("test.h5")
    """

    filename: str | os.PathLike | None = None
    file_handle: h5py.File | None = None
    file_factory: t.Callable[[], t.ContextManager[h5py.File]] | None = None
    pbc_group: bool = True
    save_units: bool = True
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
    chunk_size: int | None | list[int] | tuple[int, ...] = (64, 64)
    use_ase_calc: bool = True
    variable_shape: bool = True
    include: list[str] | None = None
    mask: list[int] | slice | None = None

    _store_ase_origin: bool = True  # for testing purposes only

    def __post_init__(self):
        sources = [
            self.filename is not None,
            self.file_handle is not None,
            self.file_factory is not None,
        ]
        if sum(sources) == 0:
            raise ValueError(
                "Either filename, file_handle, or file_factory must be provided"
            )
        if sum(sources) > 1:
            raise ValueError(
                "Only one of filename, file_handle, or file_factory can be provided"
            )
        if self.filename is not None:
            self.filename = pathlib.Path(self.filename)
        self._set_particle_group()
        self._read_author_creator()
        if self.include is not None:
            if "position" not in self.include:
                raise ValueError("'position' must be in keys")
        if self.mask is not None and self.variable_shape is True:
            raise ValueError("mask is not supported with variable_shape=True. ")

    def _set_particle_group(self):
        if self.particles_group is not None:
            pass
        elif self.filename is not None and self.filename.exists():
            with open_file(
                self.filename, self.file_handle, self.file_factory, mode="r"
            ) as f:
                self.particles_group = next(iter(f["particles"].keys()))
        elif (
            self.file_handle is not None
            and pathlib.Path(self.file_handle.filename).exists()
        ):
            with open_file(
                self.filename, self.file_handle, self.file_factory, mode="r"
            ) as f:
                self.particles_group = next(iter(f["particles"].keys()))
        else:
            self.particles_group = "atoms"  # Default group name

    def _read_author_creator(self):
        with contextlib.suppress(FileNotFoundError, KeyError):
            # FileNotFoundError if the filename does not exist
            # KeyError if the file has not yet been initialized as H5MD
            #   or the keys are not provided, which is officially
            #   not allowed in H5MD.
            with open_file(
                self.filename, self.file_handle, self.file_factory, mode="r"
            ) as f:
                self.author = f["h5md"]["author"].attrs["name"]
                self.author_email = f["h5md"]["author"].attrs["email"]
                self.creator = f["h5md"]["creator"].attrs["name"]
                self.creator_version = f["h5md"]["creator"].attrs["version"]

    def create_file(self):
        with open_file(
            self.filename, self.file_handle, self.file_factory, mode="w"
        ) as f:
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
        with open_file(
            self.filename, self.file_handle, self.file_factory, mode="r"
        ) as f:
            return len(f["particles"][self.particles_group]["species"]["value"])

    @t.overload
    def __getitem__(self, index: int) -> ase.Atoms: ...
    @t.overload
    def __getitem__(self, index: np.int_) -> ase.Atoms: ...
    @t.overload
    def __getitem__(self, index: slice) -> list[ase.Atoms]: ...
    @t.overload
    def __getitem__(self, index: np.ndarray) -> list[ase.Atoms]: ...
    @t.overload
    def __getitem__(self, index: list[int]) -> list[ase.Atoms]: ...

    def __getitem__(
        self, index: int | np.int_ | slice | np.ndarray | list[int]
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
            with open_file(
                self.filename, self.file_handle, self.file_factory, mode="r"
            ) as f:
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
