"""ZnH5MD: High Performance Interface for H5MD Trajectories."""
import contextlib
import importlib.metadata

from znh5md import io, utils
from znh5md.format import FormatHandler
from znh5md.znh5md import ASEH5MD

__all__ = ["DaskH5MD", "ASEH5MD", "io", "FormatHandler", "utils"]

with contextlib.suppress(ImportError):
    from znh5md.znh5md.h5dask import DaskH5MD

    __all__.append("DaskH5MD")


__version__ = importlib.metadata.version("znh5md")
