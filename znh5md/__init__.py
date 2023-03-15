"""ZnH5MD: High Performance Interface for H5MD Trajectories."""
import importlib.metadata

from znh5md import io
from znh5md.znh5md import ASEH5MD, DaskH5MD, FormatHandler

__all__ = ["DaskH5MD", "ASEH5MD", "io", "FormatHandler"]


__version__ = importlib.metadata.version("znh5md")
