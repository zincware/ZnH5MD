"""ZnH5MD: High Performance Interface for H5MD Trajectories."""
from znh5md.znh5md import DaskH5MD, ASEH5MD
import importlib.metadata

__all__ = ["DaskH5MD", "ASEH5MD"]


__version__ = importlib.metadata.version("znh5md")
