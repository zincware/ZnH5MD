import importlib.metadata

from .io import IO
from .znh5md import iread, read, write

__all__ = ["IO", "read", "write", "iread"]

__version__ = importlib.metadata.version("znh5md")
