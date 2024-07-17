from .io import IO
from .znh5md import iread, read, write
import importlib.metadata

__all__ = ["IO", "read", "write", "iread"]

__version__ = importlib.metadata.version("znh5md")
