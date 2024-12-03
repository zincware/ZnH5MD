import importlib.metadata

from znh5md.interface import IO

from .znh5md import iread, read, write

__all__ = ["IO", "read", "write", "iread"]

__version__ = importlib.metadata.version("znh5md")
