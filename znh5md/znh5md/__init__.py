from __future__ import annotations

import contextlib

from znh5md.znh5md.base import H5MDBase
from znh5md.znh5md.h5ase import ASEH5MD

with contextlib.suppress(ImportError):
    from znh5md.znh5md.h5dask import DaskH5MD

__all__ = ["H5MDBase", "ASEH5MD", "DaskH5MD"]
