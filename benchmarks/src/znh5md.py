import ase

import znh5md

from .abc import IOBase


class ZnH5MDIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        if self.format == "h5md":
            return znh5md.IO(self.filename)[:]
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def write(self, atoms: list[ase.Atoms]) -> None:
        if self.format == "h5md":
            znh5md.IO(self.filename, store="time", compression=None).extend(atoms)
        else:
            raise ValueError(f"Unsupported format: {self.format}")


class ZnH5MDFixedShapeIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        if self.format == "h5md":
            return znh5md.IO(self.filename, variable_shape=False)[:]
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def write(self, atoms: list[ase.Atoms]) -> None:
        if self.format == "h5md":
            znh5md.IO(self.filename, store="time", variable_shape=False).extend(atoms)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
