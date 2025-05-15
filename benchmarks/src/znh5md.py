
from .abc import IOBase
import znh5md
import ase


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
            znh5md.IO(self.filename, store="time").extend(atoms)
        else:
            raise ValueError(f"Unsupported format: {self.format}")