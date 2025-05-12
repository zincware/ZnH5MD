import ase.io

from .abc import IOBase


class ASEIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        return ase.io.read(self.filename, format=self.format, index=":")

    def write(self, atoms: list[ase.Atoms]) -> None:
        ase.io.write(self.filename, atoms, format=self.format)
