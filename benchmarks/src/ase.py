import ase.io

from .abc import IOBase


class ASEIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        return ase.io.read(self.filename, format=self.format, index=":")
        # return list(ase.io.iread(self.filename, format=self.format)) # no performance difference

    def write(self, atoms: list[ase.Atoms]) -> None:
        ase.io.write(self.filename, atoms, format=self.format)


class ASECreate(IOBase):
    def setup(self):
        self.frames = list(ase.io.iread(self.filename, format=self.format))

    def read(self) -> list[ase.Atoms]:
        frames = []
        for atoms in self.frames:
            frames.append(
                ase.Atoms(
                    positions=atoms.get_positions(),
                )
            )

    def write(self, atoms: list[ase.Atoms]) -> None:
        raise NotImplementedError