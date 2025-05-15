import ase.io
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank

import znh5md

from .abc import IOBase


class ASEIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        if self.format == "pdb":
            return read_proteindatabank(self.filename, index=slice(None, None, None))
        elif self.format == "h5md":
            # TODO
            return list(znh5md.IO(self.filename))  # basically what ase does
        elif self.format == "xtc":
            raise ValueError("xtc format not supported")
        else:
            return ase.io.read(self.filename, format=self.format, index=":")
        # return list(ase.io.iread(self.filename, format=self.format))
        # no performance difference

    def write(self, atoms: list[ase.Atoms]) -> None:
        if self.format == "pdb":
            write_proteindatabank(self.filename, atoms)
        elif self.format == "h5md":
            znh5md.IO(self.filename, store="time").extend(atoms)
        elif self.format == "xtc":
            raise ValueError("xtc format not supported")
        else:
            ase.io.write(self.filename, atoms, format=self.format)


class ASECreate(IOBase):
    def setup(self):
        if self.format == "pdb":
            self.frames = read_proteindatabank(
                self.filename, index=slice(None, None, None)
            )
        elif self.format == "h5md":
            self.frames = list(znh5md.IO(self.filename))
        elif self.format == "xtc":
            self.frames = []  # TODO!
        else:
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
