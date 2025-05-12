import ase
import MDAnalysis as mda

from .abc import IOBase


class MDAIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        universe = mda.Universe(self.filename, format="XYZ")
        frames = []
        for ts in universe.trajectory:
            positions = ts.positions
            cell_vectors = ts.dimensions
            atoms = ase.Atoms(positions=positions, cell=cell_vectors, pbc=True)
            frames.append(atoms)
        return frames

    def write(self, atoms: list[ase.Atoms]) -> None:
        raise NotImplementedError
