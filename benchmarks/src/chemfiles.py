import ase
import chemfiles

from .abc import IOBase


class ChemfilesIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        if self.format in ["xyz", "pdb", "xtc"]:
            with chemfiles.Trajectory(
                self.filename, "r", format=self.format.upper()
            ) as trajectory:
                frames = []
                for frame in trajectory:
                    positions = frame.positions
                    cell_vectors = frame.cell.matrix
                    atoms = ase.Atoms(positions=positions, cell=cell_vectors, pbc=True)
                    frames.append(atoms)
            return frames
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def write(self, atoms: list[ase.Atoms]) -> None:
        raise NotImplementedError
