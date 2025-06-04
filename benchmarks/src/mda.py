import ase
import MDAnalysis as mda  # noqa: N813
from MDAnalysis.coordinates.H5MD import H5MDReader
from MDAnalysis.coordinates.XTC import XTCReader

from .abc import IOBase


class MDAIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        if self.format == "h5md":
            universe = mda.Universe.empty(self.num_atoms, trajectory=True)
            reader = H5MDReader(
                self.filename,
                convert_units=False,
                # dt=2,
                # time_offset=10,
                # foo="bar",
            )
            universe.trajectory = reader
        elif self.format == "xtc":
            universe = mda.Universe.empty(self.num_atoms, trajectory=True)
            reader = XTCReader(
                self.filename,
                convert_units=False,
            )
            universe.trajectory = reader
        else:
            universe = mda.Universe(self.filename, format=self.format.upper())

        universe.transfer_to_memory()
        frames = []
        for ts in universe.trajectory:
            positions = ts.positions
            cell_vectors = ts.dimensions
            atoms = ase.Atoms(positions=positions, cell=cell_vectors, pbc=True)
            frames.append(atoms)
        return frames

    def write(self, atoms: list[ase.Atoms]) -> None:
        raise NotImplementedError
