import ase
from scm.plams import XYZTrajectoryFile

from .abc import IOBase


class PLAMSIO(IOBase):
    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]:
        # Read the trajectory file using PLAMS
        if self.format == "xyz":
            traj = XYZTrajectoryFile(self.filename)
            mol = traj.get_plamsmol()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        frames = []
        for i in range(traj.get_length()):
            crd, _ = traj.read_frame(i, molecule=mol)
            atoms = ase.Atoms(
                positions=crd,
                pbc=True,
            )
            frames.append(atoms)
        return frames

    def write(self, atoms: list[ase.Atoms]) -> None:
        raise NotImplementedError
