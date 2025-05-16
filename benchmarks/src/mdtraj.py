import os

import mdtraj as md
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io.proteindatabank import write_proteindatabank

from .abc import IOBase


class MDTrajIO(IOBase):
    def setup(self):
        atom_names = [f"H{i + 1}" for i in range(self.num_atoms)]  # Unique atom names
        elements = [
            "H"
        ] * self.num_atoms  # All elements are H as per original user code structure

        top_df = pd.DataFrame(
            {
                "name": atom_names,  # Atom names (e.g., H1, H2)
                "element": elements,  # Element symbols (e.g., H, C, O)
                "resSeq": [1] * self.num_atoms,
                "resName": ["MOL"] * self.num_atoms,  # Using a generic residue name
                "chainID": ["A"] * self.num_atoms,
                "serial": np.arange(self.num_atoms),
            }
        )
        self.topology = md.Topology.from_dataframe(top_df)

    def read(self) -> list[Atoms]:
        # Read the trajectory file using MDTraj
        if self.format == "xyz":
            traj = md.load_xyz(self.filename, top=self.topology)
            frames = []
            for i in range(traj.n_frames):
                positions = traj.xyz[i]
                atoms = Atoms(
                    positions=positions,
                    pbc=True,
                )
                frames.append(atoms)
            return frames
        elif self.format == "pdb":
            traj = md.load_pdb(self.filename)
            frames = []
            for i in range(traj.n_frames):
                positions = traj.xyz[i]
                atoms = Atoms(
                    positions=positions,
                    pbc=True,
                )
                frames.append(atoms)
            return frames
        elif self.format == "xtc":
            traj = md.load_xtc(self.filename, top=self.topology)
            frames = []
            for i in range(traj.n_frames):
                positions = traj.xyz[i]
                atoms = Atoms(
                    positions=positions,
                    pbc=True,
                )
                frames.append(atoms)
            return frames
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def write(self, frames: list[Atoms]):
        if self.format == "xtc":
            # use ase to write xyz, load that file and then save as xtc
            write_proteindatabank(
                "ase_pdb_to_xtc_tmp.pdb", frames
            )  # Write to a temporary XYZ file
            traj = md.load_pdb("ase_pdb_to_xtc_tmp.pdb")
            traj.save_xtc(self.filename)
            # remove the temporary file
            os.remove("ase_pdb_to_xtc_tmp.pdb")
        else:
            raise ValueError(f"Unsupported format: {self.format}")
