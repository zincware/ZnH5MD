import numpy as np
import ase.io
import mdtraj as md
import chemfiles
import MDAnalysis as mda
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

np.random.seed(42)

def create_trajectory_ase(n_frames=10, n_atoms=5) -> list[ase.Atoms]:
    """
    Create a random trajectory with n_frames and n_atoms using ASE.
    """
    frames = []
    for _ in range(n_frames):
        symbols = np.random.choice(['H', 'O', 'C'], size=n_atoms)
        positions = np.random.rand(n_atoms, 3)
        atoms = ase.Atoms(symbols=symbols, positions=positions)
        frames.append(atoms)
    return frames

class ASEIO:
    def __init__(self):
        self.filename = None
        self.trajectory = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename

    def read_xyz(self):
        return ase.io.read(self.filename, index=':')

    def write_xyz(self, frames: list[ase.Atoms]):
        ase.io.write(self.filename, frames)

class MDTrajIO:
    def __init__(self):
        self.filename = None
        self.topology = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename
        top_df = pd.DataFrame({
            'residue': ['H'] * n_atoms,
            'atom': ['H'] * n_atoms,
            'element': ['H'] * n_atoms,
            "name": ['H'] * n_atoms,
            "resSeq": [1] * n_atoms,
            "resName": ['H'] * n_atoms,
            "chainID": ['A'] * n_atoms,
            "serial": np.arange(n_atoms),
        })
        self.topology = md.Topology.from_dataframe(top_df)

    def read_xyz(self):
        return md.load_xyz(self.filename, top=self.topology)

    def write_xyz(self, frames: list[ase.Atoms]):
        pass  # Writing is done outside the reader

class ChemfilesIO:
    def __init__(self):
        self.filename = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename

    def read_xyz(self):
        trajectory = chemfiles.Trajectory(self.filename, 'r')
        frames = [frame for frame in trajectory]
        trajectory.close()
        return frames

    def write_xyz(self, frames: list[ase.Atoms]):
        pass  # Writing is done outside the reader

class MDAanalysisIO:
    def __init__(self):
        self.filename = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename

    def read_xyz(self):
        return mda.Universe(self.filename, format='XYZ')

    def write_xyz(self, frames: list[ase.Atoms]):
        pass  # Writing is done outside the reader

def benchmark_read(reader_object, num_repeats: int = 5) -> float:
    """
    Benchmark the read performance of a given reader object.
    """
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        reader_object.read_xyz()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def main():
    n_frames_list = np.logspace(1, 2, num=5, dtype=int)
    n_atoms_list = [100, 200]
    file_readers = {
        'ASE': ASE_Reader(),
        'MDTraj': MDTraj_Reader(),
        'Chemfiles': Chemfiles_Reader(),
        'MDAanalysis': MDAanalysis_Reader(),
    }
    # TODO: use the ASEIO to create the xyz files, one for each n_frames x n_atoms
    # TODO: benchmark the read performance and plot the results
