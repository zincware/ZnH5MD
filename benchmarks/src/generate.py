import ase
import numpy as np


def create_frames(num_frames=10, num_atoms=5) -> list[ase.Atoms]:
    """
    Create a random trajectory with n_frames and n_atoms using ASE.
    Each frame will have a random selection of H, O, C atoms.
    """
    frames = []
    for _ in range(num_frames):
        symbols = np.random.choice(["H", "O", "C"], size=num_atoms)
        positions = np.random.rand(num_atoms, 3) * 10
        cell_vectors = np.eye(3) * 100
        atoms = ase.Atoms(
            symbols=symbols, positions=positions, cell=cell_vectors, pbc=False
        )
        frames.append(atoms)
    return frames
