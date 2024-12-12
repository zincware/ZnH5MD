import mdtraj
import pandas as pd
import numpy as np
import chemfiles
import pytest
import ase
import uuid

def create_topology(n_atoms):
    """Create an MDTraj topology for a given number of atoms."""
    data = pd.DataFrame(
        {
            "serial": np.arange(n_atoms),
            "name": ["C"] * n_atoms,
            "resSeq": [1] * n_atoms,
            "resName": ["UNK"] * n_atoms,
            "element": ["C"] * n_atoms,
            "chainID": ["A"] * n_atoms,
        }
    )
    return mdtraj.Topology().from_dataframe(data)

def generate_frames(n_steps, n_atoms):
    """Generate a list of Chemfiles frames with random atomic positions."""
    frames = []
    for _ in range(n_steps):
        atoms = ase.Atoms("H" * n_atoms, positions=np.random.rand(n_atoms, 3))
        frame = chemfiles.Frame()
        frame.resize(n_atoms)
        frame.positions[:] = atoms.positions
        frames.append(frame)
    return frames

@pytest.fixture
def frames(request):
    """Fixture to create frames based on n_steps and n_atoms from the request."""
    n_steps, n_atoms = request.param
    return generate_frames(n_steps, n_atoms)

@pytest.mark.parametrize("frames", [(100, 10), (50, 20), (200, 5)], indirect=True)
def test_write_chemfiles_pdb(tmp_path, frames, benchmark):
    """Benchmark the PDB writing process."""
    def write_chemfiles_pdb():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.pdb"
        with chemfiles.Trajectory(filename.as_posix(), "w") as traj:
            for frame in frames:
                traj.write(frame)

    benchmark(write_chemfiles_pdb)