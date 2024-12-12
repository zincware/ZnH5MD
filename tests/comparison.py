import mdtraj
import pandas as pd
import numpy as np
import chemfiles
import pytest
import ase.io
import uuid
import znh5md
import warnings


# n_steps, n_atoms

# N_ATOMS = np.logspace(1, 3, 20, dtype=int)
# N_STEPS = 1000
WRITE = [(1000, 1000)]

def collect_file_sizes(tmp_path) -> str:
    # compute the mean and standard deviation of the file sizes in the given directory
    sizes = []
    for file in tmp_path.iterdir():
        sizes.append(file.stat().st_size)
    # print mean and standard deviation in megabytes
    return f"Mean: {np.mean(sizes) / 1e6:.2f} MB, Std: {np.std(sizes) / 1e6:.2f} MB"

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

def generate_frames(n_steps, n_atoms) -> list[ase.Atoms]:
    """Generate a list of Chemfiles frames with random atomic positions."""
    frames = []
    for _ in range(n_steps):
        atoms = ase.Atoms("H" * n_atoms, positions=np.random.rand(n_atoms, 3))
        frames.append(atoms)
    return frames

def convert_atoms_to_chemfiles(atoms: ase.Atoms) -> chemfiles.Frame:
    """Convert an ASE atoms object to a Chemfiles frame."""
    frame = chemfiles.Frame()
    frame.resize(len(atoms))
    frame.positions[:] = atoms.positions
    return frame

@pytest.fixture
def frames(request):
    """Fixture to create frames based on n_steps and n_atoms from the request."""
    n_steps, n_atoms = request.param
    return list(generate_frames(n_steps, n_atoms))

@pytest.mark.parametrize("frames", WRITE, indirect=True)
def test_write_chemfiles_pdb(tmp_path, frames, benchmark):
    """Benchmark the PDB writing process."""
    chemfiles_frames = [convert_atoms_to_chemfiles(frame) for frame in frames]

    def write_chemfiles_pdb():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.pdb"
        with chemfiles.Trajectory(filename.as_posix(), "w") as traj:
            for frame in chemfiles_frames:
                traj.write(frame)

    benchmark(write_chemfiles_pdb)
    warnings.warn(collect_file_sizes(tmp_path))

@pytest.mark.parametrize("frames", WRITE, indirect=True)
def test_write_znh5md(tmp_path, frames, benchmark):
    def write_znh5md():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.h5"
        znh5md.write(filename, frames, compression=None)

    benchmark(write_znh5md)
    warnings.warn(collect_file_sizes(tmp_path))


@pytest.mark.parametrize("frames", WRITE, indirect=True)
def test_write_znh5md_compressed(tmp_path, frames, benchmark):
    def write_znh5md():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.h5"
        znh5md.write(filename, frames)

    benchmark(write_znh5md)
    warnings.warn(collect_file_sizes(tmp_path))


@pytest.mark.parametrize("frames", WRITE, indirect=True)
def test_write_xtc(tmp_path, frames, benchmark):
    topology = create_topology(len(frames[0]))
    positions = np.array([frame.positions for frame in frames])

    def write_xtc():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.xtc"
        traj = mdtraj.Trajectory(
            positions, topology
        )
        traj.save_xtc(filename.as_posix())

    benchmark(write_xtc)
    warnings.warn(collect_file_sizes(tmp_path))

@pytest.mark.parametrize("frames", WRITE, indirect=True)
def test_ase_traj(tmp_path, frames, benchmark):
    def write_ase_traj():
        """Inner function for benchmarking."""
        filename = tmp_path / f"{uuid.uuid4()}.traj"
        ase.io.write(filename.as_posix(), frames)

    benchmark(write_ase_traj)
    warnings.warn(collect_file_sizes(tmp_path))