"""Use the MDAnalysis library to read H5 files and check compliance with the H5MD standard."""
import MDAnalysis as mda
from MDAnalysis.coordinates.H5MD import H5MDReader
import numpy as np
import ase.build
import pytest
import pathlib

import znh5md

@pytest.fixture
def trajectory() -> list[ase.Atoms]:
    """Generate ase.Atoms objects that moves linearly in space."""
    water = ase.build.molecule("H2O")
    atoms_list = [water]
    while len(atoms_list) < 100:
        atoms = atoms_list[-1].copy()
        atoms.positions += [0.1, 0.1, 0.1]
        atoms_list.append(atoms)
    return atoms_list

@pytest.fixture
def h5_trajectory(tmp_path, trajectory) -> pathlib.Path:
    """Write the trajectory to an H5 file."""
    filename = tmp_path / "trajectory.h5"
    db = znh5md.io.DataWriter(filename=filename)
    db.initialize_database_groups()
    reader = znh5md.io.AtomsReader(trajectory)
    db.add(reader)
    return filename

def test_read_h5md(h5_trajectory):
    u = mda.Universe.empty(3, n_residues=3, atom_resindex=np.arange(3), trajectory=True)
    reader = H5MDReader(h5_trajectory, convert_units=False)
    u.trajectory = reader

    assert len(u.trajectory) == 100

