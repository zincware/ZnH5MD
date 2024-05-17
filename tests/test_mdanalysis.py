"""Use the MDAnalysis library to read H5 files and check compliance with the H5MD standard."""
import pathlib

import ase.build
import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.H5MD import H5MDReader

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
    reader = znh5md.io.AtomsReader(trajectory)
    db.add(reader)
    return filename


def test_read_h5md(h5_trajectory, trajectory):
    u = mda.Universe.empty(n_atoms=3, trajectory=True)
    reader = H5MDReader(h5_trajectory)
    u.trajectory = reader
    assert len(u.trajectory) == 100
    for ref, ts in zip(trajectory, u.trajectory):
        assert np.allclose(ref.positions, ts.positions)
