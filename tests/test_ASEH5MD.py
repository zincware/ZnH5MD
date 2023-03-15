import os

import ase

import znh5md


def test_shape(example_h5):
    traj = znh5md.ASEH5MD(example_h5)
    assert traj.position.value.shape == (100, 10, 3)
    assert traj.position.time.shape == (100,)
    assert traj.position.step.shape == (100,)
    assert traj.position.species.shape == (100, 10)


def test_get_atoms_list(example_h5):
    traj = znh5md.ASEH5MD(example_h5)
    atoms = traj.get_atoms_list()
    assert len(atoms) == 100
    assert isinstance(atoms[0], ase.Atoms)


def test_get_slice(tmp_path, atoms_list):
    os.chdir(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()
    db.add(znh5md.io.AtomsReader(atoms_list))

    traj = znh5md.ASEH5MD("db.h5")
    atoms = traj[[1, 3, 6]]
    assert len(atoms) == 3

    assert atoms[0] == atoms_list[1]
    assert atoms[1] == atoms_list[3]
    assert atoms[2] == atoms_list[6]

    traj[0] == atoms_list[0]
    traj[-1] == atoms_list[-1]
    traj[1:2] == atoms_list[1:2]
