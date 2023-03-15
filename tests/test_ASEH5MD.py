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
