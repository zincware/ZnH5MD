import numpy.testing

import znh5md


def test_get_g2_h5_atoms(g2_h5):
    traj = znh5md.ASEH5MD(g2_h5)
    atoms = traj.get_atoms_list()
    assert isinstance(atoms[0], znh5md.Atoms)


def test_AtomsToDict(g2_h5):
    traj = znh5md.ASEH5MD(g2_h5)
    atoms = traj.get_atoms_list()
    atoms_dict = atoms[0].todict()

    new_atoms = znh5md.Atoms.fromdict(atoms_dict)
    assert new_atoms == atoms[0]
