import ase.build
import numpy.testing as npt

import znh5md


def test_box_pbc(tmp_path):
    water = ase.build.molecule("H2O")
    water.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    water.pbc = [False, False, False]

    path = tmp_path / "test.h5"
    db = znh5md.io.DataWriter(path)
    db.add(znh5md.io.AtomsReader([water]))

    traj = znh5md.ASEH5MD(path)
    atoms = traj.get_atoms_list()[0]

    npt.assert_array_equal(atoms.get_pbc(), water.get_pbc())
    npt.assert_array_equal(atoms.get_cell(), water.get_cell())
