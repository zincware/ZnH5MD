import ase
import numpy.testing as npt
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md


def test_empty(tmp_path):
    atoms = ase.Atoms()

    io = znh5md.IO(tmp_path / "frames.h5")
    io.append(atoms)

    new_atoms = io[0]
    assert len(new_atoms) == len(atoms)
    assert new_atoms.info.keys() == atoms.info.keys()
    for key in atoms.info.keys():
        npt.assert_equal(new_atoms.info[key], atoms.info[key])
    assert new_atoms.arrays.keys() == atoms.arrays.keys()
    for key in atoms.arrays.keys():
        npt.assert_equal(new_atoms.arrays[key], atoms.arrays[key])
    assert new_atoms.calc is None


def test_empty_info(tmp_path):
    atoms = ase.Atoms()
    atoms.info["test"] = []

    io = znh5md.IO(tmp_path / "frames.h5")
    io.append(atoms)

    new_atoms = io[0]
    assert len(new_atoms) == len(atoms)
    # ZnH5MD will remove keys in info which are an empty list!!

    # assert new_atoms.info.keys() == atoms.info.keys()
    # for key in atoms.info.keys():
    #     npt.assert_equal(new_atoms.info[key], atoms.info[key])
    assert new_atoms.arrays.keys() == atoms.arrays.keys()
    for key in atoms.arrays.keys():
        npt.assert_equal(new_atoms.arrays[key], atoms.arrays[key])
    assert new_atoms.calc is None


def test_empty_array(tmp_path):
    atoms = ase.Atoms()
    atoms.arrays["test"] = []

    io = znh5md.IO(tmp_path / "frames.h5")
    io.append(atoms)

    new_atoms = io[0]
    assert len(new_atoms) == len(atoms)
    assert new_atoms.info.keys() == atoms.info.keys()
    for key in atoms.info.keys():
        npt.assert_equal(new_atoms.info[key], atoms.info[key])

    # ZnH5MD will remove keys in arrays which are an empty list!!

    # assert new_atoms.arrays.keys() == atoms.arrays.keys()
    # for key in atoms.arrays.keys():
    #     npt.assert_equal(new_atoms.arrays[key], atoms.arrays[key])
    assert new_atoms.calc is None


def test_empty_calc(tmp_path):
    atoms = ase.Atoms()
    atoms.calc = SinglePointCalculator(atoms, energy=0.0)
    atoms.calc.results["test"] = []
    io = znh5md.IO(tmp_path / "frames.h5")
    io.append(atoms)

    new_atoms = io[0]
    assert len(new_atoms) == len(atoms)
    assert new_atoms.info.keys() == atoms.info.keys()
    for key in atoms.info.keys():
        npt.assert_equal(new_atoms.info[key], atoms.info[key])
    assert new_atoms.arrays.keys() == atoms.arrays.keys()
    for key in atoms.arrays.keys():
        npt.assert_equal(new_atoms.arrays[key], atoms.arrays[key])

    # ZnH5MD will remove keys in calc.results which are an empty list!!

    # assert new_atoms.calc.results.keys() == atoms.calc.results.keys()
    # for key in atoms.calc.results.keys():
    #     npt.assert_equal(new_atoms.calc.results[key], atoms.calc.results[key])
