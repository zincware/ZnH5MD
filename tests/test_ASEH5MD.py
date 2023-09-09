import os

import ase
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
import numpy.testing as npt

import znh5md


def test_shape(example_h5):
    traj = znh5md.ASEH5MD(example_h5)
    # ASEH5MD has no time information
    assert traj.position.shape == (100, 10, 3)
    assert traj.species.shape == (100, 10)


def test_get_atoms_list(example_h5):
    traj = znh5md.ASEH5MD(example_h5)
    atoms = traj.get_atoms_list()
    assert len(atoms) == 100
    assert isinstance(atoms[0], ase.Atoms)


@pytest.mark.parametrize("remove_calc", [True, False])
def test_get_slice(tmp_path, atoms_list, remove_calc):
    os.chdir(tmp_path)
    if remove_calc:
        for atoms in atoms_list:
            atoms.calc = None

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

    assert len(traj.position) == 21


@pytest.mark.parametrize("atoms_list", ["no_stress"], indirect=True)
@pytest.mark.parametrize("remove_calc", [True, False])
def test_request_missing_properties(tmp_path, atoms_list, remove_calc):
    os.chdir(tmp_path)
    if remove_calc:
        for atoms in atoms_list:
            atoms.calc = None

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()

    if remove_calc:
        with pytest.raises(RuntimeError):
            for chunk in znh5md.io.AtomsReader(atoms_list).yield_chunks(
                group_names=["stress"]
            ):
                db.add_chunk_data(**chunk)
    else:
        with pytest.raises(PropertyNotImplementedError):
            for chunk in znh5md.io.AtomsReader(atoms_list).yield_chunks(
                group_names=["stress"]
            ):
                db.add_chunk_data(**chunk)

def test_DataWriter_custom_arrays(tmp_path, atoms_list_with_custom_arrays):
    os.chdir(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()
    db.add(znh5md.io.AtomsReader(atoms_list_with_custom_arrays))

    traj = znh5md.ASEH5MD("db.h5")

    for a, b in zip(traj.get_atoms_list(), atoms_list_with_custom_arrays):
        for key in b.arrays:
            if key in ["initial_magmoms", "initial_charges"]:
                # TODO these are currently not supported by ZnH5MD
                continue
            npt.assert_allclose(a.arrays[key], b.arrays[key])