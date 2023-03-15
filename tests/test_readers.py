import os

import numpy.testing as npt

import znh5md


def test_AtomsReader(tmp_path, atoms_list):
    os.chdir(tmp_path)
    print(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()

    reader = znh5md.io.AtomsReader(atoms_list, frames_per_chunk=10)

    for data in reader.yield_chunks():
        db.add_chunk_data(**data)

    data = znh5md.ASEH5MD("db.h5")
    atoms = data.get_atoms_list()

    assert len(atoms) == len(atoms_list)
    for a, b in zip(atoms, atoms_list):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        assert a.get_potential_energy() == b.get_potential_energy()
        npt.assert_array_equal(a.get_forces(), b.get_forces())
