import os

import numpy.testing as npt
import numpy as np
import pytest

import znh5md


@pytest.mark.parametrize("atoms_list", ["fix_size", "vary_size"], indirect=True)
@pytest.mark.parametrize("use_add", [True, False])
def test_AtomsReader(tmp_path, atoms_list, use_add):
    os.chdir(tmp_path)
    print(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()

    reader = znh5md.io.AtomsReader(atoms_list, frames_per_chunk=3)
    # we use a really small frames_per_chunk for testing purposes

    if use_add:
        db.add(reader)
    else:
        for data in reader.yield_chunks():
            db.add_chunk_data(**data)

    data = znh5md.ASEH5MD("db.h5")
    atoms = data.get_atoms_list()

    assert len(atoms) == len(atoms_list)
    for a, b in zip(atoms, atoms_list):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_equal(a.get_forces(), b.get_forces())
        npt.assert_array_equal(a.get_cell(), b.get_cell())
        npt.assert_array_equal(a.get_potential_energy(), b.get_potential_energy())
        npt.assert_array_equal(a.get_pbc(), b.get_pbc())
        npt.assert_array_equal(a.get_stress(), b.get_stress())

    # now test with Dask
    traj = znh5md.DaskH5MD("db.h5")
    positions = traj.position.value.compute()
    for idx, atoms in enumerate(atoms_list):
        npt.assert_array_equal(znh5md.utils.rm_nan(positions[idx]), atoms.get_positions())

    # npt.assert_array_equal(traj.position.time.compute(), np.linspace(0, 1, 100))
    # npt.assert_array_equal(traj.position.step.compute(), np.arange(100))
    # npt.assert_array_equal(
    #     traj.position.species.compute(),
    #     np.concatenate([np.ones((100, 5)), 2 * np.ones((100, 5))], axis=1),
    # )
