import os

import ase.io
import numpy as np
import numpy.testing as npt
import pytest
import tqdm

import znh5md


@pytest.mark.parametrize("reader", [znh5md.io.ASEFileReader, znh5md.io.AtomsReader])
@pytest.mark.parametrize("atoms_list", ["fix_size", "vary_size"], indirect=True)
@pytest.mark.parametrize("use_add", [True, False])
def test_AtomsReader(tmp_path, reader, atoms_list, use_add):
    os.chdir(tmp_path)
    print(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()

    if reader == znh5md.io.AtomsReader:
        inputs = atoms_list
    elif reader == znh5md.io.ASEFileReader:
        inputs = "traj.xyz"
        ase.io.write(inputs, atoms_list)

    reader = reader(inputs, frames_per_chunk=3, step=1, time=0.1)

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
        npt.assert_array_almost_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_almost_equal(a.get_forces(), b.get_forces())
        npt.assert_array_almost_equal(a.get_cell(), b.get_cell())
        npt.assert_array_almost_equal(a.get_potential_energy(), b.get_potential_energy())
        npt.assert_array_equal(a.get_pbc(), b.get_pbc())
        npt.assert_array_almost_equal(a.get_stress(), b.get_stress())

    # now test with Dask
    traj = znh5md.DaskH5MD("db.h5")
    positions = traj.position.value.compute()
    time = traj.position.time.compute()
    step = traj.position.step.compute()
    species = traj.position.species.compute()
    for idx, atoms in enumerate(atoms_list):
        npt.assert_array_almost_equal(
            znh5md.utils.rm_nan(positions[idx]), atoms.get_positions()
        )
        npt.assert_array_almost_equal(time[idx], idx / 10)
        npt.assert_array_equal(step[idx], idx)
        npt.assert_array_equal(
            znh5md.utils.rm_nan(species[idx]), atoms.get_atomic_numbers()
        )


@pytest.mark.parametrize("frames_per_chunk", [3, 50000])
def test_ChemfilesReader(tmp_path, atoms_list, frames_per_chunk):
    os.chdir(tmp_path)
    print(tmp_path)

    db = znh5md.io.DataWriter(filename="db.h5")
    db.initialize_database_groups()

    inputs = "traj.xyz"
    ase.io.write(inputs, atoms_list)

    reader = znh5md.io.ChemfilesReader(inputs, frames_per_chunk=frames_per_chunk)
    db.add(reader)

    data = znh5md.ASEH5MD("db.h5")
    atoms = data.get_atoms_list()

    assert len(atoms) == len(atoms_list)

    for a, b in zip(atoms, atoms_list):
        npt.assert_array_almost_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
