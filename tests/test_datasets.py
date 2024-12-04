import ase.io
import h5py
import numpy.testing as npt
import pytest

import znh5md


@pytest.mark.parametrize(
    "dataset",
    [
        "s22",
        "s22_energy",
        "s22_all_properties",
        "s22_info_arrays_calc",
        "s22_mixed_pbc_cell",
        "s22_illegal_calc_results",
        "water",
        "s22_nested_calc",
    ],
)
def test_datasets(tmp_path, dataset, request):
    images = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(images)
    images2 = io[:]

    assert len(images) == len(images2)

    for a, b in zip(images, images2):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_equal(a.get_cell(), b.get_cell())
        npt.assert_array_equal(a.get_pbc(), b.get_pbc())
        npt.assert_array_equal(a.get_velocities(), b.get_velocities())
        if a.calc is not None:
            assert set(a.calc.results) == set(b.calc.results)
            for key in a.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])
            if "energy" in a.calc.results:
                assert b.get_potential_energy() == a.get_potential_energy()
                assert isinstance(a.get_potential_energy(), float)
                assert isinstance(b.get_potential_energy(), float)
        else:
            assert b.calc is None

        assert set(a.arrays) == set(b.arrays)
        for key in a.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])

        # h5md keys are added to info automatically
        assert set(b.info) == set(a.info)
        for key in a.info:
            npt.assert_array_equal(a.info[key], b.info[key])


@pytest.mark.parametrize(
    "dataset",
    [
        "s22",
        "s22_energy",
        "s22_all_properties",
        "s22_info_arrays_calc",
        "s22_mixed_pbc_cell",
        "water",
    ],
)
def test_datasets_extxyz(tmp_path, dataset, request):
    images = request.getfixturevalue(dataset)
    ase.io.write(tmp_path / "test.xyz", images)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(list(ase.io.iread(tmp_path / "test.xyz")))
    images2 = io[:]

    assert len(images) == len(images2)

    for a, b in zip(images, images2):
        npt.assert_array_almost_equal(a.get_positions(), b.get_positions())
        npt.assert_array_almost_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_almost_equal(a.get_cell(), b.get_cell())
        npt.assert_array_almost_equal(a.get_pbc(), b.get_pbc())
        npt.assert_array_almost_equal(a.get_velocities(), b.get_velocities())
        if a.calc is not None:
            assert set(a.calc.results) == set(b.calc.results)
            for key in a.calc.results:
                npt.assert_array_almost_equal(a.calc.results[key], b.calc.results[key])
            if "energy" in a.calc.results:
                assert b.get_potential_energy() == a.get_potential_energy()
                assert isinstance(a.get_potential_energy(), float)
                assert isinstance(b.get_potential_energy(), float)
        else:
            assert b.calc is None

        assert set(a.arrays) == set(b.arrays)
        for key in a.arrays:
            npt.assert_array_almost_equal(a.arrays[key], b.arrays[key])

        # h5md keys are added to info automatically
        assert set(b.info) == set(a.info)
        for key in a.info:
            if isinstance(a.info[key], str):
                assert a.info[key] == b.info[key]
            elif isinstance(a.info[key], dict):
                assert a.info[key] == b.info[key]
            else:
                npt.assert_array_almost_equal(a.info[key], b.info[key])


@pytest.mark.parametrize("store", ["linear"])
@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_datasets_h5py(tmp_path, dataset, request, store):
    images = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5", store=store, timestep=0.5)
    io.extend(images)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert "particles/atoms/position/value" in f
        assert "particles/atoms/species/value" in f
        assert "particles/atoms/force/value" in f
        assert "observables/atoms/force/value" not in f
        assert "particles/atoms/velocity/value" in f
        assert "observables/atoms/velocity/value" not in f

        # assert "particles/atoms/momenta/value" in f
        # assert "observables/atoms/momenta/value" not in f

        assert "particles/atoms/potential_energy/value" not in f
        assert "observables/atoms/potential_energy/value" in f

        assert "particles/atoms/mlip_forces/value" in f
        assert "particles/atoms/mlip_forces_2/value" in f

        assert "observables/atoms/mlip_energy/value" in f
        assert "observables/atoms/mlip_energy_2/value" in f
        assert "observables/atoms/mlip_stress/value" in f

        assert f["particles/atoms/velocity/value"].attrs["unit"] == "Angstrom/fs"
        assert f["particles/atoms/force/value"].attrs["unit"] == "eV/Angstrom"

        # if store == "time":
        #     assert f["observables/atoms/energy/time"].shape == (len(images),)
        #     assert f["observables/atoms/energy/step"].shape == (len(images),)
        if store == "linear":
            assert f["observables/atoms/potential_energy/time"][()] == 0.5
            assert f["observables/atoms/potential_energy/step"][()] == 1

        # assert f["observables/atoms/energy/value"].attrs["unit"] == "eV"

        npt.assert_array_equal(
            f["particles/atoms/box"].attrs["boundary"], ["none", "none", "none"]
        )


@pytest.mark.parametrize("store", ["linear"])
def test_two_datasets(tmp_path, s22_all_properties, s22_mixed_pbc_cell, store):
    io_a = znh5md.IO(tmp_path / "test.h5", particles_group="a", store=store)
    io_b = znh5md.IO(tmp_path / "test.h5", particles_group="b")
    io_a.extend(s22_all_properties)
    io_b.extend(s22_mixed_pbc_cell)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert "/particles/a/position/value" in f
        assert "/particles/b/position/value" in f

    for a, b in zip(s22_all_properties, io_a[:]):
        npt.assert_array_equal(a.get_positions(), b.get_positions())

    for a, b in zip(s22_mixed_pbc_cell, io_b[:]):
        npt.assert_array_equal(a.get_positions(), b.get_positions())


@pytest.mark.parametrize("store", ["linear"])
def test_two_datasets_external(tmp_path, s22_all_properties, s22_mixed_pbc_cell, store):
    with h5py.File(tmp_path / "test.h5", "w") as f:
        io_a = znh5md.IO(file_handle=f, particles_group="a")
        io_b = znh5md.IO(file_handle=f, particles_group="b", store=store)

        io_a.extend(s22_all_properties)
        io_b.extend(s22_mixed_pbc_cell)

        assert len(io_a) == len(s22_all_properties)
        assert len(io_b) == len(s22_mixed_pbc_cell)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        io_a = znh5md.IO(file_handle=f, particles_group="a")
        io_b = znh5md.IO(file_handle=f, particles_group="b")

        assert len(io_a) == len(s22_all_properties)
        assert len(io_b) == len(s22_mixed_pbc_cell)


def test_pbc(tmp_path, s22_mixed_pbc_cell):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22_mixed_pbc_cell)

    pbc = ["periodic" if x else "none" for x in io[0].pbc]

    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert "/particles/atoms/position/value" in f
        assert "/particles/atoms/box" in f

        npt.assert_array_equal(f["particles/atoms/box"].attrs["boundary"], pbc)
        for idx, atom in enumerate(s22_mixed_pbc_cell):
            npt.assert_array_equal(
                f["particles/atoms/box/edges/value"][idx], atom.get_cell()
            )
            # Do we want to rename "pbc" to "boundary" and make if "none"
            # or "periodic" as well? This should only exist if requested
            npt.assert_array_equal(f["particles/atoms/box/pbc/value"][idx], atom.pbc)


def test_save_load_save_load(tmp_path, s22_mixed_pbc_cell):
    io1 = znh5md.IO(tmp_path / "test_1.h5")
    io1.extend(s22_mixed_pbc_cell)

    images = io1[:]
    io2 = znh5md.IO(tmp_path / "test_2.h5")
    io2.extend(images)

    images2 = io2[:]

    assert len(images) == len(images2)


@pytest.mark.parametrize("timestep", [0.5, 1.0])
def test_time_step(tmp_path, s22_mixed_pbc_cell, timestep):
    io = znh5md.IO(tmp_path / "test.h5", timestep=timestep)
    io.extend(s22_mixed_pbc_cell)
    # Do it twice, to check for appending
    io.extend(s22_mixed_pbc_cell)
    images = io[:]
    assert len(images) == len(s22_mixed_pbc_cell) * 2
    # for idx, atoms in enumerate(images):
    #     assert atoms.info["h5md_step"] == idx
    #     assert atoms.info["h5md_time"] == idx * timestep


def test_slicing(tmp_path, s22_mixed_pbc_cell):
    # Create an instance of znh5md.IO and extend it with the test data
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22_mixed_pbc_cell)

    # Ensure the length of the io object matches the input data
    assert len(io) == len(s22_mixed_pbc_cell)

    # Test various slicing cases

    # Single element access
    assert io[0] == s22_mixed_pbc_cell[0]
    assert io[len(io) - 1] == s22_mixed_pbc_cell[len(s22_mixed_pbc_cell) - 1]

    # Simple slices
    assert io[:10] == s22_mixed_pbc_cell[:10]
    assert io[10:20] == s22_mixed_pbc_cell[10:20]
    assert io[-10:] == s22_mixed_pbc_cell[-10:]
    assert io[:-10] == s22_mixed_pbc_cell[:-10]

    # Step slices
    assert io[::2] == s22_mixed_pbc_cell[::2]
    assert io[1::2] == s22_mixed_pbc_cell[1::2]
    # not allowed in h5py
    # assert io[::-1] == s22_mixed_pbc_cell[::-1]  # Reverse

    # Complex slices
    assert io[5:20:3] == s22_mixed_pbc_cell[5:20:3]
    assert io[-20:-5:2] == s22_mixed_pbc_cell[-20:-5:2]

    # Empty slices
    assert io[5:5] == s22_mixed_pbc_cell[5:5]
    assert io[-5:-5] == s22_mixed_pbc_cell[-5:-5]

    # Ensure slicing does not affect original length
    assert len(io) == len(s22_mixed_pbc_cell)
