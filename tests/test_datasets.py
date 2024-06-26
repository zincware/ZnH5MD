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
        # "s22_illegal_calc_results",
        "water",
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

        assert set(a.arrays) == set(b.arrays)
        for key in a.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])

        assert set(a.info) == set(b.info)
        for key in a.info:
            npt.assert_array_equal(a.info[key], b.info[key])


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_datasets_h5py(tmp_path, dataset, request):
    images = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(images)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert "particles/atoms/position/value" in f
        assert "particles/atoms/species/value" in f
        assert "particles/atoms/force/value" in f
        assert "observables/atoms/force/value" not in f
        assert "particles/atoms/velocity/value" in f
        assert "observables/atoms/velocity/value" not in f

        assert "particles/atoms/energy/value" not in f
        assert "observables/atoms/energy/value" in f

        assert "particles/atoms/mlip_forces/value" in f
        assert "particles/atoms/mlip_forces_2/value" in f

        assert "observables/atoms/mlip_energy/value" in f
        assert "observables/atoms/mlip_energy_2/value" in f
        assert "observables/atoms/mlip_stress/value" in f

        assert f["particles/atoms/velocity/value"].attrs["unit"] == "Angstrom/fs"
        assert f["particles/atoms/force/value"].attrs["unit"] == "eV/Angstrom"
        # assert f["observables/atoms/energy/value"].attrs["unit"] == "eV"

        npt.assert_array_equal(
            f["particles/atoms/box"].attrs["boundary"], ["none", "none", "none"]
        )


def test_two_datasets(tmp_path, s22_all_properties, s22_mixed_pbc_cell):
    io_a = znh5md.IO(tmp_path / "test.h5", particle_group="a")
    io_b = znh5md.IO(tmp_path / "test.h5", particle_group="b")
    io_a.extend(s22_all_properties)
    io_b.extend(s22_mixed_pbc_cell)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        assert "/particles/a/position/value" in f
        assert "/particles/b/position/value" in f

    for a, b in zip(s22_all_properties, io_a[:]):
        npt.assert_array_equal(a.get_positions(), b.get_positions())

    for a, b in zip(s22_mixed_pbc_cell, io_b[:]):
        npt.assert_array_equal(a.get_positions(), b.get_positions())


def test_two_datasets_external(tmp_path, s22_all_properties, s22_mixed_pbc_cell):
    with h5py.File(tmp_path / "test.h5", "w") as f:
        io_a = znh5md.IO(file_handle=f, particle_group="a")
        io_b = znh5md.IO(file_handle=f, particle_group="b")

        io_a.extend(s22_all_properties)
        io_b.extend(s22_mixed_pbc_cell)

        assert len(io_a) == len(s22_all_properties)
        assert len(io_b) == len(s22_mixed_pbc_cell)

    with h5py.File(tmp_path / "test.h5", "r") as f:
        io_a = znh5md.IO(file_handle=f, particle_group="a")
        io_b = znh5md.IO(file_handle=f, particle_group="b")

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
