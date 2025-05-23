import numpy as np
import numpy.testing as npt
import pytest

import znh5md


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_read_position(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        # 'numbers' is always implicitly added when reading, even if only 'positions' is requested.
        # This is because 'numbers' defines the structure of the atom.
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


def test_restricted_read_empty(tmp_path):
    with pytest.raises(ValueError):
        znh5md.IO(tmp_path / "test.h5", include=[])


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_mixed_pbc_cell",
    ],
)
def test_restricted_read_box(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position", "box"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())
        npt.assert_array_equal(atoms.get_cell(), ref.get_cell())
        npt.assert_array_equal(atoms.get_pbc(), ref.get_pbc())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_read_potential_energy(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position", "potential_energy"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.get_potential_energy() == ref.get_potential_energy()
        assert atoms.calc is not None
        assert atoms.calc.results.keys() == {"energy"}
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_read_forces(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position", "force"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        npt.assert_array_equal(atoms.get_forces(), ref.get_forces())
        assert atoms.calc is not None
        assert atoms.calc.results.keys() == {"forces"}
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_read_info(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position", "mlip_energy"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info.keys() == {"mlip_energy"}
        assert atoms.info["mlip_energy"] == ref.info["mlip_energy"]
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_read_arrays(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(frames)

    rio = znh5md.IO(tmp_path / "test.h5", include=["position", "mlip_forces"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions", "mlip_forces"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())
        npt.assert_array_equal(atoms.arrays["mlip_forces"], ref.arrays["mlip_forces"])


# New tests for restricted writing


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_write_position(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position'
    io = znh5md.IO(tmp_path / "test.h5", include=["position"])
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {
            "numbers",
            "positions",
        }  # 'numbers' is always present
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_mixed_pbc_cell",
    ],
)
def test_restricted_write_box(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position' and 'box'
    io = znh5md.IO(tmp_path / "test.h5", include=["position", "box"])
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {
            "numbers",
            "positions",
        }  # 'numbers' is always present
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())
        npt.assert_array_equal(atoms.get_cell(), ref.get_cell())
        npt.assert_array_equal(atoms.get_pbc(), ref.get_pbc())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_write_potential_energy(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position' and 'potential_energy'
    io = znh5md.IO(
        tmp_path / "test.h5", include=["position", "energy"]
    )  # TODO: either use ASE keys of H5MD keys but don't mix them
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is not None
        assert atoms.calc.results.keys() == {"energy"}
        assert np.isclose(atoms.get_potential_energy(), ref.get_potential_energy())
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_write_forces(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position' and 'force'
    io = znh5md.IO(
        tmp_path / "test.h5", include=["position", "forces"]
    )  # TODO: either use ASE keys of H5MD keys but don't mix them
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is not None
        assert atoms.calc.results.keys() == {"forces"}
        npt.assert_array_equal(atoms.get_forces(), ref.get_forces())
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_write_info(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position' and 'mlip_energy' from info
    io = znh5md.IO(tmp_path / "test.h5", include=["position", "mlip_energy"])
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info.keys() == {"mlip_energy"}
        assert atoms.info["mlip_energy"] == ref.info["mlip_energy"]
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


@pytest.mark.parametrize(
    "dataset",
    [
        "s22_info_arrays_calc",
    ],
)
def test_restricted_write_arrays(tmp_path, dataset, request):
    frames = request.getfixturevalue(dataset)
    # Write only 'position' and 'mlip_forces' from arrays
    io = znh5md.IO(tmp_path / "test.h5", include=["position", "mlip_forces"])
    io.extend(frames)

    # Read everything back
    rio = znh5md.IO(tmp_path / "test.h5")
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions", "mlip_forces"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())
        npt.assert_array_equal(atoms.arrays["mlip_forces"], ref.arrays["mlip_forces"])
