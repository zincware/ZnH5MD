import pytest
import znh5md
import numpy.testing as npt


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
    
    rio = znh5md.IO(tmp_path / "test.h5", keys=["position"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())


def test_restricted_read_empty(tmp_path):
    with pytest.raises(ValueError):
        znh5md.IO(tmp_path / "test.h5", keys=[])


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
    
    rio = znh5md.IO(tmp_path / "test.h5", keys=["position", "box"])
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
    
    rio = znh5md.IO(tmp_path / "test.h5", keys=["position", "potential_energy"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.get_potential_energy() == ref.get_potential_energy()
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

    rio = znh5md.IO(tmp_path / "test.h5", keys=["position", "force"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        npt.assert_array_equal(atoms.get_forces(), ref.get_forces())
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

    rio = znh5md.IO(tmp_path / "test.h5", keys=["position", "mlip_energy"])
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

    rio = znh5md.IO(tmp_path / "test.h5", keys=["position", "mlip_forces"])
    assert len(rio) == len(rio[:])
    assert len(rio) == len(frames)

    for atoms, ref in zip(rio, frames):
        assert atoms.calc is None
        assert atoms.info == {}
        assert atoms.arrays.keys() == {"numbers", "positions", "mlip_forces"}
        npt.assert_array_equal(atoms.get_atomic_numbers(), ref.get_atomic_numbers())
        npt.assert_array_equal(atoms.get_positions(), ref.get_positions())
        npt.assert_array_equal(atoms.arrays["mlip_forces"], ref.arrays["mlip_forces"])