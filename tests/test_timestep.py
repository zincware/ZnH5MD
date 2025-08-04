import warnings

import ase.build
import numpy as np
import pytest

import znh5md


@pytest.fixture
def trajectory_time(tmp_path):
    """Fixture to create a trajectory with time information."""
    io = znh5md.IO(
        tmp_path / "test_time_step.h5", store="time", export_timestep=True, timestep=0.5
    )
    for _ in range(1, 10):
        atoms = ase.build.molecule("H2O")
        io.append(atoms)
    return io


@pytest.fixture
def trajectory_linear(tmp_path):
    """Fixture to create a trajectory with linear storage."""
    io = znh5md.IO(
        tmp_path / "test_linear_step.h5",
        store="linear",
        export_timestep=True,
        timestep=0.5,
    )
    for _ in range(1, 10):
        atoms = ase.build.molecule("H2O")
        io.append(atoms)
    return io


@pytest.mark.parametrize("trajectory", ["trajectory_time", "trajectory_linear"])
def test_time_step(trajectory, request):
    """Test that the time step is correctly stored and retrieved."""
    io = request.getfixturevalue(trajectory)

    assert io[0].info["timestep"] == 0
    assert io[1].info["timestep"] == 0.5

    timesteps = [atoms.info["timestep"] for atoms in io[:]]
    assert timesteps == list(np.arange(0, 9) * 0.5)

    timesteps = [atoms.info["timestep"] for atoms in io[::2]]
    assert timesteps == list(np.arange(0, 9) * 0.5)[::2]

    timesteps = [atoms.info["timestep"] for atoms in io[1:3]]
    assert timesteps == [0.5, 1.0]

    timesteps = [atoms.info["timestep"] for atoms in io[[2, 6, 7]]]
    assert timesteps == list(np.arange(0, 9)[[2, 6, 7]] * 0.5)

    assert io.timestep == 0.5


# def test_h5md_time(tmp_path):
#     io = znh5md.IO(tmp_path / "test_time_step.h5", store="time")
#     for step in range(1, 10):
#         atoms = ase.build.molecule("H2O")
#         atoms.calc = SinglePointCalculator(atoms, energy=step * 0.1)
#         atoms.info["h5md_step"] = step
#         atoms.info["h5md_time"] = step * 0.5
#         io.append(atoms)

#     for idx, atoms in enumerate(io[:]):
#         assert atoms.info["h5md_step"] == idx + 1
#         assert atoms.info["h5md_time"] == (idx + 1) * 0.5
#         assert atoms.get_potential_energy() == (idx + 1) * 0.1

#     with h5py.File(tmp_path / "test_time_step.h5") as f:
#         npt.assert_array_equal(
#             f["particles/atoms/position/time"][:],
#             np.arange(1, 10) * 0.5
#         )
#         npt.assert_array_equal(f["particles/atoms/position/step"][:],
#                                 np.arange(1, 10))
#         npt.assert_array_equal(
#             f["observables/atoms/energy/time"][:],
#               np.arange(1, 10) * 0.5
#         )
#         npt.assert_array_equal(f["observables/atoms/energy/step"][:],
#                                np.arange(1, 10))
#         npt.assert_array_equal(
#             f["observables/atoms/energy/value"][:],
#               np.arange(1, 10) * 0.1
#         )


# def test_inconsistent_time(tmp_path):
#     images = [ase.build.molecule("H2O") for _ in range(10)]
#     images[5].info["h5md_time"] = 0.5

#     io = znh5md.IO(tmp_path / "test_inconsistent_time.h5", store="time")
#     with pytest.raises(ValueError):
#         io.extend(images)


# def test_inconsistent_step(tmp_path):
#     images = [ase.build.molecule("H2O") for _ in range(10)]
#     images[5].info["h5md_step"] = 5

#     io = znh5md.IO(tmp_path / "test_inconsistent_step.h5", store="time")
#     with pytest.raises(ValueError):
#         io.extend(images)


# def test_wrong_store(tmp_path):
#     io = znh5md.IO(tmp_path / "test_wrong_store.h5", store="linear")
#     atoms = ase.build.molecule("H2O")
#     atoms.info["h5md_step"] = 1
#     atoms.info["h5md_time"] = 0.1

#     with pytest.warns(UserWarning, match="time is ignored in 'linear' storage mode"):
#         io.append(atoms)


def test_no_warn_correct(tmp_path):
    io = znh5md.IO(tmp_path / "test_wrong_store.h5", store="linear")
    atoms = ase.build.molecule("H2O")

    # Ensure no warning is issued
    # https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.append(atoms)
