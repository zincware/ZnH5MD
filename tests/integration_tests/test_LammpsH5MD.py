import numpy as np
import pytest
from zinchub import DataHub

from znh5md.templates.lammps import LammpsH5MD


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaClH5MD", tag="v0.1.1"
    )
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def traj(traj_file):
    return LammpsH5MD(traj_file)


def test_position_shape(traj):
    assert traj.position.shape == (201, 1000, 3)


def test_position_dtype(traj):
    assert traj.position.dtype == float


def test_position_slice(traj):
    assert traj.position[:5, :5].shape == (5, 5, 3)
    assert traj.position[slice(5), slice(5)].shape == (5, 5, 3)


def test_position_value_shape(traj):
    assert traj.position.value.shape == (201, 1000, 3)


def test_position_value_dtype(traj):
    assert traj.position.value.dtype == float


def test_position_value_slice(traj):
    assert traj.position.value[:5, :5].shape == (5, 5, 3)
    assert traj.position.value[slice(5), slice(5)].shape == (5, 5, 3)


def test_position_step_shape(traj):
    assert traj.position.step.shape == (201,)


def test_position_step_dtype(traj):
    assert traj.position.step.dtype == np.dtype("int32")


def test_position_step_slice(traj):
    assert traj.position.step[:5].shape == (5,)
    assert traj.position.step[slice(5)].shape == (5,)


def test_position_time_shape(traj):
    assert traj.position.time.shape == (201,)


def test_position_time_dtype(traj):
    assert traj.position.time.dtype == float


def test_position_time_slice(traj):
    assert traj.position.time[:5].shape == (5,)
    assert traj.position.time[slice(5)].shape == (5,)


def test_all_available(traj):
    assert traj.box[:5].shape == (5, 3)
    assert traj.force[:5].shape == (5, 1000, 3)
    assert traj.image[:5].shape == (5, 1000, 3)
    assert traj.position[:5].shape == (5, 1000, 3)
    assert traj.species[:5].shape == (5, 1000)
    assert traj.velocity[:5].shape == (5, 1000, 3)
