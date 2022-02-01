import pytest
from zinchub import DataHub

from znh5md.core.exceptions import GroupNotFound
from znh5md.core.h5md import H5MDProperty
from znh5md.templates.base import H5MDTemplate


class MockH5MD(H5MDTemplate):
    # these exist
    position = H5MDProperty(group="particles/all/position")
    species = H5MDProperty(group="particles/all/species")
    velocity = H5MDProperty(group="particles/all/velocity")
    # these do not
    a = H5MDProperty(group="particles/all/a")
    b = H5MDProperty(group="particles/all/b")
    c = H5MDProperty(group="particles/all/c")


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaClH5MD")
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def traj(traj_file) -> MockH5MD:
    return MockH5MD(traj_file)


def test_get_groups(traj):
    assert traj.get_groups() == ["position", "species", "velocity"]


def test_group_not_found(traj):
    with pytest.raises(GroupNotFound):
        _ = traj.a[0]

    with pytest.raises(GroupNotFound):
        _ = traj.a.get_dataset()

    with pytest.raises(GroupNotFound):
        _ = traj.a.value.get_dataset()
