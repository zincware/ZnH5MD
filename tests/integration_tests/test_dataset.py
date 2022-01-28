import pytest
import tensorflow as tf
from zinchub import DataHub

from znh5md.templates import LammpsH5MD


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaClH5MD")
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def traj(traj_file):
    return LammpsH5MD(traj_file)


def test_position_df_type(traj):
    assert isinstance(traj.position.get_dataset(), tf.data.Dataset)


def test_position_df_element_spec(traj):
    assert traj.position.get_dataset().element_spec == {
        "step": tf.TensorSpec(shape=(), dtype=tf.int32),
        "time": tf.TensorSpec(shape=(), dtype=tf.float64),
        "value": tf.TensorSpec(shape=(1000, 3), dtype=tf.float64),
    }


def test_position_df_access(traj):
    element = next(iter(traj.position.get_dataset().batch(4)))

    assert element["step"].shape == (4,)
    assert element["time"].shape == (4,)
    assert element["value"].shape == (4, 1000, 3)


def test_position_value_access(traj):
    # default prefetch along axis = 0
    element = next(iter(traj.position.value.get_dataset().batch(4)))

    assert element.shape == (4, 1000, 3)


def test_position_value_axis_1(traj):
    # change to prefetch along axis = 1 / particles
    ds = traj.position.value.get_dataset(axis=1).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 201, 3)


def test_position_value_axis_2(traj):
    # change to prefetch along axis = 2 / coordinate axis
    ds = traj.position.value.get_dataset(axis=2)
    element = next(iter(ds))
    assert element.shape == (201, 1000)


def test_positions_value_selection(traj):
    ds = traj.position.value.get_dataset(selection=[0, 1, 2, 3, 4]).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)
    # and now with a slice
    ds = traj.position.value.get_dataset(selection=slice(5)).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)


def test_positions_value_selection_axis_1(traj):
    ds = traj.position.value.get_dataset(axis=1, selection=[0, 1, 2, 3, 4]).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)
    # and now with a slice
    ds = traj.position.value.get_dataset(axis=1, selection=slice(5)).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)


def test_positions_value_selection_batched(traj):
    ds = traj.position.value.get_dataset(selection=[0, 1, 2, 3, 4], prefetch=16).batch(
        4
    )
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)
    # and now with a slice
    ds = traj.position.value.get_dataset(selection=slice(5), prefetch=16).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)


def test_positions_value_selection_axis_1_batched(traj):
    ds = traj.position.value.get_dataset(
        axis=1, selection=[0, 1, 2, 3, 4], prefetch=16
    ).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)
    # and now with a slice
    ds = traj.position.value.get_dataset(axis=1, selection=slice(5), prefetch=16).batch(
        4
    )
    element = next(iter(ds))
    assert element.shape == (4, 5, 3)


def test_positions_multi_axis(traj):
    ds = traj.position.value.get_dataset(axis=(0, 1)).batch(4)
    element = next(iter(ds))
    assert element.shape == (4, 3)


def test_positions_invalid_axis(traj):
    with pytest.raises(ValueError):
        _ = traj.position.value.get_dataset(axis=(0, 1, 2))


def test_loop_indices(traj):
    ds = traj.position.value.get_dataset(
        loop_indices=[1, 2, 3], selection=[0, 1, 2, 3, 4]
    ).batch(32)
    element = next(iter(ds))
    # the loop only contains 3 indices
    assert element.shape == (3, 5, 3)


def test_loop_indices_axis_1(traj):
    ds = traj.position.value.get_dataset(
        axis=1, loop_indices=[1, 2, 3], selection=[0, 1, 2, 3, 4]
    ).batch(32)
    element = next(iter(ds))
    # the loop only contains 3 indices
    assert element.shape == (3, 5, 3)


def test_position_batch_and_loop_indices_invalid(traj):
    with pytest.raises(ValueError):
        _ = traj.position.value.get_dataset(loop_indices=[1, 2], prefetch=16)
