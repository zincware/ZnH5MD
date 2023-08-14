import numpy as np
import numpy.testing as npt

import znh5md


def test_version():
    assert znh5md.__version__ == "0.1.8"


def test_shape(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    assert traj.position.value.shape == (100, 10, 3)
    assert traj.position.time.shape == (100,)
    assert traj.position.step.shape == (100,)
    assert traj.position.species.shape == (100, 10)


def test_compute(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    npt.assert_array_equal(
        traj.position.value.compute(), np.arange(100 * 10 * 3).reshape((100, 10, 3))
    )
    npt.assert_array_equal(traj.position.time.compute(), np.linspace(0, 1, 100))
    npt.assert_array_equal(traj.position.step.compute(), np.arange(100))
    npt.assert_array_equal(
        traj.position.species.compute(),
        np.concatenate([np.ones((100, 5)), 2 * np.ones((100, 5))], axis=1),
    )


def test_slice_shape(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    # species 1
    sliced = traj.position.slice_by_species([1])
    assert sliced.value.shape == (100, 5, 3)
    assert sliced.time.shape == (100,)
    assert sliced.step.shape == (100,)
    assert sliced.species.shape == (100, 5)
    # species 2
    sliced = traj.position.slice_by_species([2])
    assert sliced.value.shape == (100, 5, 3)
    assert sliced.time.shape == (100,)
    assert sliced.step.shape == (100,)
    assert sliced.species.shape == (100, 5)
    # species 1 and 2
    sliced = traj.position.slice_by_species([1, 2])
    assert sliced.value.shape == (100, 10, 3)
    assert sliced.time.shape == (100,)
    assert sliced.step.shape == (100,)
    assert sliced.species.shape == (100, 10)


def test_slice_compute(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    # species 1
    sliced = traj.position.slice_by_species([1])
    npt.assert_array_equal(
        sliced.value.compute(), np.arange(100 * 10 * 3).reshape((100, 10, 3))[:, :5, :]
    )
    npt.assert_array_equal(sliced.time.compute(), np.linspace(0, 1, 100))
    npt.assert_array_equal(sliced.step.compute(), np.arange(100))
    npt.assert_array_equal(sliced.species.compute(), np.ones((100, 5)))
    # species 2
    sliced = traj.position.slice_by_species([2])
    npt.assert_array_equal(
        sliced.value.compute(), np.arange(100 * 10 * 3).reshape((100, 10, 3))[:, 5:, :]
    )
    npt.assert_array_equal(sliced.time.compute(), np.linspace(0, 1, 100))
    npt.assert_array_equal(sliced.step.compute(), np.arange(100))
    npt.assert_array_equal(sliced.species.compute(), 2 * np.ones((100, 5)))
    # species 1 and 2
    sliced = traj.position.slice_by_species([1, 2])
    npt.assert_array_equal(
        sliced.value.compute(), np.arange(100 * 10 * 3).reshape((100, 10, 3))
    )
    npt.assert_array_equal(sliced.time.compute(), np.linspace(0, 1, 100))
    npt.assert_array_equal(sliced.step.compute(), np.arange(100))
    npt.assert_array_equal(
        sliced.species.compute(),
        np.concatenate([np.ones((100, 5)), 2 * np.ones((100, 5))], axis=1),
    )


def test_batch_shape(example_h5):
    traj = znh5md.DaskH5MD(example_h5)

    batches = list(traj.position.batch(67, axis=0))
    assert len(batches) == 2
    assert batches[0].value.shape == (67, 10, 3)
    assert batches[0].time.shape == (67,)
    assert batches[0].step.shape == (67,)
    assert batches[0].species.shape == (67, 10)
    assert batches[1].value.shape == (33, 10, 3)
    assert batches[1].time.shape == (33,)
    assert batches[1].step.shape == (33,)
    assert batches[1].species.shape == (33, 10)


def test_batch_compute(example_h5):
    traj = znh5md.DaskH5MD(example_h5)

    batches = list(traj.position.batch(67, axis=0))
    assert len(batches) == 2
    npt.assert_array_equal(
        batches[0].value.compute(), traj.position.value.compute()[:67, :, :]
    )
    npt.assert_array_equal(batches[0].time.compute(), traj.position.time.compute()[:67])
    npt.assert_array_equal(batches[0].step.compute(), traj.position.step.compute()[:67])
    npt.assert_array_equal(
        batches[0].species.compute(), traj.position.species.compute()[:67]
    )

    npt.assert_array_equal(
        batches[1].value.compute(), traj.position.value.compute()[67:, :, :]
    )
    npt.assert_array_equal(batches[1].time.compute(), traj.position.time.compute()[67:])
    npt.assert_array_equal(batches[1].step.compute(), traj.position.step.compute()[67:])
    npt.assert_array_equal(
        batches[1].species.compute(), traj.position.species.compute()[67:]
    )


def test_slice_batch_shape(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    # species 1
    sliced = traj.position.slice_by_species([1])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    assert batches[0].value.shape == (67, 5, 3)
    assert batches[0].time.shape == (67,)
    assert batches[0].step.shape == (67,)
    assert batches[0].species.shape == (67, 5)
    assert batches[1].value.shape == (33, 5, 3)
    assert batches[1].time.shape == (33,)
    assert batches[1].step.shape == (33,)
    assert batches[1].species.shape == (33, 5)
    # species 2
    sliced = traj.position.slice_by_species([2])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    assert batches[0].value.shape == (67, 5, 3)
    assert batches[0].time.shape == (67,)
    assert batches[0].step.shape == (67,)
    assert batches[0].species.shape == (67, 5)
    assert batches[1].value.shape == (33, 5, 3)
    assert batches[1].time.shape == (33,)
    assert batches[1].step.shape == (33,)
    assert batches[1].species.shape == (33, 5)
    # species 1 and 2
    sliced = traj.position.slice_by_species([1, 2])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    assert batches[0].value.shape == (67, 10, 3)
    assert batches[0].time.shape == (67,)
    assert batches[0].step.shape == (67,)
    assert batches[0].species.shape == (67, 10)
    assert batches[1].value.shape == (33, 10, 3)
    assert batches[1].time.shape == (33,)
    assert batches[1].step.shape == (33,)
    assert batches[1].species.shape == (33, 10)


def test_slice_batch_compute(example_h5):
    traj = znh5md.DaskH5MD(example_h5)
    # species 1
    sliced = traj.position.slice_by_species([1])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    npt.assert_array_equal(
        batches[0].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[:67, :5, :],
    )
    npt.assert_array_equal(batches[0].time.compute(), traj.position.time.compute()[:67])
    npt.assert_array_equal(batches[0].step.compute(), traj.position.step.compute()[:67])
    npt.assert_array_equal(
        batches[0].species.compute(), traj.position.species.compute()[:67, :5]
    )
    npt.assert_array_equal(
        batches[1].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[67:, :5, :],
    )
    npt.assert_array_equal(batches[1].time.compute(), traj.position.time.compute()[67:])
    npt.assert_array_equal(batches[1].step.compute(), traj.position.step.compute()[67:])
    npt.assert_array_equal(
        batches[1].species.compute(), traj.position.species.compute()[67:, :5]
    )

    # species 2
    sliced = traj.position.slice_by_species([2])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    npt.assert_array_equal(
        batches[0].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[:67, 5:, :],
    )
    npt.assert_array_equal(batches[0].time.compute(), traj.position.time.compute()[:67])
    npt.assert_array_equal(batches[0].step.compute(), traj.position.step.compute()[:67])
    npt.assert_array_equal(
        batches[0].species.compute(), traj.position.species.compute()[:67, 5:]
    )
    npt.assert_array_equal(
        batches[1].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[67:, 5:, :],
    )
    npt.assert_array_equal(batches[1].time.compute(), traj.position.time.compute()[67:])
    npt.assert_array_equal(batches[1].step.compute(), traj.position.step.compute()[67:])
    npt.assert_array_equal(
        batches[1].species.compute(), traj.position.species.compute()[67:, 5:]
    )

    # species 1 and 2
    sliced = traj.position.slice_by_species([1, 2])
    batches = list(sliced.batch(67, axis=0))
    assert len(batches) == 2
    npt.assert_array_equal(
        batches[0].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[:67, :, :],
    )
    npt.assert_array_equal(batches[0].time.compute(), traj.position.time.compute()[:67])
    npt.assert_array_equal(batches[0].step.compute(), traj.position.step.compute()[:67])
    npt.assert_array_equal(
        batches[0].species.compute(), traj.position.species.compute()[:67, :]
    )
    npt.assert_array_equal(
        batches[1].value.compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[67:, :, :],
    )
    npt.assert_array_equal(batches[1].time.compute(), traj.position.time.compute()[67:])
    npt.assert_array_equal(batches[1].step.compute(), traj.position.step.compute()[67:])
    npt.assert_array_equal(
        batches[1].species.compute(), traj.position.species.compute()[67:, :]
    )


def test_slice(example_h5):
    traj = znh5md.DaskH5MD(example_h5)

    npt.assert_array_equal(
        traj.position.value[::2].compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[::2],
    )
    npt.assert_array_equal(
        traj.position.value[[1, 2, 5]].compute(),
        np.arange(100 * 10 * 3).reshape((100, 10, 3))[[1, 2, 5]],
    )
