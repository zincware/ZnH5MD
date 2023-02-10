import numpy as np
import numpy.testing as npt

import znh5md


def test_version():
    assert znh5md.__version__ == "0.1.0"


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
