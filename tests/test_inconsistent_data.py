import typing as t

import numpy as np
import numpy.testing as npt
import pytest

import znh5md


def test_keys_missing(tmp_path, s22, s22_energy_forces):
    io = znh5md.IO(tmp_path / "test.h5")

    images = s22_energy_forces + s22
    io.extend(images)
    assert len(io) == len(images)
    assert len(list(io)) == len(images)

    for a, b in zip(images, io):
        assert a == b
        if b.calc is not None:
            for key in b.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])
        else:
            assert a.calc is None


@pytest.mark.parametrize("state", ["before", "middle", "after"])
def test_velocity(tmp_path, s22, state: t.Literal["before", "middle", "after"]):
    io = znh5md.IO(tmp_path / "test.h5")

    velocity = None

    images = s22
    if state == "before":
        velocity = np.random.random((len(images[0]), 3)) * 0.1
        images[0].set_velocities(velocity)
    elif state == "middle":
        velocity = np.random.random((len(images[1]), 3)) * 0.1
        images[1].set_velocities(velocity)
    elif state == "after":
        velocity = np.random.random((len(images[-1]), 3)) * 0.1
        images[-1].set_velocities(velocity)

    for atoms in images:
        io.append(atoms)
    assert len(io) == len(images)
    assert len(list(io)) == len(images)
    if state == "before":
        npt.assert_array_almost_equal(io[0].get_velocities(), velocity)
        npt.assert_array_almost_equal(io[:][0].get_velocities(), velocity)
    elif state == "middle":
        npt.assert_array_almost_equal(io[1].get_velocities(), velocity)
        npt.assert_array_almost_equal(io[:][1].get_velocities(), velocity)
    elif state == "after":
        npt.assert_array_almost_equal(io[-1].get_velocities(), velocity)
        npt.assert_array_almost_equal(io[:][-1].get_velocities(), velocity)
