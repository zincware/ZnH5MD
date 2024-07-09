"""Test against MDAnalysis."""

import ase.build
import MDAnalysis as mda  # noqa:N813
import numpy as np
import numpy.testing as npt
from ase.calculators.singlepoint import SinglePointCalculator
from MDAnalysis.coordinates.H5MD import H5MDReader

import znh5md


def test_mda_read(tmp_path):
    water = ase.build.molecule("H2O")
    water.set_momenta(np.random.rand(*water.positions.shape))
    water.calc = SinglePointCalculator(
        water, forces=np.random.rand(*water.positions.shape)
    )
    io = znh5md.IO(tmp_path / "test.h5", save_units=False, store="time")
    io.append(water)
    io.append(water)

    u = mda.Universe.empty(3, trajectory=True)
    reader = H5MDReader(tmp_path / "test.h5", convert_units=False)
    u.trajectory = reader
    for idx, ts in enumerate(u.trajectory):
        assert ts.frame == idx
        npt.assert_allclose(ts.positions, water.positions)
        npt.assert_allclose(ts.velocities, water.get_velocities())
        npt.assert_allclose(ts.forces, water.calc.results["forces"])


def test_mda_read_convert(tmp_path):
    water = ase.build.molecule("H2O")
    io = znh5md.IO(tmp_path / "test.h5", store="time")
    io.append(water)
    io.append(water)
    u = mda.Universe.empty(3, trajectory=True)
    reader = H5MDReader(tmp_path / "test.h5", convert_units=True)
    u.trajectory = reader
    for idx, ts in enumerate(u.trajectory):
        assert ts.frame == idx
        npt.assert_allclose(ts.positions, water.positions)
