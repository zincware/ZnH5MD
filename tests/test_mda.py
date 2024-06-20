"""Test against MDAnalysis."""
import MDAnalysis as mda
from MDAnalysis.coordinates.H5MD import H5MDReader
import ase.build
import znh5md

def test_mda_read(tmp_path):
    water = ase.build.molecule("H2O")
    io = znh5md.IO(tmp_path / "test.h5")
    io.append(water)

    u = mda.Universe.empty(3, trajectory=True)
    reader = H5MDReader(tmp_path / "test.h5", convert_units=False)
    u.trajectory = reader
