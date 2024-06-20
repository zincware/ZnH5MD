import znh5md
import ase
import numpy as np


def test_IO(cu_file):
    io = znh5md.IO(cu_file)

    assert isinstance(io[0], ase.Atoms)
    assert io[0].get_atomic_numbers().tolist() == 108 * [29]
    positions = io[0].get_positions()
    assert io[0].pbc.all()

    assert isinstance(io[1:2], list)
    for atoms in io[1:2]:
        assert isinstance(atoms, ase.Atoms)
        assert atoms.get_atomic_numbers().tolist() == 108 * [29]
        # check that positions change
        assert not np.array_equal(atoms.get_positions(), positions)
        positions = atoms.get_positions()
