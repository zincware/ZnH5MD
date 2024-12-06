from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md


def test_single_entry_info(tmp_path):
    # Test a special case where only the first config has the key
    # which caused an error in the past
    io = znh5md.IO(tmp_path / "test.h5")
    water = molecule("H2O")
    water.info["density"] = 0.997
    io.append(water)
    del water.info["density"]
    io.extend([water for _ in range(5)])
    assert len(io) == 6
    assert len(list(io)) == 6
    # assert len(io[:]) == 6

def test_single_entry_arrays(tmp_path):
    # Test a special case where only the first config has the key
    # which caused an error in the past
    io = znh5md.IO(tmp_path / "test.h5")
    water = molecule("H2O")
    water.arrays["density"] = [0.997, 0.998, 0.999]
    io.append(water)
    del water.arrays["density"]
    io.extend([water for _ in range(5)])
    assert len(io) == 6
    assert len(list(io)) == 6
    # assert len(io[:]) == 6


def test_single_entry_calc(tmp_path):
    # Test a special case where only the first config has the key
    # which caused an error in the past
    io = znh5md.IO(tmp_path / "test.h5")
    water = molecule("H2O")
    water.calc = SinglePointCalculator(water, energy=0.0, forces=[0.0, 0.0, 0.0])
    io.append(water)
    water.calc = None
    io.extend([water for _ in range(5)])
    assert len(io) == 6
    assert len(list(io)) == 6
    # assert len(io[:]) == 6
