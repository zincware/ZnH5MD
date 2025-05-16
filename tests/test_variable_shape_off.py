import numpy.testing as npt

import znh5md


def test_variable_shape_off(tmp_path, full_water):
    """Test the variable shape off functionality."""
    io = znh5md.IO(tmp_path / "test.h5", variable_shape=False)
    io.extend([full_water for _ in range(10)])

    assert len(io) == 10
    assert len(list(io)) == 10
    for atoms in io:
        assert atoms == full_water
        for key, val in atoms.arrays.items():
            npt.assert_equal(
                val,
                full_water.arrays[key],
            )
        for key, val in atoms.info.items():
            npt.assert_equal(
                val,
                full_water.info[key],
            )
        for key, val in atoms.calc.results.items():
            npt.assert_equal(
                val,
                full_water.calc.results[key],
            )
