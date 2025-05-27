import numpy.testing as npt
import pytest

import znh5md


@pytest.mark.parametrize("index", ["slice", "list", "int", "iter"])
def test_masked(tmp_path, full_water, index):
    """Test the variable shape off functionality."""
    io = znh5md.IO(tmp_path / "test.h5", variable_shape=False)
    io.extend([full_water for _ in range(10)])

    io_masked = znh5md.IO(
        tmp_path / "test.h5",
        variable_shape=False,
        mask=[0, 2],
    )

    if index == "slice":
        io_masked = io_masked[:]
        assert len(io_masked) == 10
    elif index == "list":
        io_masked = io_masked[[0, 2]]
        assert len(io_masked) == 2
    elif index == "int":
        io_masked = [io_masked[0]]
        assert len(io_masked) == 1
    elif index == "iter":
        assert len(io_masked) == 10

    for atoms in io_masked:
        assert len(atoms) == 2
        for key, val in atoms.arrays.items():
            npt.assert_equal(
                val,
                full_water[[0, 2]].arrays[key],
            )
        for key, val in atoms.info.items():
            npt.assert_equal(
                val,
                full_water[[0, 2]].info[key],
            )
        for key, val in atoms.calc.results.items():
            if isinstance(val, (int, float)):
                assert val == full_water.calc.results[key]
            else:
                npt.assert_equal(
                    val,
                    full_water.calc.results[key][[0, 2]],
                )


def test_masked_variable_shape():
    with pytest.raises(ValueError):
        znh5md.IO(
            "test.h5",
            variable_shape=True,
            mask=[0, 2],
        )


@pytest.mark.parametrize("index", ["slice", "list", "int", "iter"])
def test_masked_slice(tmp_path, full_water, index):
    """Test the variable shape off functionality."""
    io = znh5md.IO(tmp_path / "test.h5", variable_shape=False)
    io.extend([full_water for _ in range(10)])

    io_masked = znh5md.IO(
        tmp_path / "test.h5",
        variable_shape=False,
        mask=slice(0, 2, None),
    )
    if index == "slice":
        io_masked = io_masked[:]
        assert len(io_masked) == 10
    elif index == "list":
        io_masked = io_masked[[0, 2]]
        assert len(io_masked) == 2
    elif index == "int":
        io_masked = [io_masked[0]]
        assert len(io_masked) == 1
    elif index == "iter":
        assert len(io_masked) == 10

    for atoms in io_masked:
        assert len(atoms) == 2
