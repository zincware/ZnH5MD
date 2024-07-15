import ase.build
import h5py
import numpy.testing as npt
import pytest

import znh5md


@pytest.mark.parametrize("pbc_group", [True, False])
def test_pbc_group_water(tmp_path, pbc_group):
    io = znh5md.IO(tmp_path / "test.h5", pbc_group=pbc_group)
    io.append(ase.build.molecule("H2O"))

    io2 = znh5md.IO(tmp_path / "test.h5")
    assert io2[0] == io[0]
    assert io2[0] == ase.build.molecule("H2O")
    npt.assert_array_equal(io2[0].get_pbc(), io[0].get_pbc())

    with h5py.File(tmp_path / "test.h5", "r") as f:
        if pbc_group:
            assert "pbc" in f["particles/atoms/box"]
        else:
            assert "pbc" not in f["particles/atoms/box"]
