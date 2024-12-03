import ase.build
import numpy as np

import znh5md


def test_append_new_calc(tmp_path, s22, s22_energy):
    io = znh5md.IO(tmp_path / "test.h5")

    io.extend(s22)
    io.extend(s22_energy)
    io.extend(s22)
    io.extend(s22_energy)
    io.extend(s22)
    io.extend(s22_energy)

    assert len(s22) == 22
    assert len(s22_energy) == 22
    assert len(io) == len(s22) * 3 + len(s22_energy) * 3

    assert len(list(io)) == len(s22) * 3 + len(s22_energy) * 3

    for a, b in zip(io[:22], s22):
        assert a.calc is None
        assert b.calc is None

    for a, b in zip(io[22:44], s22_energy):
        assert a.calc is not None
        assert b.calc is not None
        assert a.calc.results == b.calc.results

    for a, b in zip(io[44:], s22):
        assert a.calc is None
        assert b.calc is None

    for a, b in zip(io[66:], s22_energy):
        assert a.calc is not None
        assert b.calc is not None
        assert a.calc.results == b.calc.results

    for a, b in zip(io[88:], s22):
        assert a.calc is None
        assert b.calc is None

    for a, b in zip(io[110:], s22_energy):
        assert a.calc is not None
        assert b.calc is not None
        assert a.calc.results == b.calc.results


def test_append_new_keys_info(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    water = ase.build.molecule("H2O")
    water.info["key1"] = np.random.random()

    io.append(ase.build.molecule("H2O"))
    io.append(water)
    io.append(ase.build.molecule("H2O"))
    io.append(water)

    assert len(list(io)) == 4

    assert len(io) == 4
    assert "key1" not in io[0].info
    assert "key1" in io[1].info
    assert "key1" not in io[2].info
    assert "key1" in io[3].info

    assert io[1].info["key1"] == water.info["key1"]


def test_append_new_keys_info_non_ascii(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    water = ase.build.molecule("H2O")
    water.info["key1"] = "γC"

    io.append(ase.build.molecule("H2O"))
    io.append(water)
    io.append(ase.build.molecule("H2O"))
    io.append(water)

    assert len(list(io)) == 4

    assert len(io) == 4
    assert "key1" not in io[0].info
    assert "key1" in io[1].info
    assert "key1" not in io[2].info
    assert "key1" in io[3].info

    assert io[1].info["key1"] == "γC" == io[3].info["key1"]


def test_append_new_keys_arrays(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    water = ase.build.molecule("H2O")
    water.arrays["key1"] = np.random.rand(len(water), 3)

    io.append(ase.build.molecule("H2O"))
    io.append(water)
    io.append(ase.build.molecule("H2O"))
    io.append(water)

    assert len(list(io)) == 4

    assert len(io) == 4
    assert "key1" not in io[0].arrays
    assert "key1" in io[1].arrays
    assert "key1" not in io[2].arrays
    assert "key1" in io[3].arrays

    assert np.array_equal(io[1].arrays["key1"], water.arrays["key1"])
