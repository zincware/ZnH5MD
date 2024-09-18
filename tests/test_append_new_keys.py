import znh5md
import numpy as np
import ase.build


def test_append_data_with_new_info(tmp_path, s22, s22_energy):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22)
    io.extend(s22_energy)
    assert len(s22) == 22
    assert len(s22_energy) == 22
    assert len(io) == 44

    # for a, b in zip(io, s22 + s22_energy):
    #     assert a.calc == b.calc

    for a, b in zip(io[: len(s22)], s22):
        assert a.calc is None
        assert b.calc is None

    for a, b in zip(io[len(s22) :], s22_energy):
        assert a.calc is not None
        assert b.calc is not None
        assert a.calc.results == b.calc.results


def test_add_new_keys_info(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    water = ase.build.molecule("H2O")

    io.append(water)
    water.info["key1"] = np.random.random()
    io.append(water)

    assert len(io) == 2
    assert "key1" not in io[0].info
    assert "key1" in io[1].info

    assert io[1].info["key1"] == water.info["key1"]


def test_add_new_keys_arrays(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    water = ase.build.molecule("H2O")

    io.append(water)
    water.arrays["key1"] = np.random.rand(len(water), 3)
    io.append(water)

    assert len(io) == 2
    assert "key1" not in io[0].arrays
    assert "key1" in io[1].arrays

    assert np.array_equal(io[1].arrays["key1"], water.arrays["key1"])

