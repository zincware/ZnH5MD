from ase.build import molecule
import znh5md

def test_single_obs_key_info(tmp_path):
    # Test a special case where only the first config has the key
    # which caused an error in the past
    io = znh5md.IO(tmp_path / "test.h5")
    water = molecule("H2O")
    water.info["density"] = 0.997
    io.append(water)
    del water.info["density"]
    io.extend([water for _ in range(5)])
    assert len(io) == 6
    assert len(io[:]) == 6
    assert len(list(io)) == 6