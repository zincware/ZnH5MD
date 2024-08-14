import znh5md
import numpy.testing as npt


def test_keys_missing(tmp_path, s22, s22_energy_forces):
    io = znh5md.IO(tmp_path / "test.h5")


    images = s22_energy_forces + s22
    io.extend(images)
    assert len(io) == len(images)
    assert len(list(io)) == len(images)

    for a, b in zip(images, io):
        assert a == b
        b.get_potential_energy()
        assert a.get_potential_energy() == b.get_potential_energy()
        npt.assert_array_equal(a.get_forces(), b.get_forces())
