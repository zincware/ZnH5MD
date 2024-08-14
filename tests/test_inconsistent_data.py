import numpy.testing as npt

import znh5md


def test_keys_missing(tmp_path, s22, s22_energy_forces):
    io = znh5md.IO(tmp_path / "test.h5")

    images = s22_energy_forces + s22
    io.extend(images)
    assert len(io) == len(images)
    assert len(list(io)) == len(images)

    for a, b in zip(images, io):
        assert a == b
        if b.calc is not None:
            for key in b.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])
        else:
            assert a.calc is None
