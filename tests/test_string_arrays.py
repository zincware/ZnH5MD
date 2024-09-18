import numpy.testing as npt

import znh5md


def test_info_non_ascii(tmp_path, frames_with_residuenames):
    io = znh5md.IO(tmp_path / "test.h5")
    # io = znh5md.IO("test.h5")
    io.extend(frames_with_residuenames)

    for a, b in zip(io, frames_with_residuenames):
        for key in b.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])
            # test json per atom?
