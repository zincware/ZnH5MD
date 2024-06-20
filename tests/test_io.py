import ase.collections
import numpy as np

import znh5md


def test_IO_extend(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    images = list(ase.collections.s22)
    io.extend(images)

    structures = io[:]
    assert len(structures) == len(images)
    for a, b in zip(images, structures):
        assert np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        assert np.allclose(a.get_positions(), b.get_positions())


def test_IO_len(tmp_path, s22_info_arrays_calc):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22_info_arrays_calc)
    assert len(io) == 22


def test_IO_append(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    images = list(ase.collections.s22)
    io.extend(images)
    io.append(images[0])

    assert len(io) == len(images) + 1
    for a, b in zip(images + [images[0]], io[:]):
        assert np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        assert np.allclose(a.get_positions(), b.get_positions())
