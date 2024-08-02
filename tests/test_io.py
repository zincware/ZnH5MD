import ase.collections
import ase.build
import numpy as np
import pytest

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


def test_author_creater(tmp_path):
    io = znh5md.IO(
        tmp_path / "test.h5",
        author="Fabian",
        author_email="email@uni-stuttgart.de",
        creator="ZnH5MD",
        creator_version="V0.3",
    )
    io.extend(list(ase.collections.s22))

    io2 = znh5md.IO(tmp_path / "test.h5")

    assert io2.author == "Fabian"
    assert io2.author_email == "email@uni-stuttgart.de"
    assert io2.creator == "ZnH5MD"
    assert io2.creator_version == "V0.3"


def test_extend_empty(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(list(ase.collections.s22))

    assert len(io) == 22
    with pytest.warns(UserWarning, match="No data provided"):
        io.extend([])
    assert len(io) == 22

def test_extend_single(tmp_path):
    vectors = np.random.rand(3, 3, 2, 3)

    water = ase.build.molecule("H2O")
    water.info["vectors"] = vectors

    znh5md.write(tmp_path / "test.h5", water)

    io = znh5md.IO(tmp_path / "test.h5")
    assert len(io) == 1
    assert np.allclose(io[0].info["vectors"], vectors)
    assert io[0].info["vectors"].shape == vectors.shape
