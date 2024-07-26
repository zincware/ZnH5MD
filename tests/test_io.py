from unittest.mock import patch

import ase.collections
import h5py
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


def test_experimental_fancy_loading(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5", experimental_fancy_loading=True)
    images = list(ase.collections.s22)
    io.extend(images)
    # ensure chunk size is smaller then the number of images
    with patch("znh5md.io.IO._read_chunk_size", return_value=5):
        indices = [1, 3, 5, 7, 21]
        assert len(io[indices]) == len([images[i] for i in indices])
        assert len(io[indices]) == 5

        for a, b in zip([images[i] for i in indices], io[indices]):
            assert np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
            assert np.allclose(a.get_positions(), b.get_positions())


def test_experimental_fancy_loading_file_handle(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(list(ase.collections.s22))

    with pytest.raises(ValueError):
        with h5py.File(tmp_path / "test.h5", "r") as f:
            znh5md.IO(file_handle=f, experimental_fancy_loading=True)
