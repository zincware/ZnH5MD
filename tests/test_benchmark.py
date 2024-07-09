import uuid

import ase
import numpy as np
import pytest

import znh5md


@pytest.fixture
def atoms():
    def _create_atoms(count: int, size: int) -> list[ase.Atoms]:
        """Create random atoms.

        Attributes
        ----------
        count : int
            Number of images.
        size : int
            Number of atoms in each image.
        """
        images = []
        for _ in range(count):
            images.append(ase.Atoms("H" * size, positions=np.random.rand(size, 3)))
        return images

    return _create_atoms


@pytest.mark.benchmark(group="write-count")
@pytest.mark.parametrize("count", [100 * x for x in range(1, 11)])
def test_bm_write_count(tmp_path, benchmark, atoms, count, size=100):
    images = atoms(count, size)

    def _write():
        filename = tmp_path / f"{uuid.uuid4()}.h5md"
        znh5md.write(filename, images, compression=None)

    benchmark(_write)


@pytest.mark.benchmark(group="write-size")
@pytest.mark.parametrize("size", [100 * x for x in range(1, 11)])
def test_bm_write_size(tmp_path, benchmark, atoms, size, count=100):
    images = atoms(count, size)

    def _write():
        filename = tmp_path / f"{uuid.uuid4()}.h5md"
        znh5md.write(filename, images, compression=None)

    benchmark(_write)


@pytest.mark.benchmark(group="read-count")
@pytest.mark.parametrize("count", [100 * x for x in range(1, 11)])
def test_bm_read_count(tmp_path, benchmark, atoms, count, size=100):
    images = atoms(count, size)
    filename = tmp_path / f"{uuid.uuid4()}.h5md"
    znh5md.write(filename, images, compression="gzip")

    def _read():
        io = znh5md.IO(filename)
        _ = io[:]

    benchmark(_read)


@pytest.mark.benchmark(group="read-size")
@pytest.mark.parametrize("size", [100 * x for x in range(1, 11)])
def test_bm_read_size(tmp_path, benchmark, atoms, size, count=100):
    images = atoms(count, size)
    filename = tmp_path / f"{uuid.uuid4()}.h5md"
    znh5md.write(filename, images, compression="gzip")

    def _read():
        io = znh5md.IO(filename)
        _ = io[:]

    benchmark(_read)
