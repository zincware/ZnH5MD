from ase.atoms import Atoms
from ase.collections import s22
from ase.io import read, write

import znh5md


def test_write_one(tmp_path):
    path = tmp_path / "test.h5"

    atoms = list(s22)[0]
    write(path, atoms)

    io = znh5md.IO(path)
    images_read = io[:]

    assert len(images_read) == 1
    assert atoms == images_read[0]


def test_write_multiple(tmp_path):
    path = tmp_path / "test.h5"

    images = list(s22)
    write(path, images)

    io = znh5md.IO(path)
    images_read = io[:]

    assert len(images) == len(images_read)
    for image, image_read in zip(images, images_read):
        assert image == image_read


def test_append(tmp_path):
    path = tmp_path / "test.h5"

    images = list(s22)
    for image in images:
        write(path, image, append=True)

    io = znh5md.IO(path)
    images_read = io[:]

    assert len(images) == len(images_read)
    for image, image_read in zip(images, images_read):
        assert image == image_read


def test_read_one(tmp_path):
    path = tmp_path / "test.h5"
    io = znh5md.IO(path)

    images = list(s22)
    io.extend(images)

    image_read = read(path, index=0)
    assert isinstance(image_read, Atoms)
    assert images[0] == image_read


def test_read_multiple(tmp_path):
    path = tmp_path / "test.h5"
    io = znh5md.IO(path)

    images = list(s22)
    io.extend(images)

    images_read = read(path, index=":")
    assert isinstance(images_read, list)
    for image, image_read in zip(images, images_read):
        assert image == image_read
