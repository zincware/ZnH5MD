import numpy.testing as npt
import pytest

import znh5md


def test_read_write(tmp_path, s22_all_properties):
    file = tmp_path / "test.h5"
    znh5md.write(file, s22_all_properties)
    atoms = znh5md.read(file)
    assert atoms == s22_all_properties[-1]
    npt.assert_array_equal(
        atoms.get_positions(), s22_all_properties[-1].get_positions()
    )

    assert len(znh5md.read(file, index=slice(None, None, None))) == len(
        s22_all_properties
    )
    assert len(znh5md.read(file, index=[0, 1, 2])) == 3


def test_iread(tmp_path, s22_all_properties):
    file = tmp_path / "test.h5"
    znh5md.write(file, s22_all_properties)

    for a, b in zip(s22_all_properties, znh5md.iread(file)):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())


def test_append_false(tmp_path, s22_all_properties):
    file = tmp_path / "test.h5"
    znh5md.write(file, s22_all_properties)
    with pytest.raises(FileExistsError):
        znh5md.write(file, s22_all_properties, append=False)
