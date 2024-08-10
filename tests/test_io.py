import ase.collections
import numpy as np
import pytest
import ase.build

import znh5md
from ase.calculators.calculator import all_properties



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

def test_not_use_ase_calc_read(tmp_path, s22_all_properties):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22_all_properties)

    new_io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=False)
    
    atoms = new_io[0]
    assert atoms.calc is None
    for key, val in s22_all_properties[0].info.items():
        assert atoms.info[key] == val
    
    for key, val in s22_all_properties[0].arrays.items():
        assert np.allclose(atoms.arrays[key], val)

    for key, val in s22_all_properties[0].calc.results.items():
        if key in atoms.arrays:
            assert np.allclose(atoms.arrays[key], val)
        else:
            assert np.allclose(atoms.info[key], val)

@pytest.mark.parametrize("key", all_properties + ["dummy"])
def test_not_use_ase_calc_write_arrays(tmp_path, key):
    water = ase.build.molecule("H2O")
    # When ignoreing the `use_calc` the info/arrays split
    # is not depenend on the type so we use dummy data for each
    water.arrays[key] = np.random.rand(len(water), 3)
    assert key in water.arrays

    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=False)
    io.append(water)

    assert io[0].calc is None
    assert key in io[0].arrays
    assert np.allclose(io[0].arrays[key], water.arrays[key])

@pytest.mark.parametrize("key", all_properties + ["dummy"])
def test_not_use_ase_calc_write_info(tmp_path, key):
    water = ase.build.molecule("H2O")
    # When ignoreing the `use_calc` the info/arrays split
    # is not depenend on the type so we use dummy data for each
    water.info[key] = np.random.rand()
    assert key in water.info

    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=False)
    io.append(water)

    assert io[0].calc is None
    assert key in io[0].info
    assert np.allclose(io[0].info[key], water.info[key])


@pytest.mark.parametrize("info_key", all_properties + ["dummy"])
@pytest.mark.parametrize("arrays_key", all_properties + ["dummy"])
def test_not_use_ase_calc_write_info_arrays(tmp_path, info_key, arrays_key):
    water = ase.build.molecule("H2O")
    # When ignoreing the `use_calc` the info/arrays split
    # is not depenend on the type so we use dummy data for each
    water.info[info_key] = np.random.rand()
    water.arrays[arrays_key] = np.random.rand(len(water), 3)
    assert info_key in water.info
    assert arrays_key in water.arrays

    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=False)
    io.append(water)

    assert io[0].calc is None
    assert info_key in io[0].info
    assert arrays_key in io[0].arrays
    assert np.allclose(io[0].info[info_key], water.info[info_key])
    assert np.allclose(io[0].arrays[arrays_key], water.arrays[arrays_key])


@pytest.mark.parametrize("key", all_properties)
def test_ase_info_key_value_error_info(tmp_path, key):
    water = ase.build.molecule("H2O")
    water.info[key] = np.random.rand()
    assert key in water.info

    io = znh5md.IO(tmp_path / "test.h5")
    with pytest.raises(ValueError):
        io.append(water)

@pytest.mark.parametrize("key", all_properties)
def test_ase_info_key_value_error_arrays(tmp_path, key):
    water = ase.build.molecule("H2O")
    water.arrays[key] = np.random.rand(len(water), 3)
    assert key in water.arrays

    io = znh5md.IO(tmp_path / "test.h5")
    with pytest.raises(ValueError):
        io.append(water)