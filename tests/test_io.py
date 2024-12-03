import ase.build
import ase.collections
import ase.io
import numpy as np
import pytest
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md
from znh5md.serialization import Frames


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
    # When ignoring the `use_calc` the info/arrays split
    # is not dependent on the type so we use dummy data for each
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
    # When ignoring the `use_calc` the info/arrays split
    # is not dependent on the type so we use dummy data for each
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
    # When ignoring the `use_calc` the info/arrays split
    # is not dependent on the type so we use dummy data for each
    water.info[info_key] = np.random.rand()
    water.info[f"{info_key}_arr"] = np.random.rand(
        1,
    )
    water.arrays[arrays_key] = np.random.rand(len(water), 3)
    assert info_key in water.info
    assert arrays_key in water.arrays

    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=False)
    if info_key == arrays_key:
        with pytest.raises(ValueError):
            io.append(water)

        return
    else:
        io.append(water)

    assert io[0].calc is None
    assert info_key in io[0].info
    assert arrays_key in io[0].arrays
    assert np.allclose(io[0].info[info_key], water.info[info_key])
    assert np.allclose(io[0].arrays[arrays_key], water.arrays[arrays_key])
    assert isinstance(io[0].info[info_key], float)
    assert isinstance(io[0].arrays[arrays_key], np.ndarray)
    assert isinstance(io[0].info[f"{info_key}_arr"], np.ndarray)


@pytest.mark.parametrize("key", all_properties)
def test_ase_info_key_value_error_info(tmp_path, key):
    water = ase.build.molecule("H2O")
    water.info[key] = np.random.rand()
    water.calc = SinglePointCalculator(water, **{key: np.random.rand()})
    with pytest.raises(ValueError):
        Frames.from_ase([water]).check()

    assert key in water.info

    io = znh5md.IO(tmp_path / "test.h5")
    with pytest.raises(ValueError):
        io.append(water)


@pytest.mark.parametrize("key", all_properties)
def test_ase_info_key_value_error_arrays(tmp_path, key):
    water = ase.build.molecule("H2O")
    water.arrays[key] = np.random.rand(len(water), 3)
    water.calc = SinglePointCalculator(water, **{key: np.random.rand()})
    with pytest.raises(ValueError):
        Frames.from_ase([water]).check()

    assert key in water.arrays

    io = znh5md.IO(tmp_path / "test.h5")
    with pytest.raises(ValueError):
        io.append(water)


@pytest.mark.parametrize("use_ase_calc", [True, False])
def test_convert_extxzy(tmp_path, s22_energy_forces, use_ase_calc):
    extxyz_fle = tmp_path / "test.extxyz"
    ase.io.write(extxyz_fle, s22_energy_forces)

    atoms = list(ase.io.iread(extxyz_fle))
    assert len(atoms) == len(s22_energy_forces)

    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=True)
    io.extend(atoms)

    # read with and without calc
    io = znh5md.IO(tmp_path / "test.h5", use_ase_calc=use_ase_calc)

    if use_ase_calc:
        assert "energy" in io[0].calc.results
        assert "forces" in io[0].calc.results
    else:
        assert "energy" in io[0].info
        assert "forces" in io[0].arrays


@pytest.mark.parametrize("expect_float", [True, False])
@pytest.mark.parametrize("key", all_properties)
def test_use_ase_calc_write_info_arrays(tmp_path, key, expect_float):
    water = ase.build.molecule("H2O")
    # When ignoring the `use_calc` the info/arrays split
    # is not dependent on the type so we use dummy data for each
    scalar_property = key in ["energy", "magmom", "free_energy"]
    if scalar_property:
        val = (
            np.random.rand()
            if expect_float
            else np.random.rand(
                1,
            )
        )
        calc = SinglePointCalculator(water, **{key: val})
    else:
        val = np.random.rand(len(water), 3)
        calc = SinglePointCalculator(water, **{key: val})
    water.calc = calc

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(water)
    if scalar_property:
        assert np.allclose(val, calc.results[key])
        assert (
            isinstance(calc.results[key], float)
            if expect_float
            else isinstance(calc.results[key], np.ndarray)
        )
    else:
        assert np.allclose(val, calc.results[key])
        assert isinstance(calc.results[key], np.ndarray)


def test_index_error(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    with pytest.raises(FileNotFoundError):
        # file does not exist yet
        io[0]

    io.append(ase.build.molecule("H2O"))
    assert io[0] is not None
    assert io[-1] is not None
    assert io[0:1] is not None
    assert io[0:2] is not None
    with pytest.raises(IndexError):
        io[1]
    with pytest.raises(IndexError):
        io[-2]


@pytest.mark.skip("ASE 3.23 does not support this anymore")
def test_np_int_getitem(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    io.append(ase.build.molecule("H2O"))
    assert io[np.int64(0)] is not None
    assert io[np.int64(-1)] is not None
    with pytest.raises(IndexError):
        io[np.int64(1)]
    with pytest.raises(IndexError):
        io[np.int64(-2)]
