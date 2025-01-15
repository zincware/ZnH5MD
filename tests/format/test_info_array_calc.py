import numpy as np
import numpy.testing as npt
import pytest
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

import znh5md


# Define assertion functions for different data types
def assert_equal(actual, expected):
    npt.assert_equal(actual, expected)


def assert_allclose(actual, expected):
    npt.assert_allclose(actual, expected)


def assert_allclose_list(actual, expected):
    for a, e in zip(actual, expected):
        npt.assert_allclose(a, e)


def assert_dict_allclose(actual, expected):
    for key in expected:
        npt.assert_allclose(actual[key], expected[key])


# Pytest fixtures
@pytest.fixture
def io_fixture(tmp_path):
    """Fixture to initialize the ZnH5MD IO object."""
    return znh5md.IO(tmp_path / "test.h5")


@pytest.fixture
def water_molecule():
    """Fixture to provide a simple water molecule."""
    return molecule("H2O")


@pytest.mark.parametrize(
    "key,value,assert_fn",
    [
        ("str", "Hello World", assert_equal),
        ("float", 3.14, assert_equal),
        ("ndarray", np.random.rand(10), assert_allclose),
        ("list_float", [1.0, 2.0, 3.0], assert_allclose),
        ("list_str", ["Hello", "World"], assert_equal),
        ("dict", {"a": 1, "b": 2}, assert_equal),
        ("dict_str", {"a": "Hello", "b": "World"}, assert_equal),
        ("list_array", [np.random.rand(10), np.random.rand(10)], assert_allclose_list),
        ("dict_dict", {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}, assert_equal),
        # (
        #     "dict_array",
        #     {"a": np.random.rand(10), "b": np.random.rand(10)},
        #     assert_dict_allclose,
        # ),
        # ("list_dict", [{"a": 1, "b": 2}, {"c": 3, "d": 4}], assert_equal),
    ],
)
def test_info(io_fixture, water_molecule, key, value, assert_fn):
    """Generic test for different info types."""
    # Assign the value to the molecule's info
    water_molecule.info[key] = value

    # Append to the ZnH5MD IO object
    io_fixture.append(water_molecule)

    # Retrieve and test
    assert_fn(io_fixture[0].info[key], value)
    assert key not in io_fixture[0].arrays
    assert io_fixture[0].calc is None


@pytest.mark.parametrize(
    "key,value,assert_fn",
    [
        ("str", "Hello World", assert_equal),
        ("float", 3.14, assert_equal),
        ("ndarray", np.random.rand(10), assert_allclose),
        ("list_float", [1.0, 2.0, 3.0], assert_allclose),
        ("list_str", ["Hello", "World"], assert_equal),
        ("dict", {"a": 1, "b": 2}, assert_equal),
        ("dict_str", {"a": "Hello", "b": "World"}, assert_equal),
        ("list_array", [np.random.rand(10), np.random.rand(10)], assert_allclose_list),
        ("dict_dict", {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}, assert_equal),
        # (
        #     "dict_array",
        #     {"a": np.random.rand(10), "b": np.random.rand(10)},
        #     assert_dict_allclose,
        # ),
        # ("list_dict", [{"a": 1, "b": 2}, {"c": 3, "d": 4}], assert_equal),
    ],
)
def test_calc(io_fixture, water_molecule, key, value, assert_fn):
    """Generic test for different calc types."""
    # Assign the value to the molecule's info
    water_molecule.calc = SinglePointCalculator(water_molecule)
    water_molecule.calc.results[key] = value

    # Append to the ZnH5MD IO object
    io_fixture.append(water_molecule)

    # Retrieve and test
    assert_fn(io_fixture[0].calc.results[key], value)
    assert key not in io_fixture[0].arrays
    assert io_fixture[0].info == {}


@pytest.mark.parametrize(
    "key,value,assert_fn",
    [
        ("str", "Hello World", assert_equal),
        ("float", 3.14, assert_equal),
        ("ndarray", np.random.rand(10), assert_allclose),
        ("list_float", [1.0, 2.0, 3.0], assert_allclose),
        ("list_str", ["Hello", "World"], assert_equal),
        ("dict", {"a": 1, "b": 2}, assert_equal),
        ("dict_str", {"a": "Hello", "b": "World"}, assert_equal),
        ("list_array", [np.random.rand(10), np.random.rand(10)], assert_allclose_list),
        ("dict_dict", {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}, assert_equal),
        # (
        #     "dict_array",
        #     {"a": np.random.rand(10), "b": np.random.rand(10)},
        #     assert_dict_allclose,
        # ),
        # ("list_dict", [{"a": 1, "b": 2}, {"c": 3, "d": 4}], assert_equal),
    ],
)
def test_arrays(io_fixture, water_molecule, key, value, assert_fn):
    """Generic test for different calc types."""
    # Assign the value to the molecule's info
    water_molecule.arrays[key] = value

    # Append to the ZnH5MD IO object
    io_fixture.append(water_molecule)

    # Retrieve and test
    assert_fn(io_fixture[0].arrays[key], value)
    # assert key not in io_fixture[0].arrays
    assert io_fixture[0].info == {}
    assert io_fixture[0].calc is None
