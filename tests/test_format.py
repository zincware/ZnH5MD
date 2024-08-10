from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import pytest

from znh5md.format import ASEData, combine_asedata, extract_atoms_data
from ase.calculators.calculator import all_properties


def test_extract_atoms_data():
    water = molecule("H2O")
    data = extract_atoms_data(water)
    assert isinstance(data, ASEData)
    # TODO assert len(data) == 1 does not work, because wrong dimension

    data = combine_asedata([data, data])
    assert isinstance(data, ASEData)
    assert len(data) == 2

@pytest.mark.parametrize("key", all_properties)
def test_extract_atoms_data_no_calc_arrays(key):
    water = molecule("H2O")
    water.arrays[key] = np.random.rand(len(water), 3)

    # TODO raise a value error if the keys are in info or arrays
    data = extract_atoms_data(water, use_ase_calc=False)
    assert key in data.particles
    assert key not in data.observables
    assert data.metadata.get(key, {}).get("calc", False) is False

@pytest.mark.parametrize("key", all_properties)
def test_extract_atoms_data_no_calc_info(key):
    water = molecule("H2O")
    water.info[key] = np.random.rand(len(water), 3)

    data = extract_atoms_data(water, use_ase_calc=False)
    assert key in data.observables
    assert key not in data.particles
    assert data.metadata.get(key, {}).get("calc", False) is False

@pytest.mark.parametrize("key", all_properties)
def test_extract_atoms_data_calc_arrays(key):
    water = molecule("H2O")
    water.arrays[key] = np.random.rand(len(water), 3)

    with pytest.raises(ValueError):
        extract_atoms_data(water, use_ase_calc=True)

@pytest.mark.parametrize("key", all_properties)
def test_extract_atoms_data_calc_info(key):
    water = molecule("H2O")
    water.info[key] = np.random.rand(len(water), 3)

    with pytest.raises(ValueError):
        extract_atoms_data(water, use_ase_calc=True)

@pytest.mark.parametrize("key", all_properties)
def test_extract_atoms_calc_results(key):
    water = molecule("H2O")
    water.calc = SinglePointCalculator(water)
    water.calc.results[key] = np.random.rand(len(water), 3)

    data = extract_atoms_data(water, use_ase_calc=True)
    if key == "forces":
        key = "force"
    assert data.metadata[key]["calc"] is True