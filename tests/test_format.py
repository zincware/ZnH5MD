from ase.build import molecule

from znh5md.format import ASEData, combine_asedata, extract_atoms_data


def test_extract_atoms_data():
    water = molecule("H2O")
    data = extract_atoms_data(water)
    assert isinstance(data, ASEData)
    # TODO assert len(data) == 1 does not work, because wrong dimension

    data = combine_asedata([data, data])
    assert isinstance(data, ASEData)
    assert len(data) == 2
