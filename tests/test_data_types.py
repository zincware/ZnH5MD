import ase.build

import znh5md
import pytest


def test_smiles(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")
    molecule.info["smiles"] = "O"

    io.append(molecule)
    assert io[0].info["smiles"] == "O"

    molecule = ase.build.molecule("H2O2")
    molecule.info["smiles"] = "OO"

    io.append(molecule)
    assert io[0].info["smiles"] == "O"
    assert io[1].info["smiles"] == "OO"

def test_very_long_text_data(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")

    molecule.info["test"] = f"{list(range(1_000))}"
    with pytest.raises(ValueError, match="String test is too long to be stored."):
        io.append(molecule)