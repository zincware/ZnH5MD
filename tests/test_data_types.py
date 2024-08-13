import ase.build

import znh5md


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
