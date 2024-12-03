import ase.build
import ase.io
import numpy.testing as npt

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


def test_very_long_text_data(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")

    molecule.info["test"] = f"{list(range(1_000))}"
    io.append(molecule)
    assert io[0].info["test"] == f"{list(range(1_000))}"


def test_int_info_data(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")
    molecule.info["test"] = 123

    io.append(molecule)
    assert io[0].info["test"] == 123


def test_dict_data(tmp_path):
    molecule = ase.build.molecule("H2O")
    molecule.info["test"] = {"a": 1, "b": 2}

    # Write to extxyz format
    extxyz_path = tmp_path / "molecule.extxyz"
    ase.io.write(extxyz_path, molecule, format="extxyz")

    # Read from extxyz format
    molecule = ase.io.read(extxyz_path, format="extxyz")

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(molecule)
    molecule.info["test"] = {"a": 1, "b": 2, "c": 3}
    io.append(molecule)
    molecule.info["b"] = {"a": 1, "b": 2, "c": 3, "d": 4}
    io.append(molecule)
    # molecule.info["elements"] = {"H": "H", 2: "O"}
    # NOTE! Mixed types in dict keys are currently not supported
    # TODO json dump / load?
    molecule.info["elements"] = {"H": "H", "O": "O"}
    io.append(molecule)

    assert io[0].info["test"] == {"a": 1, "b": 2}
    assert io[1].info["test"] == {"a": 1, "b": 2, "c": 3}
    assert io[2].info["b"] == {"a": 1, "b": 2, "c": 3, "d": 4}
    # assert io[3].info["elements"] == {"H": "H", 2: "O"}
    assert io[3].info["elements"] == {"H": "H", "O": "O"}


def test_list_data(tmp_path):
    molecule = ase.build.molecule("H2O")
    molecule.info["test"] = [1, 2]

    # Write to extxyz format
    extxyz_path = tmp_path / "molecule.extxyz"
    ase.io.write(extxyz_path, molecule, format="extxyz")

    # Read from extxyz format
    molecule = ase.io.read(extxyz_path, format="extxyz")
    npt.assert_array_equal(molecule.info["test"], [1, 2])

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(molecule)
    molecule.info["test"] = [1, 2, 3]
    io.append(molecule)
    molecule.info["b"] = [1, 2, 3, 4]
    io.append(molecule)
    molecule.info["elements"] = ["H", "O"]
    io.append(molecule)

    npt.assert_array_equal(io[0].info["test"], [1, 2])
    npt.assert_array_equal(io[1].info["test"], [1, 2, 3])
    npt.assert_array_equal(io[2].info["b"], [1, 2, 3, 4])
    npt.assert_array_equal(io[3].info["elements"], ["H", "O"])


def test_multiple_molecules_with_diff_length_dicts(tmp_path):
    molecules = [
        ase.build.molecule("H2O"),
        ase.build.molecule("CH4"),
    ]

    # Assign different length dicts to molecule info
    molecules[0].info["test"] = {"a": 1, "b": 2}
    molecules[1].info["test"] = {"a": 1}

    extxyz_path = tmp_path / "molecules.extxyz"
    ase.io.write(extxyz_path, molecules, format="extxyz")

    read_molecules = ase.io.read(extxyz_path, index=":", format="extxyz")

    io = znh5md.IO(tmp_path / "test.h5")
    for mol in read_molecules:
        io.append(mol)

    assert io[0].info["test"] == {"a": 1, "b": 2}
    assert io[1].info["test"] == {"a": 1}
