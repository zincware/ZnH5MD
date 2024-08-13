import ase.build
import pytest

import znh5md


def test_extend_wrong_error(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")

    with pytest.raises(ValueError, match="images must be a list of ASE Atoms objects"):
        io.extend(molecule)


def test_append_wrong_error(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    molecule = ase.build.molecule("H2O")

    with pytest.raises(ValueError, match="atoms must be an ASE Atoms object"):
        io.append([molecule])
