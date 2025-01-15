import numpy as np
import numpy.testing as npt
import rdkit2ase

import znh5md


def test_pdb_repeat(tmp_path):
    water = rdkit2ase.smiles2conformers(smiles="O", numConfs=1)[0]
    water.arrays["atomtypes"] = np.array(["H", "O", "H"])
    assert isinstance(water.arrays["atomtypes"], np.ndarray)

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(water)

    atoms = io[0]
    assert isinstance(atoms.arrays["atomtypes"], np.ndarray)
    npt.assert_array_equal(atoms.arrays["atomtypes"], water.arrays["atomtypes"])
