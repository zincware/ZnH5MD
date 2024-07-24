import numpy as np
import numpy.testing as npt
from ase.build import molecule

import znh5md


def test_high_dim_data_arrays(tmp_path):
    water = molecule("H2O")
    water.arrays["descriptor"] = np.random.rand(3, 10, 5, 4, 3)

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(water)

    io2 = znh5md.IO(tmp_path / "test.h5")
    water2 = io2[0]

    npt.assert_array_equal(water.arrays["descriptor"], water2.arrays["descriptor"])


def test_high_dim_data_info(tmp_path):
    water = molecule("H2O")
    water.info["descriptor"] = np.random.rand(5, 10, 4, 4, 3)

    io = znh5md.IO(tmp_path / "test.h5")
    io.append(water)

    io2 = znh5md.IO(tmp_path / "test.h5")
    water2 = io2[0]

    npt.assert_array_equal(water.info["descriptor"], water2.info["descriptor"])
