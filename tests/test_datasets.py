import numpy.testing as npt
import pytest

import znh5md


@pytest.mark.parametrize(
    "dataset",
    [
        "s22",
        "s22_energy",
        "s22_all_properties",
        "s22_info_arrays_calc",
        "s22_mixed_pbc_cell",
        "water",
    ],
)
def test_datasets(tmp_path, dataset, request):
    images = request.getfixturevalue(dataset)
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(images)
    images2 = io[:]

    assert len(images) == len(images2)

    for a, b in zip(images, images2):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_equal(a.get_cell(), b.get_cell())
        npt.assert_array_equal(a.get_pbc(), b.get_pbc())
        npt.assert_array_equal(a.get_velocities(), b.get_velocities())
        if a.calc is not None:
            assert set(a.calc.results) == set(b.calc.results)
            for key in a.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])

        for key in a.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])

        for key in a.info:
            npt.assert_array_equal(a.info[key], b.info[key])
