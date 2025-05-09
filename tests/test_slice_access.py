import numpy as np
import pytest

import znh5md


@pytest.mark.parametrize(
    "dataset_name",
    [
        "s22",
        "s22_energy",
        "s22_energy_forces",
        "s22_all_properties",
        "s22_info_arrays_calc",
        "s22_mixed_pbc_cell",
        "s22_illegal_calc_results",
        "s22_no_ascii",
        "frames_with_residuenames",
        "s22_info_arrays_calc_missing_inbetween",
    ],
)
@pytest.mark.parametrize("append", [False])
def test_frames_iter(dataset_name, append, request, tmp_path):
    dataset = request.getfixturevalue(dataset_name)

    io = znh5md.IO(tmp_path / "test.h5")
    if append:
        for frame in dataset:
            io.append(frame)
    else:
        io.extend(dataset)

    for atoms in io:
        for value in atoms.arrays.values():
            assert isinstance(value, np.ndarray), f"Exp. np.ndarray, got {type(value)}"

    for atoms in io[:]:
        for value in atoms.arrays.values():
            assert isinstance(value, np.ndarray), f"Exp. np.ndarray, got {type(value)}"
