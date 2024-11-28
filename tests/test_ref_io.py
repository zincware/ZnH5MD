import pytest
from znh5md.interface import IO
import numpy.testing as npt



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
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_frames_iter(dataset_name, append, request, tmp_path):
    dataset = request.getfixturevalue(dataset_name)

    io = IO(tmp_path / "test.h5")
    if append:
        for frame in dataset:
            io.append(frame)
    else:
        io.extend(dataset)
    
    assert len(io) == len(dataset)
    for a, b in zip(io, dataset):
        assert a == b
        for key in set(a.arrays.keys()) | set(b.arrays.keys()):
            npt.assert_array_equal(a.arrays[key], b.arrays[key])
        for key in set(a.info.keys()) | set(b.info.keys()):
            npt.assert_array_equal(a.info[key], b.info[key])
        if b.calc is not None or a.calc is not None:
            for key in set(a.calc.results.keys()) | set(b.calc.results.keys()):
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])
        
