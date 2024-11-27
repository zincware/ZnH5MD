from znh5md import serialization as zns
import numpy as np
import numpy.testing as npt
import pytest

def test_serialisation_s22(s22):
    frames = zns.encode(s22)

    assert frames.positions.shape == (22,)
    assert frames.numbers.shape == (22,)
    assert frames.pbc.shape == (22, 3)
    assert frames.cell.shape == (22, 3, 3)
    assert len(frames.arrays) == 0
    assert len(frames.info) == 0
    assert len(frames.calc) == 0

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
def test_frames_iter(dataset_name, request):
    dataset = request.getfixturevalue(dataset_name)
    frames = zns.encode(dataset)

    for a, b in zip(frames, dataset):
        npt.assert_array_equal(a.positions, b.positions)
        npt.assert_array_equal(a.cell, b.cell)
        npt.assert_array_equal(a.pbc, b.pbc)
        npt.assert_array_equal(a.numbers, b.numbers)

        for key in b.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])
        for key in b.info:
            npt.assert_array_equal(a.info[key], b.info[key])
        if b.calc is not None:
            for key in b.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])


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
def test_frames_getitem(dataset_name, request):
    dataset = request.getfixturevalue(dataset_name)
    frames = zns.encode(dataset)

    for idx, frame in enumerate(dataset):
        a = frames[idx]
        b = frame

        npt.assert_array_equal(a.positions, b.positions)
        npt.assert_array_equal(a.cell, b.cell)
        npt.assert_array_equal(a.pbc, b.pbc)
        npt.assert_array_equal(a.numbers, b.numbers)

        for key in b.arrays:
            npt.assert_array_equal(a.arrays[key], b.arrays[key])
        for key in b.info:
            npt.assert_array_equal(a.info[key], b.info[key])
        if b.calc is not None:
            for key in b.calc.results:
                npt.assert_array_equal(a.calc.results[key], b.calc.results[key])


def test_len(s22):
    frames = zns.encode(s22)

    assert len(frames) == 22

def test_serialisation_s22_energy(s22_energy):
    frames = zns.encode(s22_energy)

    assert len(frames.arrays) == 0
    assert len(frames.info) == 0
    assert len(frames.calc) == 1
    assert frames.calc["energy"].shape == (22,)
    assert all(isinstance(x, float) for x in frames.calc["energy"])


def test_serialisation_s22_energy_forces(s22_energy_forces):
    frames = zns.encode(s22_energy_forces)

    assert len(frames.arrays) == 0
    assert len(frames.info) == 0
    assert len(frames.calc) == 2
    assert frames.calc["energy"].shape == (22,)
    assert all(isinstance(x, float) for x in frames.calc["energy"])
    assert frames.calc["forces"].shape == (22,)
    assert all(isinstance(x, np.ndarray) for x in frames.calc["forces"])