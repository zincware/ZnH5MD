import numpy as np
import numpy.testing as npt
import pytest

from znh5md import serialization as zns


def test_serialisation_s22(s22):
    frames = zns.Frames.from_ase(s22)

    assert len(frames.positions) == 22
    assert len(frames.numbers) == 22
    assert len(frames.pbc) == 22
    assert len(frames.cell) == 22
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
@pytest.mark.parametrize("append", [True, False])
def test_frames_iter(dataset_name, append, request):
    dataset = request.getfixturevalue(dataset_name)
    frames = zns.Frames()
    if append:
        for frame in dataset:
            frames.append(frame)
    else:
        frames.extend(dataset)

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
        "s22_info_arrays_calc_missing_inbetween",
    ],
)
def test_frames_getitem(dataset_name, request):
    dataset = request.getfixturevalue(dataset_name)
    frames = zns.Frames.from_ase(dataset)

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
    frames = zns.Frames.from_ase(s22)

    assert len(frames) == 22

    assert len(list(frames)) == 22


def test_serialisation_s22_energy(s22_energy):
    frames = zns.Frames.from_ase(s22_energy)

    assert len(frames.arrays) == 0
    assert len(frames.info) == 0
    assert len(frames.calc) == 1
    assert len(frames.calc["energy"]) == 22
    assert all(isinstance(x, float) for x in frames.calc["energy"])


def test_serialisation_s22_energy_forces(s22_energy_forces):
    frames = zns.Frames.from_ase(s22_energy_forces)

    assert len(frames.arrays) == 0
    assert len(frames.info) == 0
    assert len(frames.calc) == 2
    assert len(frames.calc["energy"]) == 22
    assert all(isinstance(x, float) for x in frames.calc["energy"])
    assert len(frames.calc["forces"]) == 22
    assert all(isinstance(x, np.ndarray) for x in frames.calc["forces"])
