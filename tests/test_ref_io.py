import h5py
import numpy as np
import numpy.testing as npt
import pytest
import rdkit2ase

from znh5md.interface import IO


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
@pytest.mark.parametrize("append", [False])  # TODO: True raises error!!
# @pytest.mark.parametrize("store_ase_origin ", [False, True])
def test_frames_iter(dataset_name, append, request, tmp_path):
    dataset = request.getfixturevalue(dataset_name)

    io = IO(tmp_path / "test.h5")
    # io._store_ase_origin = store_ase_origin
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


def test_arrays_list_array(tmp_path):
    atoms = rdkit2ase.smiles2atoms("Cl")
    atoms.arrays["lst"] = list(atoms.symbols)
    atoms.arrays["arr"] = np.array(atoms.symbols)

    io = IO(tmp_path / "test.h5")
    io.extend([atoms, rdkit2ase.smiles2atoms("C")])

    with h5py.File(io.filename, "r") as f:
        assert np.array_equal(f["particles/atoms/lst/value"][0], b'["Cl", "H"]')
        assert np.array_equal(f["particles/atoms/lst/value"][1], b"")
        # now for arr
        assert np.array_equal(f["particles/atoms/arr/value"][0], b'["Cl", "H"]')
        assert np.array_equal(f["particles/atoms/arr/value"][1], b"")
