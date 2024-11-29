import znh5md
import rdkit2ase
import h5py


def test_extend_same_size(tmp_path):
    water = rdkit2ase.smiles2atoms("O")
    io = znh5md.IO(tmp_path / "test.h5")

    io.append(water)
    assert len(io) == 1

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (1, 3, 3)
    
    io.append(water)
    assert len(io) == 2

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (2, 3, 3)


def test_extend_different_size(tmp_path):
    water = rdkit2ase.smiles2atoms("O")
    ammonia = rdkit2ase.smiles2atoms("N")

    io = znh5md.IO(tmp_path / "test.h5")

    io.append(water)
    assert len(io) == 1

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (1, 3, 3)

    io.append(ammonia)
    assert len(io) == 2

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (2, 4, 3)

def test_extend_misssing(tmp_path):
    water = rdkit2ase.smiles2atoms("O")
    ammonia = rdkit2ase.smiles2atoms("N")

    ammonia.info["val"] = 1

    io = znh5md.IO(tmp_path / "test.h5")

    io.append(water)
    assert len(io) == 1

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (1, 3, 3)

    io.append(ammonia)
    assert len(io) == 2

    with h5py.File(io.filename, "r") as f:
        assert f["particles/atoms/position/value"].shape == (2, 4, 3)
        assert f["observables/atoms/val/value"].shape == (2,)