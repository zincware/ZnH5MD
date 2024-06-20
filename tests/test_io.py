import ase.collections
import numpy as np

import znh5md


def test_IO_read(cu_file):
    io = znh5md.IO(cu_file)

    assert isinstance(io[0], ase.Atoms)
    assert io[0].get_atomic_numbers().tolist() == 108 * [29]
    positions = io[0].get_positions()

    assert np.sum(positions**2) > 0

    momenta = io[0].get_momenta()
    assert np.sum(momenta**2) > 0

    assert io[0].pbc.all()

    assert "forces" in io[0].calc.results
    forces = io[0].get_forces()

    assert "energy" in io[0].calc.results
    energy = io[0].get_potential_energy()

    assert isinstance(io[1:2], list)
    for atoms in io[1:2]:
        assert isinstance(atoms, ase.Atoms)
        assert atoms.get_atomic_numbers().tolist() == 108 * [29]
        # check that positions change
        assert not np.array_equal(atoms.get_positions(), positions)
        positions = atoms.get_positions()

        assert not np.array_equal(atoms.get_momenta(), momenta)
        momenta = atoms.get_momenta()

        assert atoms.pbc.all()

        assert "forces" in atoms.calc.results
        assert not np.array_equal(atoms.get_forces(), forces)
        forces = atoms.get_forces()

        assert "energy" in atoms.calc.results
        assert not np.array_equal(atoms.get_potential_energy(), energy)
        energy = atoms.get_potential_energy()


def test_IO_extend(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    images = list(ase.collections.s22)
    io.extend(images)

    structures = io[:]
    assert len(structures) == len(images)
    for a, b in zip(images, structures):
        assert np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        assert np.allclose(a.get_positions(), b.get_positions())


def test_IO_len(cu_file):
    io = znh5md.IO(cu_file)
    assert len(io) == 10


def test_IO_append(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    images = list(ase.collections.s22)
    io.extend(images)
    io.append(images[0])

    assert len(io) == len(images) + 1
    for a, b in zip(images + [images[0]], io[:]):
        assert np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        assert np.allclose(a.get_positions(), b.get_positions())
