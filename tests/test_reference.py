"""Test against pyh5md reference implementation."""

import ase
import numpy as np
import numpy.testing as npt
import pytest
from pyh5md import File, element

import znh5md


@pytest.fixture
def md() -> list[ase.Atoms]:
    """Run a simple MD simulation with ASE."""
    from ase import units
    from ase.calculators.emt import EMT
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.lattice.cubic import FaceCenteredCubic
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    size = 3

    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol="Cu",
        size=(size, size, size),
        pbc=True,
    )

    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = VelocityVerlet(atoms, 5 * units.fs)

    structures = []

    for _ in range(10):
        dyn.run(1)
        structures.append(atoms.copy())
        structures[-1].set_calculator(
            SinglePointCalculator(
                atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
            )
        )
    return structures


def ase_to_pyh5md(structures, path):
    with File(path, "w") as f:
        at = f.particles_group("atoms")
        at.create_box(
            dimension=3,
            boundary=["periodic"] * 3,
            store="time",
            shape=(3,),
            dtype=np.float64,
        )

        at_pos = element(
            at,
            "position",
            store="time",
            time=True,
            shape=(len(structures[0]), 3),
            dtype=np.float64,
        )

        at_species = element(
            at,
            "species",
            store="time",
            step_from=at_pos,
            time=True,
            shape=(len(structures[0]),),
            dtype=np.int32,
        )

        at_v = element(
            at,
            "momentum",
            store="time",
            shape=(len(structures[0]), 3),
            dtype=np.float64,
        )
        at_f = element(
            at, "forces", store="time", shape=(len(structures[0]), 3), dtype=np.float64
        )

        obs_at_e = element(
            f, "observables/atoms/energy", store="time", shape=(1,), dtype=np.float64
        )

        DT = 0.1

        for idx, atoms in enumerate(structures):
            at_pos.append(atoms.get_positions(), idx, idx * DT)
            at_species.append(atoms.get_atomic_numbers(), idx, idx * DT)
            at_v.append(atoms.get_momenta(), idx, idx * DT)
            obs_at_e.append(atoms.get_potential_energy(), idx * DT)
            at_f.append(atoms.get_forces(), idx, idx * DT)
            at.box.edges.append(atoms.get_cell().diagonal(), idx, idx * DT)


def test_open(example_h5):
    with File(example_h5, "r") as f:
        print(f)


def test_pyh5md_ASEH5MD(md, tmp_path):
    """Read pyh5md file with ASEH5MD."""
    path = tmp_path / "db.h5"
    ase_to_pyh5md(md, path)
    znh5md.ASEH5MD(path)
    structures = znh5md.ASEH5MD(path).get_atoms_list()
    assert len(structures) == len(md)
    # check that the data is not the same
    assert md[0].get_positions()[0][0] != md[1].get_positions()[0][0]

    for a, b in zip(md, structures):
        npt.assert_array_equal(a.get_positions(), b.get_positions())
        npt.assert_array_equal(a.get_momenta(), b.get_momenta())
        npt.assert_array_equal(a.get_forces(), b.get_forces())
        npt.assert_array_equal(a.get_potential_energy(), b.get_potential_energy())
        npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
        npt.assert_array_equal(a.get_cell(), b.get_cell())


def test_DataWriter_pyh5md(md, tmp_path):
    """Test reading DataWriter with pyh5md."""
    path = tmp_path / "db.h5"
    db = znh5md.io.DataWriter(path)
    db.add(znh5md.io.AtomsReader(md))

    with File(path, "r") as f:
        g = f.particles_group("atoms")
        position = element(g, "position").value[:]
        species = element(g, "species").value[:]
        momentum = element(g, "momentum").value[:]
        forces = element(g, "forces").value[:]
        energy = element(f, "observables/atoms/energy").value[:]
        # cell = g.box.edges.value[:]

    assert len(position) == len(md)
    for idx, atoms in enumerate(md):
        npt.assert_array_equal(position[idx], atoms.get_positions())
        npt.assert_array_equal(species[idx], atoms.get_atomic_numbers())
        npt.assert_array_equal(momentum[idx], atoms.get_momenta())
        npt.assert_array_equal(forces[idx], atoms.get_forces())
        npt.assert_array_equal(energy[idx], atoms.get_potential_energy())
        # npt.assert_array_equal(cell[idx], atoms.get_cell().diagonal())
