import ase.collections
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture
def s22():
    return ase.collections.s22


@pytest.fixture
def s22_energy():
    images = []
    for atoms in ase.collections.s22:
        calc = SinglePointCalculator(atoms, energy=np.random.rand())
        atoms.set_calculator(calc)
        images.append(atoms)
    return images


@pytest.fixture
def s22_all_properties():
    images = []
    for atoms in ase.collections.s22:
        # shapes taken from https://gitlab.com/ase/ase/-/blob/master/ase/outputs.py
        energy = np.random.rand()
        energies = np.random.rand(len(atoms))
        free_energy = np.random.rand()

        forces = np.random.rand(len(atoms), 3)
        stress = np.random.rand(6)
        stresses = np.random.rand(len(atoms), 6)

        dipole = np.random.rand(3)
        magmom = np.random.rand()
        magmoms = np.random.rand(len(atoms))

        dielectric_tensor = np.random.rand(3, 3)
        born_effective_charges = np.random.rand(len(atoms), 3, 3)
        polarization = np.random.rand(3)

        calc = SinglePointCalculator(
            atoms,
            energy=energy,
            energies=energies,
            free_energy=free_energy,
            forces=forces,
            stress=stress,
            stresses=stresses,
            dipole=dipole,
            magmom=magmom,
            magmoms=magmoms,
            dielectric_tensor=dielectric_tensor,
            born_effective_charges=born_effective_charges,
            polarization=polarization,
        )

        atoms.calc = calc
        images.append(atoms)
    return images
