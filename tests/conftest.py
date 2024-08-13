import ase.collections
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture
def s22() -> list[ase.Atoms]:
    return list(ase.collections.s22)


@pytest.fixture
def s22_energy() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        calc = SinglePointCalculator(atoms, energy=np.random.rand())
        atoms.calc = calc
        images.append(atoms)
    return images


@pytest.fixture
def s22_energy_forces() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        calc = SinglePointCalculator(
            atoms, energy=np.random.rand(), forces=np.random.rand(len(atoms), 3)
        )
        atoms.calc = calc
        images.append(atoms)
    return images


@pytest.fixture
def s22_all_properties() -> list[ase.Atoms]:
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
        born_effective_charges = np.random.rand(len(atoms), 3)
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


@pytest.fixture
def s22_info_arrays_calc() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms: ase.Atoms
        atoms.info.update(
            {
                "mlip_energy": np.random.rand(),
                "mlip_energy_2": np.random.rand(),
                "mlip_stress": np.random.rand(6),
                "collection": "s22",
                "metadata": {"author": "Jane Doe", "date": "2021-09-01"},
            }
        )
        atoms.new_array("mlip_forces", np.random.rand(len(atoms), 3))
        atoms.new_array("mlip_forces_2", np.random.rand(len(atoms), 3))
        atoms.set_velocities(np.random.rand(len(atoms), 3))
        calc = SinglePointCalculator(
            atoms, energy=np.random.rand(), forces=np.random.rand(len(atoms), 3)
        )
        atoms.calc = calc
        images.append(atoms)
    return images


@pytest.fixture
def s22_mixed_pbc_cell() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms.set_pbc(np.random.rand(3) > 0.5)
        atoms.set_cell(np.random.rand(3, 3))
        images.append(atoms)
    return images


@pytest.fixture
def s22_illegal_calc_results() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results["mlip_energy"] = np.random.rand()

        images.append(atoms)
    return images


@pytest.fixture
def water() -> list[ase.Atoms]:
    """Get a dataset without positions."""
    return [ase.Atoms("H2O")]
