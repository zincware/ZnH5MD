import ase.build
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
                "lst": [1, 2, 3],
                # "lst_str": ["Jane Doe", "John Doe"],
                # "lst_array": [np.random.rand(3), np.random.rand(3)],
            }
        )
        atoms.new_array("mlip_forces", np.random.rand(len(atoms), 3))
        atoms.new_array("mlip_forces_2", np.random.rand(len(atoms), 3))
        # atoms.arrays["arr_lst_arr"] = [np.random.rand(3) for _ in range(len(atoms))]
        # atoms.arrays["arr_lst"] = [[1, 2, 3] for _ in range(len(atoms))]
        # atoms.new_array("arr_str", np.array(["abc" for _ in range(len(atoms))]))
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
        # add something to info
        atoms.info["mlip_energy"] = np.random.rand()
        # add something to arrays
        atoms.new_array("mlip_forces", np.random.rand(len(atoms), 3))
        images.append(atoms)
    return images


@pytest.fixture
def s22_illegal_calc_results() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results["mlip_energy"] = np.random.rand()
        atoms.calc.results["dict"] = {"author": "Jane Doe", "date": "2021-09-01"}
        atoms.calc.results["float"] = 3.14
        atoms.calc.results["int"] = 42
        atoms.calc.results["list"] = [1, 2, 3]
        atoms.calc.results["str"] = '{"author": "Jane Doe", "date": "2021-09-01"}'
        atoms.calc.results["list_array"] = [np.random.rand(3), np.random.rand(3)]
        atoms.calc.results["list_str"] = ["Jane Doe", "John Doe"]
        # atoms.calc.results["list_dict"] = [
        #     {"author": "Jane Doe", "date": "2021-09-01"},
        #     {"author": "John Doe", "date": "2021-09-02"},
        # ]
        images.append(atoms)
    return images


@pytest.fixture
def water() -> list[ase.Atoms]:
    """Get a dataset without positions."""
    return [ase.Atoms("H2O")]


@pytest.fixture
def s22_no_ascii() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms.info["config"] = "βγ"
        images.append(atoms)
    return images


@pytest.fixture
def frames_with_residuenames() -> list[ase.Atoms]:
    water = ase.build.molecule("H2O")
    # typical PDB array data
    water.arrays["residuenames"] = np.array(["H2O"] * len(water))
    water.arrays["atomtypes"] = np.array(["γO", "βH", "βH"])

    ethane = ase.build.molecule("C2H6")
    ethane.arrays["residuenames"] = np.array(["C2H6"] * len(ethane))
    ethane.arrays["atomtypes"] = np.array(["γC", "βH", "βH", "βH", "βH", "βH"])
    return [water, ethane]


@pytest.fixture
def s22_info_arrays_calc_missing_inbetween() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms: ase.Atoms
        if np.random.random() > 0.5:
            atoms.info.update({"mlip_energy": np.random.rand()})
        if np.random.random() > 0.5:
            atoms.new_array("mlip_forces", np.random.rand(len(atoms), 3))
        if np.random.random() > 0.5:
            calc = SinglePointCalculator(atoms)
            set_calc = False
            if np.random.random() > 0.5:
                calc.results["energy"] = np.random.rand()
                set_calc = True
            if np.random.random() > 0.5:
                calc.results["forces"] = np.random.rand(len(atoms), 3)
                set_calc = True
            if set_calc:
                atoms.calc = calc
        images.append(atoms)
    return images


@pytest.fixture
def s22_nested_calc() -> list[ase.Atoms]:
    images = []
    for atoms in ase.collections.s22:
        atoms: ase.Atoms
        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results["forces"] = np.random.rand(len(atoms), 3)
        atoms.calc.results["forces_contributions"] = [
            [
                np.random.rand(len(atoms), 3),
                np.random.rand(len(atoms), 3),
            ]
        ]
        images.append(atoms)
    return images


@pytest.fixture
def full_water(water) -> ase.Atoms:
    """Get a dataset with full water molecules."""
    # add a calculator and info and arrays
    water = water[0]
    water.calc = SinglePointCalculator(
        water, energy=1.0, forces=np.zeros((len(water), 3))
    )
    water.info["smiles"] = "O"
    water.arrays["mlip_forces"] = np.zeros((len(water), 3))

    return water
