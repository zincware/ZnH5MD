import pathlib
import random

import ase.build
import ase.calculators.singlepoint
import ase.collections
import ase.io
import h5py
import numpy as np
import pytest


@pytest.fixture
def example_h5(tmp_path) -> pathlib.Path:
    filename = tmp_path / "example.h5"

    n_steps = 100
    n_particles = 10

    with h5py.File(filename, "w") as file:
        particles = file.create_group("particles")
        atoms = particles.create_group("atoms")
        position = atoms.create_group("position")
        position.create_dataset(
            "value",
            data=np.arange(n_steps * n_particles * 3).reshape((n_steps, n_particles, 3)),
        )
        position.create_dataset("time", data=np.linspace(0, 1, n_steps))
        position.create_dataset("step", data=np.arange(n_steps))

        species = atoms.create_group("species")
        species.create_dataset(
            "value",
            data=np.concatenate(
                [
                    np.ones((n_steps, int(n_particles / 2))),
                    2 * np.ones((n_steps, int(n_particles / 2))),
                ],
                axis=1,
            ),
        )
        species.create_dataset("time", data=np.linspace(0, 1, n_steps))
        species.create_dataset("step", data=np.arange(n_steps))

    return filename


@pytest.fixture
def atoms_list(request) -> list[ase.Atoms]:
    """
    Generate ase.Atoms objects with random positions and increasing energy
    and random force values

    Parameters
    ----------
    request
        pytest request object. Possible values for the "param" attribute are:
        - None: use default values
        - "vary_size": use ase.collections.g2
        - "no_stress": do not set stress
    """
    if getattr(request, "param", None) == "vary_size":
        atoms = [ase.build.molecule(x) for x in ase.collections.g2.names]
    else:
        random.seed(1234)
        atoms = [
            ase.Atoms(
                "CO",
                positions=[(0, 0, 0), (0, 0, random.random())],
                cell=(1, 1, 1),
                pbc=True,
            )
            for _ in range(21)
        ]
    # TODO create some variations in PBC
    # atoms[0].pbc = np.array([True, True, False])
    # atoms[1].pbc = np.array([True, False, True])
    # atoms[2].pbc = False
    for atom in atoms:
        atom.pbc = True

    for idx, atom in enumerate(atoms):
        stress = (
            np.random.rand(6) if getattr(request, "param", None) != "no_stress" else None
        )
        atom.calc = ase.calculators.singlepoint.SinglePointCalculator(
            atoms=atom,
            energy=idx / 21,
            forces=np.random.rand(len(atom), 3),
            stress=stress,
        )

    return atoms
