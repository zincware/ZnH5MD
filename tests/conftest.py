import pathlib

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
