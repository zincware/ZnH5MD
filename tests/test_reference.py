"""Test against pyh5md reference implementation."""

import os

import ase
import numpy as np
from pyh5md import File, element

import znh5md


def test_open(example_h5):
    with File(example_h5, "r") as f:
        print(f)


def test_DataWriter(tmp_path):
    os.chdir(tmp_path)

    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    with File("db.h5", "w") as f:
        at = f.particles_group("atoms")

        # Creating position data
        r = np.zeros((100, 3), dtype=np.float64)
        at_pos = element(at, "position", store="time", data=r, time=True)

        # Creating species
        s = np.ones(r.shape[:1])
        # element(at, 'species', data=s, store='fixed')
        at_species = element(
            at, "species", data=s, store="time", step_from=at_pos, time=True
        )

        # Creating velocity data
        v = np.zeros((100, 3), dtype=np.float64)
        at_v = element(
            at, "velocity", store="time", data=v, step_from=at_pos, time=True
        )

        # Create an observable
        com = r.mean(axis=0)
        obs_com = element(
            f, "observables/center_of_mass", store="linear", data=com, step=10
        )
        # Create a scalar time independent observable
        element(f, "observables/random_number", data=np.random.random(), store="fixed")

        edges = (1.0, 1.0, 1.0)
        box = at.create_box(
            dimension=3,
            boundary=["none", "none", "none"],
            store="time",
            data=edges,
            step_from=at_pos,
        )

        DT = 0.1
        time = 0.0

        def dump(t):
            if t % 10 == 0:
                at_pos.append(r, t, t * DT)
                at_v.append(v, t, t * DT)
                obs_com.append(r.mean(axis=0))
                at.box.edges.append(edges, t, t * DT)
                at_species.append(s, t, t * DT)

        dump(0)
        t = 0
        t_max = 200
        while t < t_max:
            r += DT * 0.5 * v
            v += DT * np.random.normal(0.0, 1.0, v.shape)
            r += DT * 0.5 * v
            time += DT
            t += 1
            dump(t)
    data = znh5md.ASEH5MD("db.h5")

    new_atoms = data.get_atoms_list()[0]
