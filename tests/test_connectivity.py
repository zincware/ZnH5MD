import ase.build
import networkx as nx
import numpy as np
import pytest
from ase.neighborlist import natural_cutoffs


def build_graph(atoms: ase.Atoms) -> nx.Graph:
    cutoffs = [
        1.0,
    ]
    atoms_copy = atoms.copy()
    connectivity_matrix = np.zeros((len(atoms_copy), len(atoms_copy)), dtype=int)
    atoms_copy.pbc = False
    distance_matrix = atoms_copy.get_all_distances(mic=False)
    np.fill_diagonal(distance_matrix, np.inf)
    for cutoff in cutoffs:
        cutoffs = np.array(natural_cutoffs(atoms_copy, mult=cutoff))
        cutoffs = cutoffs[:, None] + cutoffs[None, :]
        connectivity_matrix[distance_matrix <= cutoffs] += 1
    return nx.from_numpy_array(connectivity_matrix)


def set_connectivity(atoms: ase.Atoms, graph: nx.Graph):
    bonds = []
    for edge in graph.edges:
        bonds.append((edge[0], edge[1], graph.edges[edge]["weight"]))
    atoms.connectivity = bonds


@pytest.fixture
def connected_atoms() -> list[ase.Atoms]:
    """Create ase atoms that are connected."""
    water = ase.build.molecule("H2O")
    ammonia = ase.build.molecule("NH3")
    methane = ase.build.molecule("CH4")

    set_connectivity(water, build_graph(water))
    set_connectivity(ammonia, build_graph(ammonia))
    set_connectivity(methane, build_graph(methane))

    return [water, ammonia, methane]


def test_connectivity(tmp_path, connected_atoms):
    assert connected_atoms[0].connectivity == [(0, 1, 1), (0, 2, 1)]
