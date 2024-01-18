"""
This module contains utility functions for doing reverse Monte Carlo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import build_neighbor_list

if TYPE_CHECKING:
    from networkx import Graph


def clean_zeolite(atoms: Atoms, allowed_elements: list[str] = None) -> Atoms:
    """
    Clean a zeolite such that it consists only of `allowed_elements`.

    Parameters
    ----------
    atoms : Atoms
        The zeolite to clean.
    allowed_elements : list[str], optional
        The elements to keep, by default None.
        If None, only Si, Al, and O are kept.

    Returns
    -------
    Atoms
        The cleaned zeolite.
    """
    new_atoms = atoms.copy()
    allowed_elements = (
        allowed_elements if allowed_elements is not None else ["Si", "Al", "O"]
    )
    remove_indices = [
        atom.index for atom in atoms if atom.symbol not in allowed_elements
    ]
    del new_atoms[remove_indices]
    return new_atoms


def make_graph(atoms: Atoms) -> Graph:
    """
    Make a networkX graph representation of an ASE Atoms object.

    Parameters
    ----------
    atoms : Atoms
        The structure to convert.

    Returns
    -------
    nx.Graph
        The graph representation of the structure.
    """
    nl = build_neighbor_list(atoms, self_interaction=False, bothways=True)
    return nx.from_numpy_array(nl.get_connectivity_matrix(), parallel_edges=True)


def get_kth_neighbors(graph: Graph, index: int, k: int) -> list[int]:
    """
    Get the indices of all atoms that are k bonds away from a given atom.

    Parameters
    ----------
    graph : nx.Graph
        The graph representation of the structure.
    index : int
        The index of the atom to check.
    k : int
        The number of bonds away to check.

    Returns
    -------
    list[int]
        The indices of all atoms that are k bonds away from the given atom.
    """
    node_distances = nx.single_source_shortest_path_length(graph, index, cutoff=k)
    kth_neighbors = [node for node, distance in node_distances.items() if distance == k]
    return kth_neighbors


def is_non_lowenstein(
    atoms: Atoms,
    heteroatom: str = "Al",
    graph: Graph | None = None,
) -> bool:
    """
    Determines if a given structure is non-Lowenstein, i.e. has
    T-O-T sites where T is the heteroatom.

    Parameters
    ----------
    atoms : Atoms
        The structure to check.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    graph
        The graph representation of the structure, by default None.
        If None, it will be computed.

    Returns
    -------
    bool
        True if the structure is non-Lowenstein, False otherwise.
    """

    graph = graph if graph is not None else make_graph(atoms)

    heteroatom_indices = [atom.index for atom in atoms if atom.symbol == heteroatom]

    for index in heteroatom_indices:
        indices_in_shell = get_kth_neighbors(graph, index, 2)
        close_heteroatoms = [
            index_in_shell
            for index_in_shell in indices_in_shell
            if atoms[index_in_shell].symbol == heteroatom
        ]
        if np.sum(close_heteroatoms) > 0:
            return True
    return False


def make_ratio_randomized(
    atoms: Atoms,
    ratio: float | None = None,
    heteroatom: str = "Al",
    enforce_lowenstein: bool = True,
) -> Atoms:
    """
    Make a zeolite with a given Si/heteroatom ratio by randomly exchanging
    Si atoms with heteroatoms.

    Parameters
    ----------
    atoms : Atoms
        The zeolite to modify.
    ratio : float
        The desired Si/heteroatom ratio. If None, use the current
        ratio.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    enforce_lowenstein : bool, optional
        Whether to enforce the Lowenstein rule, by default True.

    Returns
    -------
    Atoms
        The modified zeolite.
    """

    def _random_swap(
        atoms: Atoms, Si_indices: list[int], heteroatom: str = "Al"
    ) -> Atoms:
        proposed_atoms = atoms.copy()
        swap_index = np.random.choice(Si_indices)
        proposed_atoms[swap_index].symbol = heteroatom
        return proposed_atoms, swap_index

    current_atoms = atoms.copy()

    if ratio is None:
        chemical_symbols = atoms.get_chemical_symbols()
        ratio = chemical_symbols.count("Si") / chemical_symbols.count(heteroatom)

    # Start with an all-Si zeolite since we are going to randomize this
    for atom in current_atoms:
        if atom.symbol == heteroatom:
            atom.symbol = "Si"

    # Find how many Si atoms to exchange
    Si_indices = [atom.index for atom in current_atoms if atom.symbol == "Si"]
    n_to_exchange = round(len(Si_indices) / (1 + ratio))

    # Pre-compute this for later
    graph = make_graph(current_atoms)

    # Randomly exchange Si atoms with heteroatoms
    for _ in range(n_to_exchange):
        proposed_atoms, swap_index = _random_swap(
            current_atoms, Si_indices, heteroatom=heteroatom
        )

        # Check that the resulting structure is Lowenstein
        if enforce_lowenstein:
            while is_non_lowenstein(proposed_atoms, heteroatom=heteroatom, graph=graph):
                Si_indices.remove(swap_index)
                if len(Si_indices) == 0:
                    raise ValueError("This zeolite cannot be made Lowenstein")
                proposed_atoms, swap_index = _random_swap(
                    current_atoms, Si_indices, heteroatom=heteroatom
                )

        current_atoms = proposed_atoms
        Si_indices = [atom.index for atom in current_atoms if atom.symbol == "Si"]

    return current_atoms
