"""
This module contains functions for doing reverse Monte Carlo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from zse.rmc_utilities import (
    get_kth_neighbors,
    is_non_lowenstein,
    make_graph,
    make_ratio_randomized,
)

if TYPE_CHECKING:
    from networkx import Graph


def calculate_alpha(
    atoms: Atoms,
    heteroatom: str = "Al",
    j: int = 2,
    graph: Graph | None = None,
) -> float:
    """
    Given a zeolite, calculate the Warren-Cowley parameter, defined as follows:

    alpha_j = 1 - P_{j}/x_Si

    where P_j is the probability of finding a Si atom as the j-th nearest neighbor
    of a heteroatom, and x_Si is the fraction of Si to heteroatom.

    A value of 0 for alpha_j indicates a random distribution of heteroatoms,
    while a value of 1 indicates clustering. A negative value indicates sparsity.

    Reference:
    - https://pubs.acs.org/doi/10.1021/acs.jpcc.8b03475

    Parameters
    ----------
    atoms : Atoms
        The zeolite to analyze.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    j : int
        The coordination shell to consider, by default 2.
        In the context of zeolites, we only consider T sites in the graph,
        so T1 and T2 in -T1-O-T2- are j = 1 apart, -T1-O-T-O-T2- are j = 2 apart,
        and so on.
    graph : nx.Graph, optional
        The graph representation of the structure, by default None.
        If None, one will be generated.

    Returns
    -------
    float
        The Warren-Cowley parameter.
    """

    graph = graph if graph is not None else make_graph(atoms)

    heteroatom_indices = [atom.index for atom in atoms if atom.symbol == heteroatom]
    if len(heteroatom_indices) == 0:
        raise ValueError(
            "There are no heteroatoms, so the Warren-Cowley parameter is undefined."
        )

    Si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    T_indices = Si_indices + heteroatom_indices
    x_Si = len(Si_indices) / len(T_indices)

    count = 0
    total = 0
    for heteroatom_index in heteroatom_indices:
        indices_in_shell = get_kth_neighbors(graph, heteroatom_index, j * 2)
        Si_in_second_shell = [
            index_in_shell
            for index_in_shell in indices_in_shell
            if atoms[index_in_shell].symbol == "Si"
        ]
        count += len(Si_in_second_shell)
        total += len(indices_in_shell)

    P_j = count / total

    return 1 - P_j / x_Si


def rmc_simulation(
    atoms: Atoms,
    heteroatom: str = "Al",
    beta: float = 0.005,
    max_steps: int = 100000,
    stop_tol: float = 0.01,
    alpha_target: float = -1.0,
    enforce_lowenstein: bool = True,
    j: int = 2,
    verbose: bool = True,
) -> tuple[Atoms, float]:
    """
    Run an atomistic reverse Monte Carlo simulation to generate a zeolite with
    a desired (second-shell) Warren-Cowley parameter.

    References:
    - https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.014301
    - https://pubs.acs.org/doi/10.1021/acs.jpcc.8b03475

    Parameters
    ----------
    atoms : Atoms
        The zeolite to modify.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    beta : float, optional
        The inverse RMC temperature, by default 0.005. Larger values
        will decrease the likelihood of accepting a move that shifts
        the Warren-Cowley parameter away from the target value.
    max_steps : int, optional
        The maximum number of steps to take, by default 100000.
    stop_tol : float, optional
        The tolerance for the Warren-Cowley parameter, by default 0.01.
        If the difference between the current and target values is less
        than this, the simulation will stop.
    alpha_target : float, optional
        The target Warren-Cowley parameter, by default -1.0. If minimize_alpha
        or maximize_alpha is True, this value will be ignored. Takes a value
        between -1 and 1.
    enforce_lowenstein : bool, optional
        Whether to enforce the Lowenstein rule, by default True.
    j : int, optional
        The coordination shell to consider in alpha_j, by default 2.
    verbose : bool, optional
        Whether to print the current value of alpha at each step, by default False.
        Formatted as (step, alpha).

    Returns
    -------
    Atoms
        The modified zeolite.
    float
        The Warren-Cowley parameter of the modified zeolite.
    """

    def _random_swap(atoms: Atoms, T_indices: list[int]) -> tuple[Atoms, list[int]]:
        proposed_atoms = atoms.copy()
        swap_indices = np.random.choice(T_indices, size=2, replace=False)
        (
            proposed_atoms[swap_indices[0]].symbol,
            proposed_atoms[swap_indices[1]].symbol,
        ) = (
            proposed_atoms[swap_indices[1]].symbol,
            proposed_atoms[swap_indices[0]].symbol,
        )
        return proposed_atoms, swap_indices

    current_atoms = make_ratio_randomized(
        atoms, heteroatom=heteroatom, enforce_lowenstein=enforce_lowenstein
    )

    T_indices = [atom.index for atom in atoms if atom.symbol in ["Si", heteroatom]]
    graph = make_graph(atoms)
    current_alpha = calculate_alpha(
        current_atoms,
        heteroatom=heteroatom,
        j=j,
        graph=graph,
    )

    stored_atoms = []
    stored_alphas = []
    for i in range(max_steps):
        dE_current = np.abs(current_alpha - alpha_target)
        if dE_current < stop_tol:
            break

        # Attempt to swap their identities
        proposed_atoms, _ = _random_swap(current_atoms, T_indices)

        if enforce_lowenstein:
            while is_non_lowenstein(
                proposed_atoms,
                heteroatom=heteroatom,
                graph=graph,
            ):
                proposed_atoms, _ = _random_swap(current_atoms, T_indices)

        # Calculate alpha for the proposed structure
        proposed_alpha = calculate_alpha(
            proposed_atoms, heteroatom=heteroatom, j=j, graph=graph
        )

        # Calculate acceptance probability
        dE_proposed = np.abs(proposed_alpha - alpha_target)
        acceptance_prob = (
            1.0 if dE_proposed < dE_current else np.exp(-beta * dE_proposed)
        )

        # Accept or reject the move
        if acceptance_prob >= np.random.rand():
            current_atoms = proposed_atoms
            current_alpha = proposed_alpha
            stored_atoms.append(current_atoms)
            stored_alphas.append(current_alpha)
            if verbose:
                print(i, current_alpha)

    closest_alpha_index = np.argsort(np.abs(np.array(stored_alphas) - alpha_target))[0]
    return stored_atoms[closest_alpha_index], stored_alphas[closest_alpha_index]
