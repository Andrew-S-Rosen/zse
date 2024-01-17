"""
This module contains functions for doing reverse Monte Carlo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from zse.rmc_utilities import is_non_lowenstein, make_ratio_randomized

if TYPE_CHECKING:
    from numpy.typing import NDArray


def calculate_alpha(
    atoms: Atoms,
    heteroatom: str = "Al",
    distance_matrix: NDArray | None = None,
) -> float:
    """
    Given a zeolite, calculate the Warren-Cowley parameter, defined as follows:

    alpha_j = 1 - P_{j}/x_Si

    where P_j is the probability that a heteroatom is surrounded by a Si atom
    in the j-th coordination shell, and x_Si is the Si:heteroatom fraction.

    A value of 0 indicates a random distribution of heteroatoms, while a value
    of 1 indicates clustering. A negative value indicates sparsity.

    Here, we have chosen to set j = 2 per the convention in the literature.

    Reference:
    - https://pubs.acs.org/doi/10.1021/acs.jpcc.8b03475

    Parameters
    ----------
    atoms : Atoms
        The zeolite to analyze.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    distance_matrix : NDArray, optional
        A precomputed distance matrix, by default None.

    Returns
    -------
    float
        The Warren-Cowley parameter.
    """
    atoms = atoms.copy()

    # Strip off any cations and such. We are only interested in Si and the heteroatom,
    # as well as O to define the coordination shells.
    del atoms[
        [atom.index for atom in atoms if atom.symbol not in ["Si", heteroatom, "O"]]
    ]

    heteroatom_indices = [atom.index for atom in atoms if atom.symbol == heteroatom]
    if len(heteroatom_indices) == 0:
        raise ValueError(
            "There are no heteroatoms, so the Warren-Cowley parameter is undefined."
        )

    Si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    T_indices = Si_indices + heteroatom_indices
    x_Si = len(Si_indices) / len(T_indices)
    distance_matrix = (
        distance_matrix
        if distance_matrix is not None
        else atoms.get_all_distances(mic=True)
    )

    count = 0
    total = 0
    for heteroatom_index in heteroatom_indices:
        # Here, we use the fact that each T site must have four O neighbors
        distances = distance_matrix[heteroatom_index, :].copy()
        distances[heteroatom_index] = np.inf
        O_neighbor_indices = np.argsort(distances)[:4]

        T_neighbors = []
        for O_neighbor_index in O_neighbor_indices:
            # Here, we use the fact that each O site must be bound to only
            # two T sites, one of which was already counted above
            distances = distance_matrix[O_neighbor_index, :].copy()
            distances[heteroatom_index] = np.inf
            distances[O_neighbor_index] = np.inf
            T_neighbor = np.argsort(distances)[0]
            T_neighbors.append(T_neighbor)

        Si_in_second_shell = [
            index for index in T_neighbors if atoms[index].symbol == "Si"
        ]
        count += len(Si_in_second_shell)
        total += len(T_neighbors)

    P_j = count / total

    return 1 - P_j / x_Si


def rmc_simulation(
    atoms: Atoms,
    heteroatom: str = "Al",
    enforce_lowenstein: bool = True,
    beta: float = 0.005,
    max_steps: int = 100000,
    stop_tol: float = 0.01,
    alpha_target: float | None = -1.0,
    minimize_alpha: bool = False,
    maximize_alpha: bool = False,
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
    enforce_lowenstein : bool, optional
        Whether to enforce the Lowenstein rule, by default True.
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
        or maximize_alpha is True, this value will be ignored.
    minimize_alpha : bool, optional
        Whether to minimize the Warren-Cowley parameter, by default False.
        This is equivalent to setting alpha_target = -1.0 but will also return
        the structure that minimizes alpha rather than the last structure
        generated.
    maximize_alpha : bool, optional
        Whether to maximize the Warren-Cowley parameter, by default False.
        This is equivalent to setting alpha_target = 1.0 but will also return
        the structure that maximizes alpha rather than the last structure
        generated.
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

    def _random_swap(atoms: Atoms, T_indices: list[int]) -> list[int]:
        swap_indices = np.random.choice(T_indices, size=2, replace=False)
        proposed_atoms = atoms.copy()
        (
            proposed_atoms[swap_indices[0]].symbol,
            proposed_atoms[swap_indices[1]].symbol,
        ) = (
            proposed_atoms[swap_indices[1]].symbol,
            proposed_atoms[swap_indices[0]].symbol,
        )
        return proposed_atoms, swap_indices

    current_atoms = atoms.copy()
    if minimize_alpha and maximize_alpha:
        raise ValueError("Must specify one of minimize_alpha, maximize_alpha.")
    elif minimize_alpha:
        alpha_target = -1.0
    elif maximize_alpha:
        alpha_target = 1.0

    T_indices = [atom.index for atom in atoms if atom.symbol in ["Si", heteroatom]]
    distance_matrix = current_atoms.get_all_distances(mic=True)
    current_alpha = calculate_alpha(
        current_atoms,
        heteroatom=heteroatom,
        distance_matrix=distance_matrix,
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
                distance_matrix=distance_matrix,
            ):
                proposed_atoms, _ = _random_swap(current_atoms, T_indices)

        # Calculate alpha for the proposed structure
        proposed_alpha = calculate_alpha(
            proposed_atoms, heteroatom=heteroatom, distance_matrix=distance_matrix
        )

        # Calculate acceptance probability
        dE_proposed = np.abs(proposed_alpha - alpha_target)
        acceptance_prob = (
            1.0 if dE_proposed < dE_current else np.exp(-beta * dE_proposed)
        )

        # Accept or reject the move
        rand = np.random.rand()
        if rand < acceptance_prob:
            current_atoms = proposed_atoms
            current_alpha = proposed_alpha
            stored_atoms.append(current_atoms)
            stored_alphas.append(current_alpha)
            if verbose:
                print(i, current_alpha)

    if minimize_alpha:
        min_alpha_index = np.argsort(stored_alphas)[0]
        return stored_atoms[min_alpha_index], stored_alphas[min_alpha_index]
    elif maximize_alpha:
        max_alpha_index = np.argsort(stored_alphas)[-1]
        return stored_atoms[max_alpha_index], stored_alphas[max_alpha_index]
    else:
        return current_atoms, current_alpha
