"""
This module contains utility functions for doing reverse Monte Carlo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_non_lowenstein(
    atoms: Atoms,
    heteroatom: str = "Al",
    d_cutoff: float = 3.5,
    distance_matrix: NDArray | None = None,
) -> bool:
    """
    Determines if a given structure is non-Lowenstein, defined as having
    at least one heteroatom-heteroatom distance less than d_cutoff.

    Parameters
    ----------
    atoms : Atoms
        The structure to check.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    d_cutoff : float, optional
        The minimum allowable heteroatom-heteroatom distance, by default 3.5.
    distance_matrix : NDArray, optional
        A precomputed distance matrix, by default None.

    Returns
    -------
    bool
        True if the structure is non-Lowenstein, False otherwise.
    """

    distance_matrix = (
        distance_matrix
        if distance_matrix is not None
        else atoms.get_all_distances(mic=True)
    )

    heteroatom_indices = [atom.index for atom in atoms if atom.symbol == heteroatom]

    for index in heteroatom_indices:
        distances = distance_matrix[index, heteroatom_indices]
        values_in_range = (distances > 0) & (distances < d_cutoff)
        close_heteroatoms = np.sum(values_in_range)
        if close_heteroatoms > 0:
            return True
    return False


def make_ratio_randomized(
    atoms: Atoms,
    ratio: float,
    heteroatom: str = "Al",
    d_cutoff: float = 3.5,
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
        The desired Si/heteroatom ratio.
    heteroatom : str, optional
        The symbol of the heteroatom, by default "Al".
    d_cutoff : float, optional
        The minimum allowable heteroatom-heteroatom distance, by default 3.5.
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
        swap_index = np.random.choice(Si_indices)
        proposed_atoms = atoms.copy()
        proposed_atoms[swap_index].symbol = heteroatom
        return proposed_atoms, swap_index

    current_atoms = atoms.copy()

    # Start with an all-Si zeolite since we are going to randomize this
    for atom in current_atoms:
        if atom.symbol == heteroatom:
            atom.symbol = "Si"

    # Find how many Si atoms to exchange
    Si_indices = [atom.index for atom in current_atoms if atom.symbol == "Si"]
    n_to_exchange = round(len(Si_indices) / (1 + ratio))

    # Pre-compute this for later
    distance_matrix = current_atoms.get_all_distances(mic=True)

    # Randomly exchange Si atoms with heteroatoms
    for _ in range(n_to_exchange):
        proposed_atoms, swap_index = _random_swap(
            current_atoms, Si_indices, heteroatom=heteroatom
        )

        # Check that the resulting structure is Lowenstein
        if enforce_lowenstein:
            while is_non_lowenstein(
                proposed_atoms,
                heteroatom=heteroatom,
                d_cutoff=d_cutoff,
                precomputed_distance_matrix=distance_matrix,
            ):
                Si_indices.remove(swap_index)
                if len(Si_indices) == 0:
                    raise ValueError("This zeolite cannot be made Lowenstein")
                proposed_atoms, swap_index = _random_swap(
                    current_atoms, Si_indices, heteroatom=heteroatom
                )

        current_atoms = proposed_atoms
        Si_indices = [atom.index for atom in current_atoms if atom.symbol == "Si"]

    return current_atoms
