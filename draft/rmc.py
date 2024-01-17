from __future__ import annotations
import numpy as np
from ase import Atoms

def make_ratio_randomized(atoms:Atoms,ratio:float, heteroatom:str="Al")->Atoms:
    atoms = atoms.copy()
    for atom in atoms:
        if atom.symbol == heteroatom:
            atom.symbol = "Si"

    Si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    n_to_exchange = round(len(Si_indices)/(1+ratio))
    
    for _ in range(n_to_exchange):
        index = np.random.choice(Si_indices)
        atoms[index].symbol = heteroatom
        Si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    
    return atoms

def calculate_alpha(atoms: Atoms, heteroatom: str = "Al", precomputed_dist_mat=None) -> float:
    atoms = atoms.copy()
    del atoms[[atom.index for atom in atoms if atom.symbol not in ["Si", heteroatom, "O"]]]

    heteroatom_indices = [atom.index for atom in atoms if atom.symbol == heteroatom]
    if len(heteroatom_indices) == 0:
        raise ValueError("There are no heteroatoms, so alpha is undefined.")

    Si_indices = [atom.index for atom in atoms if atom.symbol == "Si"]
    T_indices = Si_indices + heteroatom_indices
    x_Si = len(Si_indices) / len(T_indices)
    if precomputed_dist_mat is not None:
        dist_mat = precomputed_dist_mat
    else:
        dist_mat = atoms.get_all_distances(mic=True)

    count = 0
    total = 0
    for i in heteroatom_indices:
        distances = dist_mat[i, :].copy()
        distances[i] = np.inf
        O_neighbors = np.argsort(distances)[:4]

        T_neighbors = []
        for j in O_neighbors:
            distances = dist_mat[j, :].copy()
            distances[i] = np.inf
            distances[j] = np.inf
            T_neighbor = np.argsort(distances)[0]
            T_neighbors.append(T_neighbor)

        Si_in_second_shell = [index for index in T_neighbors if atoms[index].symbol == "Si"]
        count += len(Si_in_second_shell)
        total += len(T_neighbors)

    P_j = count / total

    alpha_j = 1 - P_j / x_Si

    return alpha_j

def rmc_simulation(
    atoms: Atoms,heteroatom:str = "Al", alpha_targ: float = -1, beta: float = 0.005, n_steps: int = 100000, stop_tol: float = 0.01, sparse:bool = False, dense: bool = False,
):
    if sparse:
        alpha_targ = -1
    if dense:
        alpha_targ = 1
    current_atoms = atoms.copy()
    dist_mat = current_atoms.get_all_distances(mic=True)
    alpha_current = calculate_alpha(current_atoms,heteroatom=heteroatom,precomputed_dist_mat=dist_mat)
    T_indices = [atom.index for atom in atoms if atom.symbol in ["Si",heteroatom]]
    stored_atoms = []
    stored_alphas = []
    for i in range(n_steps):

        dE_current = np.abs(alpha_current-alpha_targ)
        if dE_current < stop_tol:
            break

        # Choose a random pair of atoms
        swap_indices = np.random.choice(
            T_indices, size=2, replace=False
        )
        

        # Attempt to swap their identities
        proposed_atoms = current_atoms.copy()
        proposed_atoms[swap_indices[0]].symbol, proposed_atoms[swap_indices[1]].symbol = (
            proposed_atoms[swap_indices[1]].symbol,
            proposed_atoms[swap_indices[0]].symbol,
        )

        # Calculate alpha for the proposed structure
        alpha_proposed = calculate_alpha(proposed_atoms,heteroatom=heteroatom,precomputed_dist_mat=dist_mat)

        # Calculate acceptance probability
        dE_proposed = np.abs(alpha_proposed - alpha_targ)
        acceptance_prob = (
            1.0
            if dE_proposed < dE_current
            else np.exp(-beta * dE_proposed)
        )

        # Accept or reject the move
        rand = np.random.rand()
        if rand < acceptance_prob:
            current_atoms = proposed_atoms
            alpha_current = alpha_proposed
            stored_atoms.append(current_atoms)
            stored_alphas.append(alpha_current)
            print(i, alpha_current)
    
    if sparse:
        min_alpha_index = np.argsort(stored_alphas)[0]
        return stored_atoms[min_alpha_index], stored_alphas[min_alpha_index]
    elif dense:
        max_alpha_index = np.argsort(stored_alphas)[-1]
        return stored_atoms[max_alpha_index], stored_alphas[max_alpha_index]
    else:
        return current_atoms, alpha_current