from __future__ import annotations

import random
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from tqdm import tqdm

from zse.cation import monovalent
from zse.collections.framework import get_framework
from zse.substitute import tsub
from zse.t_utilities import get_T_info, get_T_info_exchangeable
from zse.utilities import get_unique_structures, site_labels

if TYPE_CHECKING:
    from ase.atoms import Atoms


def make_iza_zeolite(code: str) -> Atoms:
    """
    Make an idealized zeolite from an IZA code, populate the atoms.info
    dictionary with the framework name, and add a labels array to the
    atoms object.

    Parameters
    ----------
    code : str
        IZA code of the zeolite to make

    Returns
    -------
    Atoms
        Atoms object of the zeolite
    """
    zeolite = get_framework(code)
    labels = site_labels(zeolite, code)
    zeolite.set_array("labels", np.array(list(labels.values())))
    zeolite.info["framework"] = code
    zeolite.info["T_info"] = get_T_info(zeolite, code)
    return zeolite


def make_all_exchanged_zeolites(
    code: str, heteroatom: str, cation: str | None = None
) -> list[Atoms]:
    """
    Enumerate all unique T sites and, for each, exchange a single Si atom with a heteroatom.
    Each is optionally charge balanced with the specified cation, and all heteratom-cation configurations
    are returned.

    Limitations:
    - Only supports monovalent cations currently.

    Parameters
    ----------
    code : str
        IZA code of the zeolite to make
    heteroatom : str
        heteroatom to substitute
    cation : str, optional
        cation to charge balance with, by default None

    Returns
    -------
    list[Atoms]
        list of all exchanged zeolites
    """

    zeolite = make_iza_zeolite(code)
    T_info = get_T_info(zeolite, code)

    zeolites = []
    for _, T_indices in T_info.items():
        T_index = T_indices[0]
        tsubbed_zeolite = tsub(zeolite, T_index, heteroatom)
        if cation:
            exchanged_zeolites, _ = monovalent(tsubbed_zeolite, T_index, cation)
            for exchanged_zeolite in exchanged_zeolites:
                zeolites.append(exchanged_zeolite)
        else:
            zeolites.append(tsubbed_zeolite)
    return get_unique_structures(zeolites)


def make_with_ratio(
    code: str,
    ratio: float,
    heteroatom: str = "Al",
    cation: str | None = None,
    max_samples: int = 50,
    min_heteroatom_distance: float | None = 3.5,
    min_interatomic_distance: float | None = 1.5,
    deduplicate: bool = True,
) -> list[Atoms]:
    """
    Make exchanged zeolites with a specified Si:heteroatom ratio. If a cation is specified,
    the zeolite will be charge balanced. Up to `max_samples` zeolites will be returned.

    Method:
    1. Enumerate all unique T labels and their corresponding sites.
    2. Randomly pick a unique T label.
    3. Randomly pick a T site from the list of T sites with that label.
    4. Substitute the T site from Step 3 with the heteroatom.
    5. If a cation is specified, charge balance the zeolite. The cation is placed randomly
    in one of the possible, adjacent adsorption sites.
    6. Repeat Steps 2-5 until the desired Si:heteroatom ratio is achieved.
    7. Repeat steps 2-6 until `max_samples` zeolites have been generated.

    Limitations:
    - Only supports monovalent cations currently.

    Parameters
    ----------
    code : str
        IZA code of the zeolite to make
    ratio : float
        Desired Si:heteroatom ratio
    heteroatom : str, optional
        Heteroatom to substitute, by default "Al"
    cation : str, optional
        Cation to charge balance with, by default None
    max_samples : int, optional
        Maximum number of zeolites to generate, by default 50
    min_heteroatom_distance : float, optional
        Minimum allowable distance between heteroatoms, by default 3.5 A.
        This can be used to prevent Al-O-Al bridges. Set to `None` if
        heteroatom-O-heteroatom bridges are not an issue.
    min_interatomic_distance : float, optional
        Minimum allowable interatomic distance between any pair of atoms,
        by default 1.5 A. This is useful to prevent cations from being too
        close to one another.
    deduplicate : bool, optional
        Whether to remove duplicate zeolites at the end, by default True

    Returns
    -------
    list[Atoms]
        list of all exchanged zeolites
    """

    # Set defaults
    if min_heteroatom_distance is None:
        min_heteroatom_distance = 0.0
    if min_interatomic_distance is None:
        min_interatomic_distance = 0.0

    # Make the Si zeolite and get T site labels/indices
    iza_zeolite = make_iza_zeolite(code)
    T_info = get_T_info(iza_zeolite, code)

    # Calculate the number of heteroatoms to add
    n_Si = len([atom for atom in iza_zeolite if atom.symbol == "Si"])
    n_heteroatoms_target = round(n_Si / (1 + ratio))

    zeolites = []
    with tqdm(total=max_samples, desc="Generating zeolites", unit="item") as pbar:
        # Generate zeolites until we have enough
        while len(zeolites) < max_samples:
            zeolite = deepcopy(iza_zeolite)
            # Add heteroatoms until we reach the target ratio
            for _ in range(n_heteroatoms_target):
                # Get only the valid T sites to consider for exchange
                T_info_valid = get_T_info_exchangeable(
                    T_info,
                    zeolite,
                    heteroatom,
                    min_heteroatom_distance=min_heteroatom_distance,
                )
                if not T_info_valid:
                    warnings.warn("Stopping early. No T sites left to exchange.")
                    return get_unique_structures(zeolites) if deduplicate else zeolites

                # Pick a random T site label and get the corresponding indices
                _, T_indices = random.choice(list(T_info_valid.items()))

                # Pick a random T site from the list of T sites with that label
                T_index = random.choice(T_indices)

                # Substitute the T site with the heteroatom
                zeolite = tsub(zeolite, T_index, heteroatom)

                # Charge balance if a cation is specified
                if cation:
                    # Get charge-balanced zeolites with the cation at various adsorption sites
                    balanced_zeolites, _ = monovalent(
                        zeolite, T_index, cation, cutoff=min_interatomic_distance
                    )
                    if not balanced_zeolites:
                        warnings.warn(
                            "Stopping early. No valid adsorption sites left for cation,"
                            "at least for this distribution of heteroatoms."
                        )
                        return (
                            get_unique_structures(zeolites) if deduplicate else zeolites
                        )

                    # Pick a random charge-balanced zeolite
                    zeolite = random.choice(balanced_zeolites)

            # Add the zeolite to the list
            if zeolite not in zeolites:
                zeolites.append(zeolite)
                pbar.update(1)

    return get_unique_structures(zeolites) if deduplicate else zeolites
