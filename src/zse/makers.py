from __future__ import annotations

import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from tqdm import tqdm

from zse.cation import monovalent
from zse.collections.framework import get_framework
from zse.substitute import tsub
from zse.t_utilities import get_T_info
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
    deduplicate : bool, optional
        Whether to remove duplicate zeolites at the end, by default True

    Returns
    -------
    list[Atoms]
        list of all exchanged zeolites
    """

    iza_zeolite = make_iza_zeolite(code)
    T_info = get_T_info(iza_zeolite, code)
    n_Si = len([atom for atom in iza_zeolite if atom.symbol == "Si"])
    n_heteroatoms_target = round(n_Si / ratio)

    zeolites = []
    with tqdm(total=max_samples, desc="Generating zeolites:", unit="item") as pbar:
        while len(zeolites) < max_samples:
            zeolite = deepcopy(iza_zeolite)
            n_heteroatoms = 0
            while n_heteroatoms < n_heteroatoms_target:
                _, T_indices = random.choice(list(T_info.items()))
                T_index = random.choice(T_indices)
                zeolite = tsub(zeolite, T_index, heteroatom)
                if cation:
                    zeolite = random.choice(monovalent(zeolite, T_index, cation)[0])
                n_heteroatoms = len(
                    [atom for atom in zeolite if atom.symbol == heteroatom]
                )

            if zeolite not in zeolites:
                zeolites.append(zeolite)
            pbar.update(1)

    if deduplicate:
        return get_unique_structures(zeolites)
    else:
        return zeolites
