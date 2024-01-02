from pathlib import Path

from ase.io import read
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zse.makers import make_all_exchanged_zeolites, make_iza_zeolite, make_with_ratio

REF_DATA = Path(__file__).parent / Path("data")


def test_all_silica():
    atoms = make_iza_zeolite("CHA")
    ref_atoms = read(f"{REF_DATA}/CHA.cif")
    assert_array_equal(atoms.get_chemical_symbols(), ref_atoms.get_chemical_symbols())
    assert_array_almost_equal(atoms.get_positions(), ref_atoms.get_positions())
    assert_array_almost_equal(atoms.get_cell(), ref_atoms.get_cell())


def test_exchange_unique_cha():
    exchanged_zeolites = make_all_exchanged_zeolites("CHA", "B", "Na")
    assert len(exchanged_zeolites) == 4

    atoms = exchanged_zeolites[3]
    ref_atoms = read(f"{REF_DATA}/MOR_B_Na_3.cif")
    assert_array_equal(atoms.get_chemical_symbols(), ref_atoms.get_chemical_symbols())
    assert_array_almost_equal(atoms.get_positions(), ref_atoms.get_positions())
    assert_array_almost_equal(atoms.get_cell(), ref_atoms.get_cell())


def test_exchange_unique_mor():
    exchanged_zeolites = make_all_exchanged_zeolites("MOR", "B", "Na")
    assert len(exchanged_zeolites) == 26


def test_make_with_ratio():
    exchanged_zeolites = make_with_ratio(
        "CHA", 10.0, heteroatom="B", cation="Na", max_samples=2, deduplicate=False
    )
    assert len(exchanged_zeolites) == 2
    assert len([atom for atom in exchanged_zeolites[0] if atom.symbol == "Na"]) == 4
    assert len([atom for atom in exchanged_zeolites[0] if atom.symbol == "B"]) == 4
    assert len([atom for atom in exchanged_zeolites[-1] if atom.symbol == "Na"]) == 4
    assert len([atom for atom in exchanged_zeolites[-1] if atom.symbol == "B"]) == 4


def test_make_with_ratio2():
    exchanged_zeolites = make_with_ratio(
        "CHA",
        4.0,
        heteroatom="B",
        cation="Na",
        max_samples=2,
        deduplicate=False,
        min_heteroatom_distance=None,
        min_cation_distance=None,
    )
    assert len(exchanged_zeolites) == 2
