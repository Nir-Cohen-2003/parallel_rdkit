import numpy as np
import polars as pl

from parallel_rdkit import smiles_to_formula


# Element order: ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]
FORMULA_ELEMENT_ORDER = ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]


def _expected(formula_dict):
    return np.array([formula_dict.get(el, 0) for el in FORMULA_ELEMENT_ORDER], dtype=np.int64)


def test_smiles_to_formula_list():
    smiles = ["CCO", "CF", "[Na+].[Cl-]", "not_a_smiles"]
    result = smiles_to_formula(smiles)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 12)
    assert result.dtype == np.int64

    expected = np.stack([
        _expected({"C": 2, "H": 6, "O": 1}),
        _expected({"C": 1, "H": 3, "F": 1}),
        _expected({"Na": 1, "Cl": 1}),
        np.zeros(12, dtype=np.int64),
    ])
    np.testing.assert_array_equal(result, expected)


def test_smiles_to_formula_iterable():
    result = smiles_to_formula(iter(["CCO", "CF"]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 12)


def test_smiles_to_formula_polars():
    smiles = pl.Series(["CCO", "CF", "[Na+].[Cl-]", "not_a_smiles"])
    result = smiles_to_formula(smiles)

    assert isinstance(result, pl.Series)
    assert result.dtype == pl.Array(pl.Int64, 12)
    assert result.len() == 4

    expected = np.stack([
        _expected({"C": 2, "H": 6, "O": 1}),
        _expected({"C": 1, "H": 3, "F": 1}),
        _expected({"Na": 1, "Cl": 1}),
        np.zeros(12, dtype=np.int64),
    ])
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_smiles_to_formula_various_elements():
    # Ethanol
    np.testing.assert_array_equal(
        smiles_to_formula(["CCO"])[0],
        _expected({"C": 2, "H": 6, "O": 1}),
    )
    # Bromotrichloromethane
    np.testing.assert_array_equal(
        smiles_to_formula(["C(Cl)(Cl)(Cl)Br"])[0],
        _expected({"C": 1, "Cl": 3, "Br": 1}),
    )
    # Cysteine
    np.testing.assert_array_equal(
        smiles_to_formula(["C([C@@H](C(=O)O)N)S"])[0],
        _expected({"C": 3, "H": 7, "N": 1, "O": 2, "S": 1}),
    )
    # ATP-like
    np.testing.assert_array_equal(
        smiles_to_formula(["C1=NC2=C(C(=N1)N)N=CN2[C@@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O"])[0],
        _expected({"C": 10, "H": 16, "N": 5, "O": 13, "P": 3}),
    )
