import pytest
from parallel_rdkit import msready_smiles, msready_inchi_inchikey_parallel

def test_msready_basic():
    assert msready_smiles("C") == "C"
    assert msready_smiles("CC") == "CC"

def test_msready_metal():
    # Sodium acetate should become acetic acid
    assert msready_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    # Potassium methoxide should become methanol
    assert msready_smiles("C[O-].[K+]") == "CO"

def test_msready_hcl_salt():
    # Methylamine hydrochloride should become methylamine
    assert msready_smiles("CN.Cl") == "CN"
    # Pyridine hydrochloride should become pyridine
    assert msready_smiles("c1ccncc1.Cl") == "c1ccncc1"

def test_msready_charge():
    # Phenolate should become phenol
    assert msready_smiles("[O-]c1ccccc1") == "Oc1ccccc1"
    # Ammonium should become amine
    assert msready_smiles("[NH4+]") == "N"

def test_msready_tautomer():
    # Imidazole tautomers should canonicalize to the same thing
    s1 = "C1=CNC=N1"
    s2 = "c1nc[nH]c1"
    assert msready_smiles(s1) == msready_smiles(s2)

def test_msready_complex():
    # Magnesium acetate
    assert msready_smiles("CC(=O)[O-].CC(=O)[O-].[Mg+2]") == "CC(=O)O"
    # Aniline sulfate
    assert msready_smiles("c1ccccc1N.c1ccccc1N.O=S(=O)(O)O") == "Nc1ccccc1"
    # Sodium phenolate (alternate SMILES)
    assert msready_smiles("c1ccccc1O[Na]") == "Oc1ccccc1"
    # Ferrous chloride in ethanol
    assert msready_smiles("[Fe+2].[Cl-].[Cl-].CCO") == "CCO"

def test_parallel_multi():
    smiles = ["C", "CC", "CC(=O)[O-].[Na+]", "CN.Cl", "[O-]c1ccccc1"]
    results = msready_inchi_inchikey_parallel(smiles)
    
    assert len(results) == 5
    # Check MS-Ready SMILES
    assert results[0][0] == "C"
    assert results[1][0] == "CC"
    assert results[2][0] == "CC(=O)O"
    assert results[3][0] == "CN"
    assert results[4][0] == "Oc1ccccc1"
    
    # Check that InChI and InChIKey are present (non-empty)
    for res in results:
        assert res[1].startswith("InChI=")
        assert len(res[2]) == 27 # Standard InChIKey length
