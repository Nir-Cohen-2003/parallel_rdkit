import pytest
import numpy as np
from parallel_rdkit import FingerprintParams, get_fp_list
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit import DataStructs

def test_morgan_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='morgan', radius=2, fpSize=2048)
    
    # Get FPs using the new backend
    fps = get_fp_list(smiles, params)
    
    assert len(fps) == 3
    assert fps[0].shape == (2048,)
    
    # Verify against RDKit directly
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        expected_fp = gen.GetFingerprint(mol)
        expected_np = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(expected_fp, expected_np)
        
        np.testing.assert_array_almost_equal(fp, expected_np)

def test_rdkit_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='rdkit', minPath=1, maxPath=7, fpSize=2048)
    
    fps = get_fp_list(smiles, params)
    
    assert len(fps) == 3
    assert fps[0].shape == (2048,)
    
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=2048)
        expected_fp = gen.GetFingerprint(mol)
        expected_np = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(expected_fp, expected_np)
        
        np.testing.assert_array_almost_equal(fp, expected_np)

def test_maccs_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='maccs')
    
    fps = get_fp_list(smiles, params)
    
    assert len(fps) == 3
    assert fps[0].shape == (167,)
    
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        expected_fp = MACCSkeys.GenMACCSKeys(mol)
        expected_np = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(expected_fp, expected_np)
        
        np.testing.assert_array_almost_equal(fp, expected_np)

def test_atompair_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='atompair', fpSize=2048)
    
    fps = get_fp_list(smiles, params)
    
    assert len(fps) == 3
    assert fps[0].shape == (2048,)
    
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        expected_fp = gen.GetFingerprint(mol)
        expected_np = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(expected_fp, expected_np)
        
        np.testing.assert_array_almost_equal(fp, expected_np)

def test_torsion_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='torsion', fpSize=2048)
    
    fps = get_fp_list(smiles, params)
    
    assert len(fps) == 3
    assert fps[0].shape == (2048,)
    
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
        expected_fp = gen.GetFingerprint(mol)
        expected_np = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(expected_fp, expected_np)
        
        np.testing.assert_array_almost_equal(fp, expected_np)

def test_count_fp():
    smiles = ["CCO", "CCN", "CCC"]
    params = FingerprintParams(fp_type='morgan', radius=2, fpSize=2048, fp_method='GetCountFingerprint')
    
    fps = get_fp_list(smiles, params)
    
    for s, fp in zip(smiles, fps):
        mol = Chem.MolFromSmiles(s)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        expected_fp = gen.GetCountFingerprint(mol)
        expected_np = np.zeros((2048,), dtype=np.float32)
        for bit_id, count in expected_fp.GetNonzeroElements().items():
            expected_np[bit_id % 2048] = count
        
        np.testing.assert_array_almost_equal(fp, expected_np)
