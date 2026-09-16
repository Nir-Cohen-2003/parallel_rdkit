import numpy as np
import pytest

from parallel_rdkit import FingerprintParams, SimilarityResult, cross_similarity
from parallel_rdkit.parallel_rdkit_backend import (
    FingerprintOptions, cross_similarity_coo_fill_into,
    cross_similarity_dense_into,
)


def test_rectangular_masks_and_nan():
    result = cross_similarity(["CCO", "not-a-smiles", "CCO"], ["CCO", "c1ccccc1"])
    assert isinstance(result, SimilarityResult)
    assert result.shape == (3, 2)
    assert result.dense.dtype == np.float32 and result.dense.flags.c_contiguous
    assert result.left_valid.tolist() == [True, False, True]
    assert np.isnan(result.dense[1]).all()
    assert result.dense[0, 0] == np.float32(1.0)


def test_output_buffers_survive_result_collection():
    result = cross_similarity(["CCO"], ["CCO"], batch_size=1, tile_size=1)
    dense = result.dense
    del result
    import gc; gc.collect()
    assert dense[0, 0] == np.float32(1.0)


def test_threshold_is_sorted_and_inclusive():
    result = cross_similarity(["CCO", "CCO"], ["CCO", "c1ccccc1"], threshold=1.0,
                              batch_size=1, tile_size=1)
    rows, cols, values = result.coo
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [0, 0]
    assert values.dtype == np.float32


def test_empty_axes_and_zero_threshold():
    result = cross_similarity([], ["CCO"], threshold=0.0)
    assert result.shape == (0, 1)
    assert all(a.size == 0 for a in result.coo)
    assert result.right_valid.tolist() == [True]


@pytest.mark.parametrize("bits", [1024, 2048, 4096])
def test_specialized_word_lengths(bits):
    result = cross_similarity(["CCO"], ["CCO", "c1ccccc1"],
                              fp_params=FingerprintParams(fpSize=bits),
                              batch_size=1, tile_size=1)
    assert result.dense.shape == (1, 2)
    assert result.dense[0, 0] == np.float32(1.0)


def test_count_methods_are_rejected():
    with pytest.raises(ValueError):
        cross_similarity(["CCO"], ["CCO"], fp_params=FingerprintParams(fp_method="GetCountFingerprint"))


def test_native_coo_fill_requires_exact_capacity_and_valid_threshold():
    opts = FingerprintOptions()
    for capacity in (0, 2):
        rows = np.empty(capacity, dtype=np.int64)
        columns = np.empty(capacity, dtype=np.int64)
        values = np.empty(capacity, dtype=np.float32)
        with pytest.raises(ValueError, match="exactly"):
            cross_similarity_coo_fill_into(
                ["CCO"], ["CCO"], opts, False, 0.0, 4096, 256,
                rows, columns, values)
    for threshold in (-0.01, 1.01, np.nan, np.inf):
        rows = np.empty(1, dtype=np.int64)
        columns = np.empty(1, dtype=np.int64)
        values = np.empty(1, dtype=np.float32)
        with pytest.raises(ValueError, match="threshold"):
            cross_similarity_coo_fill_into(
                ["CCO"], ["CCO"], opts, False, threshold, 4096, 256,
                rows, columns, values)


def test_native_dense_into_writes_caller_owned_buffer():
    out = np.full((2, 2), -7.0, dtype=np.float32)
    left_valid = np.empty(2, dtype=np.bool_)
    right_valid = np.empty(2, dtype=np.bool_)
    cross_similarity_dense_into(
        ["CCO", "bad"], ["CCO", "CCN"], FingerprintOptions(), False,
        2, 1, out, left_valid, right_valid)
    assert out[0, 0] == np.float32(1.0)
    assert np.isnan(out[1]).all()
    assert left_valid.tolist() == [True, False]


def test_float32_threshold_boundary_is_inclusive():
    dense = cross_similarity(["CCO"], ["CCN"]).dense
    threshold = float(dense[0, 0])
    result = cross_similarity(["CCO"], ["CCN"], threshold=threshold)
    assert result.coo[0].tolist() == [0]
    assert result.coo[2][0] == dense[0, 0]


@pytest.mark.parametrize("field,value", [
    ("fp_type", 1), ("fp_method", 1), ("fpSize", 0),
    ("fpSize", True), ("radius", -1), ("radius", True),
    ("minPath", 0), ("maxPath", 0), ("minDistance", 0),
    ("maxDistance", np.array(30)), ("targetSize", 0),
    ("useBondTypes", "yes"), ("use2D", 1),
    ("includeChirality", None), ("countSimulation", None),
])
def test_fingerprint_parameters_are_validated(field, value):
    params = FingerprintParams()
    setattr(params, field, value)
    with pytest.raises((TypeError, ValueError)):
        cross_similarity(["CCO"], ["CCN"], fp_params=params)


def test_fingerprint_parameter_bounds_are_consistent():
    params = FingerprintParams(minPath=6, maxPath=5)
    with pytest.raises(ValueError, match="minPath"):
        cross_similarity(["CCO"], ["CCN"], fp_params=params)
    params = FingerprintParams(minDistance=8, maxDistance=7)
    with pytest.raises(ValueError, match="minDistance"):
        cross_similarity(["CCO"], ["CCN"], fp_params=params)


def test_rdkit_score_matches_independent_reference():
    from rdkit import Chem, DataStructs
    left = Chem.MolFromSmiles("CCO")
    right = Chem.MolFromSmiles("CCN")
    result = cross_similarity(["CCO"], ["CCN"],
                              fp_params=FingerprintParams(fp_type="rdkit"))
    expected = np.float32(DataStructs.TanimotoSimilarity(
        Chem.RDKFingerprint(left, minPath=1, maxPath=7, fpSize=2048),
        Chem.RDKFingerprint(right, minPath=1, maxPath=7, fpSize=2048)))
    assert result.dense[0, 0] == expected


@pytest.mark.parametrize("family", ["atompair", "torsion"])
def test_hashed_family_scores_match_rdkit_reference(family):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    left = Chem.MolFromSmiles("CCO")
    right = Chem.MolFromSmiles("CCN")
    params = FingerprintParams(fp_type=family, fpSize=2048)
    result = cross_similarity(["CCO"], ["CCN"], fp_params=params)
    if family == "atompair":
        make = lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol, nBits=2048, minLength=1, maxLength=30, includeChirality=False)
    else:
        make = lambda mol: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol, nBits=2048, targetSize=4, includeChirality=False)
    expected = np.float32(DataStructs.TanimotoSimilarity(make(left), make(right)))
    assert result.dense[0, 0] == expected


def test_specialized_morgan_score_matches_rdkit_reference():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    params = FingerprintParams(fpSize=1024, radius=2)
    result = cross_similarity(["CCO"], ["CCN"], fp_params=params)
    left = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CCO"), 2, nBits=1024)
    right = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CCN"), 2, nBits=1024)
    expected = np.float32(DataStructs.TanimotoSimilarity(left, right))
    assert result.dense[0, 0] == expected
