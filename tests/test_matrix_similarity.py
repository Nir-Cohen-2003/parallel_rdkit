"""
Tests for matrix_similarity module - GPU-accelerated similarity calculation.

These tests require nvmolkit and GPU access. Run with:
    pixi run -e gpu test-matrix-similarity
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit import DataStructs

# Skip all tests if nvmolkit is not available
pytest.importorskip("nvmolkit", reason="nvmolkit not available")

from parallel_rdkit.fingerprint import FingerprintParams
from parallel_rdkit.matrix_similarity import (
    calculate_similarity_matrix,
    calculate_similarity_matrix_streaming,
    calculate_tanimoto_matrix,
    calculate_cosine_matrix,
    calculate_tanimoto_matrix_streaming,
    calculate_cosine_matrix_streaming,
    butina_split,
)


# Test data
SIMPLE_SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CCN"]


def test_tanimoto_similarity_matrix():
    """Test basic tanimoto similarity calculation."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            similarity_metric="tanimoto",
        )
        
        assert parquet_path.exists()
        
        # Read and verify parquet content
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # Should have columns: mol1_idx, mol2_idx, tanimoto
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "tanimoto"]
        
        # Should only have lower triangular (mol1_idx <= mol2_idx)
        assert (df["mol1_idx"] <= df["mol2_idx"]).all()
        
        # For N=3, lower triangular has N*(N+1)/2 = 6 pairs
        assert len(df) == 6
        
        # All similarities should be between 0 and 1
        assert (df["tanimoto"] >= 0.0).all()
        assert (df["tanimoto"] <= 1.0).all()
        
        # Diagonal entries (self-similarity) should be 1.0
        diagonal = df[df["mol1_idx"] == df["mol2_idx"]]
        assert len(diagonal) == 3
        np.testing.assert_array_almost_equal(diagonal["tanimoto"].values, [1.0] * 3)


def test_cosine_similarity_matrix():
    """Test basic cosine similarity calculation."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            similarity_metric="cosine",
        )
        
        assert parquet_path.exists()
        
        # Read and verify parquet content
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # Should have columns: mol1_idx, mol2_idx, cosine
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "cosine"]
        
        # All similarities should be between 0 and 1
        assert (df["cosine"] >= 0.0).all()
        assert (df["cosine"] <= 1.0).all()
        
        # Diagonal entries (self-similarity) should be 1.0
        diagonal = df[df["mol1_idx"] == df["mol2_idx"]]
        assert len(diagonal) == 3
        np.testing.assert_array_almost_equal(diagonal["cosine"].values, [1.0] * 3)


def test_both_similarities_matrix():
    """Test calculation of both tanimoto and cosine similarities."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            similarity_metric="both",
        )
        
        assert parquet_path.exists()
        
        # Read and verify parquet content
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # Should have columns: mol1_idx, mol2_idx, tanimoto, cosine
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "tanimoto", "cosine"]
        
        # For N=3, lower triangular has N*(N+1)/2 = 6 pairs
        assert len(df) == 6
        
        # Both similarities should be between 0 and 1
        assert (df["tanimoto"] >= 0.0).all()
        assert (df["tanimoto"] <= 1.0).all()
        assert (df["cosine"] >= 0.0).all()
        assert (df["cosine"] <= 1.0).all()


def test_backward_compatibility_tanimoto():
    """Test backward compatibility alias for tanimoto."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_tanimoto_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
        )
        
        assert parquet_path.exists()
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "tanimoto"]


def test_backward_compatibility_cosine():
    """Test backward compatibility alias for cosine."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_cosine_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
        )
        
        assert parquet_path.exists()
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "cosine"]


def test_similarity_matrix_with_threshold():
    """Test parquet mode with similarity threshold filtering."""
    smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CCN"]
    indices = np.arange(len(smiles), dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            similarity_metric="tanimoto",
            threshold=0.5,
        )
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # All similarities should be >= 0.5
        assert (df["tanimoto"] >= 0.5).all()
        
        # Should have fewer entries than without threshold
        assert len(df) < 15


def test_similarity_matrix_with_invalid_smiles():
    """Test handling of invalid SMILES - they should be filtered out."""
    smiles = ["CCO", "INVALID", "CCCO", "", "CCCCO"]
    indices = np.array([100, 200, 300, 400, 500], dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        log_path = Path(tmpdir) / "output.log"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            log_path=log_path,
        )
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # Should only have 3 valid molecules
        # Indices 100, 300, 500 should remain
        unique_indices = set(df["mol1_idx"].unique()) | set(df["mol2_idx"].unique())
        assert unique_indices == {100, 300, 500}
        
        # Should have 3*(3+1)/2 = 6 pairs
        assert len(df) == 6
        
        # Check log mentions invalid molecules
        log_content = log_path.read_text()
        assert "invalid" in log_content.lower() or "Warning" in log_content


def test_similarity_matrix_streaming_from_parquet():
    """Test streaming function that reads from input parquet."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input parquet with SMILES and indices
        input_parquet = Path(tmpdir) / "input.parquet"
        df_input = pl.DataFrame({
            "smiles": smiles,
            "index": indices,
            "extra_col": ["a", "b", "c"]  # Should be ignored
        })
        df_input.write_parquet(str(input_parquet))
        
        output_parquet = Path(tmpdir) / "output.parquet"
        
        calculate_similarity_matrix_streaming(
            parquet_path=input_parquet,
            output_parquet_path=output_parquet,
            smiles_column="smiles",
            index_column="index",
            similarity_metric="tanimoto",
        )
        
        assert output_parquet.exists()
        
        # Verify output
        table = pq.read_table(str(output_parquet))
        df_output = table.to_pandas()
        
        # Should use the caller-provided indices
        unique_indices = set(df_output["mol1_idx"].unique()) | set(df_output["mol2_idx"].unique())
        assert unique_indices == {10, 20, 30}
        
        # Should have 6 pairs (lower triangular of 3x3)
        assert len(df_output) == 6


def test_similarity_matrix_default_params():
    """Test that default fingerprint parameters work correctly."""
    smiles = ["CCO", "CCCO"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        # Should work without specifying fp_params
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
        )
        
        assert parquet_path.exists()
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        assert len(df) == 3  # Lower triangular of 2x2


def test_similarity_matrix_invalid_metric():
    """Test that invalid similarity_metric raises error."""
    smiles = ["CCO", "CCCO"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        with pytest.raises(ValueError, match="similarity_metric must be"):
            calculate_similarity_matrix(
                smiles=smiles,
                parquet_path=parquet_path,
                similarity_metric="invalid_metric",
            )


def test_butina_split():
    """Test Butina clustering on similarity matrix."""
    # Create a simple similarity matrix
    # Molecules 0 and 1 are similar, molecule 2 is different
    sim_matrix = np.array([
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ], dtype=np.float32)
    
    # With threshold 0.3 (distance 0.7), molecules 0 and 1 should be in same cluster
    labels = butina_split(sim_matrix, dist_threshold=0.3)
    
    assert len(labels) == 3
    # Molecules 0 and 1 should have same cluster ID
    assert labels[0] == labels[1]
    # Molecule 2 should be in different cluster
    assert labels[2] != labels[0]


def test_memory_usage_fraction():
    """Test that memory_usage_fraction parameter works."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        log_path = Path(tmpdir) / "output.log"
        
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            memory_usage_fraction=0.3,
            log_path=log_path,
        )
        
        assert parquet_path.exists()
        
        # Check log mentions memory fraction
        log_content = log_path.read_text()
        assert "memory_fraction=0.3" in log_content or "0.3" in log_content


@pytest.mark.skip(reason="Long-running test for large datasets")
def test_similarity_matrix_large_dataset():
    """Test with larger dataset to verify memory efficiency.
    
    This test requires a large dataset and is skipped by default.
    Run manually with: pytest tests/test_matrix_similarity.py::test_similarity_matrix_large_dataset -v
    """
    # Generate many similar molecules
    n_mols = 10000
    smiles = [f"C" * (i % 20 + 1) for i in range(n_mols)]
    indices = np.arange(n_mols, dtype=np.uint32)
    
    fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "output.parquet"
        
        # This should complete without OOM
        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=parquet_path,
            indices=indices,
            fp_params=fp_params,
            similarity_metric="tanimoto",
            memory_usage_fraction=0.5,
            threshold=0.5,
        )
        
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        
        # Verify we got results
        assert len(df) > 0
        print(f"Generated {len(df)} similarity pairs for {n_mols} molecules")


def test_backward_compatibility_streaming_tanimoto():
    """Test backward compatibility streaming alias for tanimoto."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input parquet
        input_parquet = Path(tmpdir) / "input.parquet"
        df_input = pl.DataFrame({
            "smiles": smiles,
            "index": indices,
        })
        df_input.write_parquet(str(input_parquet))
        
        output_parquet = Path(tmpdir) / "output.parquet"
        
        calculate_tanimoto_matrix_streaming(
            parquet_path=input_parquet,
            output_parquet_path=output_parquet,
            smiles_column="smiles",
            index_column="index",
        )
        
        assert output_parquet.exists()
        
        table = pq.read_table(str(output_parquet))
        df = table.to_pandas()
        
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "tanimoto"]


def test_backward_compatibility_streaming_cosine():
    """Test backward compatibility streaming alias for cosine."""
    smiles = ["CCO", "CCCO", "CCCCO"]
    indices = np.array([10, 20, 30], dtype=np.uint32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input parquet
        input_parquet = Path(tmpdir) / "input.parquet"
        df_input = pl.DataFrame({
            "smiles": smiles,
            "index": indices,
        })
        df_input.write_parquet(str(input_parquet))
        
        output_parquet = Path(tmpdir) / "output.parquet"
        
        calculate_cosine_matrix_streaming(
            parquet_path=input_parquet,
            output_parquet_path=output_parquet,
            smiles_column="smiles",
            index_column="index",
        )
        
        assert output_parquet.exists()
        
        table = pq.read_table(str(output_parquet))
        df = table.to_pandas()
        
        assert list(df.columns) == ["mol1_idx", "mol2_idx", "cosine"]
