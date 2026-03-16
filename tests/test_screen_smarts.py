import pytest
import numpy as np
from pathlib import Path
from parallel_rdkit import screen_smarts


@pytest.fixture
def sample_smiles_file(tmp_path):
    """Create a temporary SMILES file for testing."""
    smiles_file = tmp_path / "test_molecules.smi"
    smiles = [
        "CCO",  # ethanol - has OH
        "CCN",  # ethylamine - has NH2
        "c1ccccc1",  # benzene - has aromatic ring
        "CC(=O)O",  # acetic acid - has OH and C=O
        "CC(C)O",  # isopropanol - has OH
    ]
    smiles_file.write_text("\n".join(smiles))
    return str(smiles_file)


@pytest.fixture
def sample_smarts():
    """SMARTS patterns for testing."""
    return [
        "[OH]",  # hydroxyl group
        "[NH2]",  # primary amine
        "c1ccccc1",  # benzene ring
    ]


def test_screen_smarts_direct(sample_smiles_file, sample_smarts):
    """Test direct mode returns correct N x M boolean matrix."""
    result = screen_smarts(
        smarts_list=sample_smarts,
        smiles_file=sample_smiles_file,
        mode="direct",
    )
    
    # Should return numpy array
    assert isinstance(result, np.ndarray)
    # Shape should be (5 molecules, 3 SMARTS patterns)
    assert result.shape == (5, 3)
    # Should be boolean array
    assert result.dtype == bool
    
    # Check specific matches:
    # Row 0: CCO - should match [OH] only
    assert result[0, 0] == True  # [OH]
    assert result[0, 1] == False  # [NH2]
    assert result[0, 2] == False  # benzene
    
    # Row 1: CCN - should match [NH2] only
    assert result[1, 0] == False  # [OH]
    assert result[1, 1] == True  # [NH2]
    assert result[1, 2] == False  # benzene
    
    # Row 2: benzene - should match benzene only
    assert result[2, 0] == False  # [OH]
    assert result[2, 1] == False  # [NH2]
    assert result[2, 2] == True  # benzene
    
    # Row 3: acetic acid - should match [OH] only (not NH2 or benzene)
    assert result[3, 0] == True  # [OH]
    assert result[3, 1] == False  # [NH2]
    assert result[3, 2] == False  # benzene
    
    # Row 4: isopropanol - should match [OH] only
    assert result[4, 0] == True  # [OH]
    assert result[4, 1] == False  # [NH2]
    assert result[4, 2] == False  # benzene


def test_screen_smarts_with_cache(sample_smiles_file, sample_smarts, tmp_path):
    """Test caching functionality."""
    cache_path = tmp_path / "test_cache.bin"
    
    # First call - should compute and cache
    result1 = screen_smarts(
        smarts_list=sample_smarts,
        smiles_file=sample_smiles_file,
        mode="direct",
        cache_path=str(cache_path),
    )
    
    assert cache_path.exists(), "Cache file should be created"
    
    # Second call - should use cache
    result2 = screen_smarts(
        smarts_list=sample_smarts,
        smiles_file=sample_smiles_file,
        mode="direct",
        cache_path=str(cache_path),
    )
    
    # Results should be identical
    np.testing.assert_array_equal(result1, result2)


def test_screen_smarts_streaming(sample_smiles_file, sample_smarts, tmp_path):
    """Test streaming mode writes to .npy file."""
    output_path = tmp_path / "output.npy"
    
    count = screen_smarts(
        smarts_list=sample_smarts,
        smiles_file=sample_smiles_file,
        mode="streaming",
        batch_size=2,  # Small batch to test batching
        output_path=str(output_path),
    )
    
    # Should process 5 molecules
    assert count == 5
    
    # Output file should exist
    assert output_path.exists()
    
    # Load and verify the numpy array
    result = np.load(output_path)
    assert result.shape == (5, 3)
    assert result.dtype == bool


def test_screen_smarts_streaming_requires_output_path(sample_smiles_file, sample_smarts):
    """Test that streaming mode requires output_path."""
    with pytest.raises(ValueError, match="output_path is required"):
        screen_smarts(
            smarts_list=sample_smarts,
            smiles_file=sample_smiles_file,
            mode="streaming",
        )


def test_screen_smarts_invalid_mode(sample_smiles_file, sample_smarts):
    """Test that invalid mode raises error."""
    with pytest.raises(ValueError, match="mode must be 'direct' or 'streaming'"):
        screen_smarts(
            smarts_list=sample_smarts,
            smiles_file=sample_smiles_file,
            mode="invalid",
        )


def test_screen_smarts_empty_smarts(sample_smiles_file):
    """Test with empty SMARTS list."""
    result = screen_smarts(
        smarts_list=[],
        smiles_file=sample_smiles_file,
        mode="direct",
    )
    
    # Should have 5 rows and 0 columns
    assert result.shape == (5, 0)


def test_screen_smarts_single_pattern(sample_smiles_file):
    """Test with single SMARTS pattern."""
    result = screen_smarts(
        smarts_list=["[OH]"],  # Only hydroxyl
        smiles_file=sample_smiles_file,
        mode="direct",
    )
    
    assert result.shape == (5, 1)
    # CCO, CC(=O)O, CC(C)O should match (indices 0, 3, 4)
    assert result[0, 0] == True
    assert result[1, 0] == False  # CCN
    assert result[2, 0] == False  # benzene
    assert result[3, 0] == True
    assert result[4, 0] == True
