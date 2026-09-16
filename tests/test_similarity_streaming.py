from pathlib import Path
import numpy as np
import pytest
from parallel_rdkit import cross_similarity


def test_npy_output_and_mmap(tmp_path):
    path = tmp_path / "cross.data"
    result = cross_similarity(["CCO", "bad"], ["CCO", "c1ccccc1"],
                              output_path=path, batch_size=1, tile_size=1)
    assert result.output_path == path
    loaded = np.load(path, mmap_mode="r")
    assert loaded.shape == (2, 2) and loaded.dtype == np.float32
    assert np.isnan(loaded[1]).all()


def test_noclobber(tmp_path):
    path = tmp_path / "cross.npy"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        cross_similarity(["CCO"], ["CCO"], output_path=path)
    assert path.read_bytes() == b"original"


def test_overwrite_is_boolean_and_successful_overwrite(tmp_path):
    path = tmp_path / "cross.npy"
    path.write_bytes(b"original")
    with pytest.raises(TypeError):
        cross_similarity(["CCO"], ["CCO"], output_path=path, overwrite="yes")
    assert path.read_bytes() == b"original"
    cross_similarity(["CCO"], ["CCN"], output_path=path, overwrite=True)
    loaded = np.load(path)
    assert loaded.shape == (1, 1) and loaded.dtype == np.float32


def test_streaming_empty_left_preserves_right_mask(tmp_path):
    path = tmp_path / "empty-left.npy"
    result = cross_similarity([], ["CCO"], output_path=path)
    assert result.left_valid.tolist() == []
    assert result.right_valid.tolist() == [True]
    assert np.load(path).shape == (0, 1)


def test_streaming_partial_tiles_match_dense(tmp_path):
    left = ["CCO", "CCN", "c1ccccc1", "CCCl", "bad"]
    right = ["CCO", "CCN", "c1ccccc1", "CCCl", "CCC", "bad", "C"]
    expected = cross_similarity(left, right, batch_size=8, tile_size=3).dense
    path = tmp_path / "partial.data"
    cross_similarity(left, right, output_path=path, batch_size=8, tile_size=3)
    np.testing.assert_equal(np.load(path), expected)


def test_streaming_failure_leaves_destination_and_no_temp(tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"parent")
    with pytest.raises(OSError):
        cross_similarity(["CCO"], ["CCN"], output_path=parent / "out.npy")
    assert parent.read_bytes() == b"parent"
    assert not list(tmp_path.glob(".not-a-directory.*.tmp"))
