"""Rectangular Tanimoto similarity for two SMILES libraries.

Invalid input positions are retained.  In dense results every pair involving an
invalid molecule is ``NaN``; use ``left_valid`` and ``right_valid`` masks (and
NaN-aware reductions) rather than interpreting those entries as zero.
"""
from __future__ import annotations

import errno
import math
import numbers
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .fingerprint import (FingerprintParams, get_fp_list,
                          _validate_fingerprint_params)


class SimilarityResult:
    """Result of :func:`cross_similarity`.

    Exactly one of ``dense``, ``coo``, and ``output_path`` is populated.  COO
    coordinates are int64 and row-major; values are float32.  ``output_path``
    points at a completed C-order NumPy ``.npy`` file.
    """
    __slots__ = ("shape", "left_valid", "right_valid", "dense", "coo", "output_path")

    def __init__(self, shape, left_valid, right_valid, *, dense=None, coo=None,
                 output_path=None):
        if not isinstance(shape, tuple) or len(shape) != 2 or any(
                isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
                for x in shape):
            raise TypeError("shape must be a pair of nonnegative integers")
        self.shape = (int(shape[0]), int(shape[1]))
        if not isinstance(left_valid, np.ndarray) or left_valid.dtype != np.bool_ or left_valid.ndim != 1:
            raise TypeError("left_valid must be a one-dimensional bool ndarray")
        if not isinstance(right_valid, np.ndarray) or right_valid.dtype != np.bool_ or right_valid.ndim != 1:
            raise TypeError("right_valid must be a one-dimensional bool ndarray")
        if left_valid.shape != (self.shape[0],) or right_valid.shape != (self.shape[1],):
            raise ValueError("validity masks do not match result shape")
        self.left_valid, self.right_valid = left_valid, right_valid
        payloads = int(dense is not None) + int(coo is not None) + int(output_path is not None)
        if payloads != 1:
            raise ValueError("exactly one result payload is required")
        self.dense = self.coo = self.output_path = None
        if dense is not None:
            if not isinstance(dense, np.ndarray) or dense.dtype != np.float32 or dense.shape != self.shape or not dense.flags.c_contiguous:
                raise TypeError("dense payload must be a C-contiguous float32 ndarray")
            self.dense = dense
        elif coo is not None:
            if not isinstance(coo, tuple) or len(coo) != 3:
                raise TypeError("coo must be (rows, columns, values)")
            rows, cols, values = coo
            if any(not isinstance(a, np.ndarray) for a in coo):
                raise TypeError("COO arrays must be ndarrays")
            if rows.dtype != np.int64 or cols.dtype != np.int64 or values.dtype != np.float32:
                raise TypeError("COO dtypes must be int64, int64, float32")
            if rows.ndim != 1 or cols.ndim != 1 or values.ndim != 1 or not (rows.size == cols.size == values.size):
                raise ValueError("COO arrays must be equal-length one-dimensional arrays")
            self.coo = coo
        else:
            self.output_path = Path(output_path)


def _materialize(values, name):
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of SMILES, not a string")
    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of SMILES") from exc
    if any(not isinstance(s, str) for s in result):
        raise TypeError(f"{name} must contain only strings")
    return result


def _positive_int(value, name):
    if isinstance(value, (bool, np.ndarray)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive scalar integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _threshold(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.ndarray)) or not isinstance(value, numbers.Real):
        raise TypeError("threshold must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    return value


def _snapshot_params(params):
    if params is None:
        return FingerprintParams()
    if not isinstance(params, FingerprintParams):
        raise TypeError("fp_params must be a FingerprintParams instance")
    # The native options object is made at call time; this snapshot prevents
    # changes to a caller-owned object during a computation from affecting it.
    fields = ("fp_type", "fp_method", "fpSize", "radius", "useBondTypes",
              "minPath", "maxPath", "numBitsPerFeature", "use2D", "minDistance",
              "maxDistance", "countSimulation", "includeChirality", "targetSize")
    clone = FingerprintParams.__new__(FingerprintParams)
    for field in fields:
        setattr(clone, field, getattr(params, field))
    _validate_fingerprint_params(clone)
    for field in ("fpSize", "radius", "minPath", "maxPath",
                  "numBitsPerFeature", "minDistance", "maxDistance",
                  "targetSize"):
        setattr(clone, field, int(getattr(clone, field)))
    for field in ("useBondTypes", "use2D", "countSimulation", "includeChirality"):
        setattr(clone, field, bool(getattr(clone, field)))
    return clone


def cross_similarity(left_smiles: Iterable[str], right_smiles: Iterable[str], *,
                     backend="cpu", threshold=None, fp_params=None,
                     assume_sanitized=False, output_path=None, overwrite=False,
                     batch_size=4096, tile_size=256):
    """Compute rectangular binary Tanimoto similarity.

    ``threshold`` produces deterministic row-major COO arrays and is inclusive.
    The comparison is against the original double threshold after formation of
    the float32 score.  ``output_path`` produces a dense C-order ``.npy`` and
    cannot be combined with ``threshold``.  Invalid source positions remain in
    the result: dense pairs are NaN and COO pairs are omitted.  ``assume_sanitized``
    asserts caller responsibility for valid strings; it does not disable RDKit's
    molecule construction and parser failures still propagate as invalid input.
    """
    left_smiles = _materialize(left_smiles, "left_smiles")
    right_smiles = _materialize(right_smiles, "right_smiles")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    threshold = _threshold(threshold)
    if output_path is not None and threshold is not None:
        raise ValueError("threshold and output_path cannot be combined")
    batch_size = _positive_int(batch_size, "batch_size")
    tile_size = _positive_int(tile_size, "tile_size")
    if not isinstance(assume_sanitized, (bool, np.bool_)):
        raise TypeError("assume_sanitized must be boolean")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError("overwrite must be boolean")
    overwrite = bool(overwrite)
    params = _snapshot_params(fp_params)
    if backend == "gpu":
        from ._similarity_gpu import cross_similarity_gpu
        return cross_similarity_gpu(left_smiles, right_smiles, threshold=threshold,
                                    fp_params=params, assume_sanitized=bool(assume_sanitized),
                                    output_path=output_path, overwrite=overwrite,
                                    batch_size=batch_size, tile_size=tile_size)
    if output_path is not None:
        try:
            from .parallel_rdkit_backend import cross_similarity_dense_to_file
        except ImportError as exc:
            raise ImportError("the rebuilt native streaming similarity extension is required") from exc
        native_opts = params.to_backend_opts()
        path = Path(output_path)
        try:
            masks = cross_similarity_dense_to_file(
                left_smiles, right_smiles, native_opts, bool(assume_sanitized),
                batch_size, tile_size, str(path), bool(overwrite))
        except RuntimeError as exc:
            # Native no-clobber publication uses an atomic hard-link.  Nanobind
            # exposes its filesystem_error as RuntimeError, so translate only
            # that publication collision to Python's public OSError type.
            text = str(exc)
            if (not overwrite and os.path.lexists(path) and
                    "publish similarity file" in text):
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST),
                                      str(path)) from exc
            if "filesystem error:" in text:
                raise OSError(text) from exc
            raise
        left_valid = np.asarray(masks[0], dtype=np.bool_)
        right_valid = np.asarray(masks[1], dtype=np.bool_)
        return SimilarityResult((len(left_smiles), len(right_smiles)), left_valid, right_valid,
                                output_path=path)
    # Dense and COO in-memory results use the packed native entry points.
    try:
        from .parallel_rdkit_backend import (cross_similarity_dense_into,
            cross_similarity_coo_count_into, cross_similarity_coo_fill_into)
        native_opts = params.to_backend_opts()
    except ImportError as exc:
        raise ImportError("the rebuilt native rectangular similarity extension is required") from exc
    if threshold is not None:
        left_valid = np.empty(len(left_smiles), dtype=np.bool_)
        right_valid = np.empty(len(right_smiles), dtype=np.bool_)
        counts = np.asarray(cross_similarity_coo_count_into(
            left_smiles, right_smiles, native_opts, bool(assume_sanitized), threshold,
            batch_size, tile_size, left_valid, right_valid), dtype=np.int64)
        offsets = np.empty(len(counts) + 1, dtype=np.int64); offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        rows = np.empty(int(offsets[-1]), dtype=np.int64)
        cols = np.empty_like(rows); values = np.empty(rows.size, dtype=np.float32)
        cross_similarity_coo_fill_into(left_smiles, right_smiles, native_opts,
            bool(assume_sanitized), threshold, batch_size, tile_size,
            rows, cols, values)
        return SimilarityResult((len(left_smiles), len(right_smiles)), left_valid, right_valid,
                                coo=(rows, cols, values))
    if output_path is None:
        dense = np.empty((len(left_smiles), len(right_smiles)), dtype=np.float32, order="C")
        left_valid = np.empty(len(left_smiles), dtype=np.bool_)
        right_valid = np.empty(len(right_smiles), dtype=np.bool_)
        cross_similarity_dense_into(left_smiles, right_smiles, native_opts,
            bool(assume_sanitized), batch_size, tile_size, dense, left_valid, right_valid)
        return SimilarityResult((len(left_smiles), len(right_smiles)), left_valid, right_valid,
                                dense=dense)


__all__ = ["SimilarityResult", "cross_similarity"]
