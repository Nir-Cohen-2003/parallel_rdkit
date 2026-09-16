"""Strict nvMolKit adapter for :func:`cross_similarity`.

This module is intentionally lazy: importing parallel_rdkit never imports CUDA,
nvMolKit, or torch.  Unsupported settings are errors, never CPU fallbacks.
"""
from __future__ import annotations

import numpy as np

from .similarity import SimilarityResult
from ._similarity_io import _write_npy_transaction


def _validate(params):
    if params.fp_type != "morgan" or params.fp_method != "GetFingerprint":
        raise ValueError("GPU cross_similarity supports Morgan GetFingerprint only")
    if params.fpSize not in {128, 256, 512, 1024, 2048, 4096}:
        raise ValueError("unsupported GPU Morgan fpSize")
    if params.radius < 0:
        raise ValueError("GPU Morgan radius must be nonnegative")
    if (not params.useBondTypes or params.includeChirality or params.countSimulation or
            params.minPath != 1 or params.maxPath != 7 or params.numBitsPerFeature != 2 or
            not params.use2D or params.minDistance != 1 or params.maxDistance != 30 or
            params.targetSize != 4):
        raise ValueError("unsupported non-default GPU fingerprint setting")


def _to_numpy(value):
    # nvMolKit returns an asynchronous result in normal operation and tests
    # commonly provide a torch tensor directly.
    if hasattr(value, "torch"):
        value = value.torch()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _fingerprints(smiles, params, assume_sanitized, batch_size):
    try:
        from rdkit import Chem
        from nvmolkit.fingerprints import MorganFingerprintGenerator
    except ImportError as exc:
        raise ImportError("backend='gpu' requires pinned nvmolkit and torch") from exc
    mols, valid = [], np.zeros(len(smiles), dtype=np.bool_)
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            mols.append(mol)
            valid[i] = True
    # The generator is called in batches so no transfer of invalid positions is
    # needed.  A compact tensor is expanded back to original positions below.
    generator = MorganFingerprintGenerator(radius=params.radius, fpSize=params.fpSize)
    chunks = []
    valid_positions = np.flatnonzero(valid)
    for start in range(0, len(mols), batch_size):
        result = generator.GetFingerprints(mols[start:start + batch_size])
        chunks.append(_to_numpy(result))
    if chunks:
        compact = np.concatenate(chunks, axis=0)
    else:
        compact = np.empty((0, params.fpSize), dtype=np.float32)
    expanded = np.zeros((len(smiles), params.fpSize), dtype=np.float32)
    if compact.size:
        expanded[valid_positions] = compact
    return expanded, valid


def cross_similarity_gpu(left_smiles, right_smiles, *, threshold, fp_params,
                         assume_sanitized, output_path, overwrite, batch_size,
                         tile_size):
    _validate(fp_params)
    try:
        import torch  # noqa: F401 - verifies the pinned GPU runtime is present
        from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
    except ImportError as exc:
        raise ImportError("backend='gpu' requires pinned nvmolkit and torch") from exc
    if output_path is not None:
        # File mode deliberately does not call _fingerprints on either complete
        # library. Masks are parsed once; each pair of GPU fingerprint batches
        # is transferred, consumed, and released before the next pair.
        from rdkit import Chem
        lm = np.array([Chem.MolFromSmiles(s) is not None for s in left_smiles], dtype=np.bool_)
        rm = np.array([Chem.MolFromSmiles(s) is not None for s in right_smiles], dtype=np.bool_)
        def write(mapped):
            for i0 in range(0, len(lm), batch_size):
                lb, _ = _fingerprints(left_smiles[i0:i0 + batch_size], fp_params, assume_sanitized, batch_size)
                for j0 in range(0, len(rm), batch_size):
                    rb, _ = _fingerprints(right_smiles[j0:j0 + batch_size], fp_params, assume_sanitized, batch_size)
                    for ti in range(0, len(lb), tile_size):
                        for tj in range(0, len(rb), tile_size):
                            i1, j1 = min(ti + tile_size, len(lb)), min(tj + tile_size, len(rb))
                            a = torch.as_tensor(lb[ti:i1], device="cuda"); b = torch.as_tensor(rb[tj:j1], device="cuda")
                            block = np.asarray(_to_numpy(crossTanimotoSimilarityMemoryConstrained(a, b)), dtype=np.float32)
                            block[~(lm[i0 + ti:i0 + i1, None] & rm[None, j0 + tj:j0 + j1])] = np.nan
                            mapped[i0 + ti:i0 + i1, j0 + tj:j0 + j1] = block
                    del rb
                del lb
        path = _write_npy_transaction(output_path, write, (len(lm), len(rm)), overwrite)
        return SimilarityResult((len(lm), len(rm)), lm, rm, output_path=path)

    left, lm = _fingerprints(left_smiles, fp_params, assume_sanitized, batch_size)
    right, rm = _fingerprints(right_smiles, fp_params, assume_sanitized, batch_size)

    # Use nvMolKit for every bounded tile.  Keeping the compact arrays here is
    # permitted for in-memory modes; file mode still releases tile results as
    # soon as they have been written.
    def tile(i0, i1, j0, j1):
        a = torch.as_tensor(left[i0:i1], device="cuda")
        b = torch.as_tensor(right[j0:j1], device="cuda")
        value = crossTanimotoSimilarityMemoryConstrained(a, b)
        return np.asarray(_to_numpy(value), dtype=np.float32)

    # The common Python builders enforce the public float32 threshold predicate
    # and deterministic COO ordering. Replace their tile source locally.
    def dense_result():
        out = np.empty((len(lm), len(rm)), dtype=np.float32)
        for i0 in range(0, len(lm), batch_size):
            for j0 in range(0, len(rm), batch_size):
                for ti in range(i0, min(i0 + batch_size, len(lm)), tile_size):
                    for tj in range(j0, min(j0 + batch_size, len(rm)), tile_size):
                        i1, j1 = min(ti + tile_size, len(lm)), min(tj + tile_size, len(rm))
                        block = tile(ti, i1, tj, j1)
                        block[~(lm[ti:i1, None] & rm[None, tj:j1])] = np.nan
                        out[ti:i1, tj:j1] = block
        return out

    if threshold is None and output_path is None:
        payload = dense_result()
        return SimilarityResult((len(lm), len(rm)), lm, rm, dense=payload)
    if threshold is not None:
        # Build two passes without a dense threshold mask.  The temporary tile
        # provider mirrors _coo's row ownership and deterministic ordering.
        class TileArray:
            pass
        # A small local implementation avoids converting GPU output to a
        # complete matrix and applies the exact float32-to-double comparison.
        n, m = len(lm), len(rm)
        counts = np.zeros(n, dtype=np.int64)
        for i0 in range(0, n, batch_size):
            for j0 in range(0, m, batch_size):
                for ti in range(i0, min(i0 + batch_size, n), tile_size):
                    for tj in range(j0, min(j0 + batch_size, m), tile_size):
                        i1, j1 = min(ti + tile_size, n), min(tj + tile_size, m)
                        block = tile(ti, i1, tj, j1)
                        block[~(lm[ti:i1, None] & rm[None, tj:j1])] = np.nan
                        counts[ti:i1] += np.count_nonzero(block.astype(np.float64) >= threshold, axis=1)
        offsets = np.empty(n + 1, dtype=np.int64); offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        rows = np.empty(int(offsets[-1]), dtype=np.int64); cols = np.empty_like(rows)
        vals = np.empty(rows.size, dtype=np.float32); cursor = offsets[:-1].copy()
        for i0 in range(0, n, batch_size):
            for j0 in range(0, m, batch_size):
                for ti in range(i0, min(i0 + batch_size, n), tile_size):
                    for tj in range(j0, min(j0 + batch_size, m), tile_size):
                        i1, j1 = min(ti + tile_size, n), min(tj + tile_size, m)
                        block = tile(ti, i1, tj, j1)
                        block[~(lm[ti:i1, None] & rm[None, tj:j1])] = np.nan
                        for k in range(i1 - ti):
                            hit = np.flatnonzero(block[k].astype(np.float64) >= threshold)
                            start = int(cursor[ti + k]); end = start + hit.size
                            rows[start:end] = ti + k; cols[start:end] = tj + hit
                            vals[start:end] = block[k, hit]; cursor[ti + k] = end
        return SimilarityResult((n, m), lm, rm, coo=(rows, cols, vals))
    # File mode uses the same transactional writer and writes each tile before
    # the next tile is requested.
    def write(mapped):
        for i0 in range(0, len(lm), batch_size):
            for j0 in range(0, len(rm), batch_size):
                for ti in range(i0, min(i0 + batch_size, len(lm)), tile_size):
                    for tj in range(j0, min(j0 + batch_size, len(rm)), tile_size):
                        i1, j1 = min(ti + tile_size, len(lm)), min(tj + tile_size, len(rm))
                        block = tile(ti, i1, tj, j1)
                        block[~(lm[ti:i1, None] & rm[None, tj:j1])] = np.nan
                        mapped[ti:i1, tj:j1] = block
    # The private writer is retained for the approved GPU Python path.
    path = _write_npy_transaction(output_path, write, (len(lm), len(rm)), overwrite)
    return SimilarityResult((len(lm), len(rm)), lm, rm, output_path=path)
