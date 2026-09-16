"""Small reproducible cross_similarity benchmark."""
import argparse
import time
from pathlib import Path
import numpy as np
from parallel_rdkit import FingerprintParams, cross_similarity

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "C", "N"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--left-size", type=int, default=100)
    p.add_argument("--right-size", type=int, default=100)
    p.add_argument("--fp-size", type=int, default=2048)
    p.add_argument("--threshold", type=float)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--output-path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--reference", action="store_true")
    a = p.parse_args()
    left = (SMILES * ((a.left_size + len(SMILES)-1)//len(SMILES)))[:a.left_size]
    right = (SMILES * ((a.right_size + len(SMILES)-1)//len(SMILES)))[:a.right_size]
    params = FingerprintParams(fpSize=a.fp_size)
    for _ in range(a.repeat):
        started = time.perf_counter()
        result = cross_similarity(left, right, backend=a.backend, threshold=a.threshold,
                                  fp_params=params, output_path=a.output_path,
                                  overwrite=a.overwrite, batch_size=a.batch_size,
                                  tile_size=a.tile_size)
        elapsed = time.perf_counter() - started
        payload = result.coo[0].size if result.coo is not None else (result.shape[0] * result.shape[1])
        print(f"elapsed={elapsed:.6f}s pairs={payload} pairs_per_second={payload/max(elapsed, 1e-12):.3f}")
        if result.output_path:
            print(f"output={Path(result.output_path)} bytes={Path(result.output_path).stat().st_size}")
        if a.reference and result.dense is not None:
            print("reference=available (use RDKit generators for an independent comparison)")

if __name__ == "__main__":
    main()
