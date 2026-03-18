"""
Benchmark script for similarity matrix calculation using GPU acceleration.

This script benchmarks the calculate_similarity_matrix function on datasets
of various sizes to measure performance and memory usage.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

from parallel_rdkit.fingerprint import FingerprintParams

# Import after setting environment variables
from parallel_rdkit.matrix_similarity import calculate_similarity_matrix


def load_smiles_from_csv(n_molecules):
    """Load SMILES strings from DSSTox.csv file using polars."""
    csv_path = "/gpfs01/work/nircoh/parallel_rdkit/data/DSSTox.csv"

    smiles = (
        pl.scan_csv(csv_path)
        .head(n_molecules)
        .collect()
        .get_column("MS_READY_SMILES")
        .to_list()
    )

    # Ensure we have a plain Python list of strings
    smiles = [str(s) for s in smiles]

    # If we need more molecules than available, duplicate
    while len(smiles) < n_molecules:
        smiles.extend(smiles[: min(n_molecules - len(smiles), len(smiles))])

    return smiles[:n_molecules]


def run_benchmark(
    n_molecules,
    similarity_metric="tanimoto",
    threshold=None,
    memory_fraction=0.5,
    seed=42,
):
    """Run benchmark for similarity matrix calculation."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {n_molecules} molecules")
    print(f"Metric: {similarity_metric}")
    print(f"Threshold: {threshold if threshold is not None else 'None (no filtering)'}")
    print(f"Memory fraction: {memory_fraction}")
    print(f"{'=' * 60}\n")

    # Load SMILES from CSV
    print("Loading SMILES from CSV...", flush=True)
    smiles = load_smiles_from_csv(n_molecules)
    indices = np.arange(len(smiles), dtype=np.uint32)
    print(f"Loaded {len(smiles)} SMILES from CSV\n", flush=True)

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        output_path = f.name

    log_path = output_path + ".log"

    try:
        # Run benchmark
        print("Starting similarity matrix calculation...", flush=True)
        start_time = time.time()

        calculate_similarity_matrix(
            smiles=smiles,
            parquet_path=output_path,
            indices=indices,
            fp_params=FingerprintParams(fp_type="morgan", radius=2, fpSize=2048),
            similarity_metric=similarity_metric,
            threshold=threshold,
            memory_usage_fraction=memory_fraction,
            log_path=log_path,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Get output file size
        output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        print(f"\nResults:", flush=True)
        print(f"  Time taken: {duration:.2f} seconds", flush=True)
        print(
            f"  Throughput: {n_molecules / duration:.1f} molecules/second", flush=True
        )
        print(f"  Output file size: {output_size_mb:.2f} MB", flush=True)
        print(f"  Output path: {output_path}", flush=True)
        print(f"  Log path: {log_path}", flush=True)

        # Print log summary if exists
        if Path(log_path).exists():
            print(f"\nLog summary:", flush=True)
            with open(log_path, "r") as f:
                log_content = f.read()
                # Show last few lines
                lines = log_content.strip().split("\n")
                for line in lines[-5:]:
                    print(f"  {line}", flush=True)

        return {
            "n_molecules": n_molecules,
            "duration": duration,
            "throughput": n_molecules / duration,
            "output_size_mb": output_size_mb,
        }

    finally:
        # Cleanup
        if Path(output_path).exists():
            Path(output_path).unlink()
        if Path(log_path).exists():
            Path(log_path).unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark similarity matrix calculation"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
        help="Dataset sizes to benchmark (default: 1000 5000 10000)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["tanimoto", "cosine", "both"],
        help="Similarity metric to calculate (default: both)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold (default: no threshold)",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.3,
        help="Memory fraction for chunking (default: 0.3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Similarity Matrix Benchmark")
    print("=" * 60)
    print(f"Sizes: {args.sizes}")
    print(f"Metric: {args.metric}")
    print(
        f"Threshold: {args.threshold if args.threshold is not None else 'None (no filtering)'}"
    )
    print(f"Memory fraction: {args.memory_fraction}")
    print("=" * 60)

    results = []
    for size in args.sizes:
        result = run_benchmark(
            n_molecules=size,
            similarity_metric=args.metric,
            threshold=args.threshold,
            memory_fraction=args.memory_fraction,
            seed=args.seed,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(
        f"{'Molecules':>12} | {'Time (s)':>10} | {'Throughput':>12} | {'Size (MB)':>10}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['n_molecules']:>12} | {r['duration']:>10.2f} | {r['throughput']:>10.1f} m/s | {r['output_size_mb']:>10.2f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
