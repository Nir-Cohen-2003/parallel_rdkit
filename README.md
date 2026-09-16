# Parallel RDKit

## MS-Ready SMILES

MS-Ready (Mass Spectrometry Ready) SMILES are standardized molecular representations designed for mass spectrometry analysis. The standardization process:

1. **Cleanup**: Metal disconnection, normalization, and reionization
2. **Fragment Parent**: Salt stripping (removes counterions)
3. **Charge Parent**: Neutralization (removes charges)
4. **Tautomer Canonicalization**: Converts to canonical tautomer form
5. **Carbon Check**: Only organic molecules (containing carbon) produce MS-Ready SMILES

## Function Documentation

### Module: `parallel_rdkit`

#### `msready_smiles(smiles: str) -> str`

Transforms a SMILES string into an MS-Ready SMILES string.

**Return Values:**
- Valid MS-Ready SMILES string for organic molecules (containing carbon)
- `"<INORGANIC>"` for inorganic molecules (no carbon atoms), indicating no MS-Ready form exists
- `""` for invalid/parse errors

#### `msready_smiles_parallel(smiles: Iterable[str]) -> List[str]`

Parallel MS-Ready transformation of SMILES.

Returns a list where each element follows the same convention as `msready_smiles()`:
- MS-Ready SMILES for organic molecules
- `"<INORGANIC>"` for inorganic molecules (no carbon atoms)
- `""` for invalid molecules

#### `sanitize_smiles_parallel(smiles: Iterable[str]) -> List[str]`

Parallel SMILES sanitization.

#### `inchi_to_smiles_parallel(inchis: Iterable[str]) -> List[str]`

Parallel InChI to SMILES conversion.

#### `smiles_to_inchi_parallel(smiles: Iterable[str]) -> List[str]`

Parallel SMILES to InChI conversion.

#### `smiles_to_inchikey_parallel(smiles: Iterable[str]) -> List[str]`

Parallel SMILES to InChIKey conversion.

#### `msready_inchi_inchikey_parallel(smiles: Iterable[str]) -> Tuple[List[str], List[str], List[str]]`

Parallel conversion to MS-Ready SMILES, InChI, and InChIKey simultaneously.
Returns a tuple of lists: (list of MS-Ready SMILES, list of InChIs, list of InChIKeys)

The MS-Ready SMILES list follows the same convention as `msready_smiles()`:
- MS-Ready SMILES for organic molecules
- `"<INORGANIC>"` for inorganic molecules (no carbon atoms)
- `""` for invalid molecules

### Module: `parallel_rdkit.fingerprint`

#### `class FingerprintParams`

Configuration class for fingerprint parameters.

**Parameters:**
- `fp_type` (str): Type of fingerprint - "morgan", "atompair", "torsion", "rdkit", or "maccs" (default: "morgan")
- `fp_method` (str): Fingerprint method - "GetFingerprint" or "GetHashedFingerprint" (default: "GetFingerprint")
- `fpSize` (int): Number of bits in the fingerprint (default: 2048)
- `radius` (int): Morgan fingerprint radius (default: 3 for morgan, 2 for others)
- `useBondTypes` (bool): Include bond types in fingerprint (default: True)
- `minPath` (int): Minimum path length for RDKit fingerprints (default: 1)
- `maxPath` (int): Maximum path length for RDKit fingerprints (default: 7)
- `numBitsPerFeature` (int): Number of bits set per feature for RDKit fingerprints (default: 2)
- `use2D` (bool): Use 2D coordinates for atom pair/torsion fingerprints (default: True)
- `minDistance` (int): Minimum distance for atom pair fingerprints (default: 1)
- `maxDistance` (int): Maximum distance for atom pair/torsion fingerprints (default: 30)
- `countSimulation` (bool): Use count simulation for atom pair/torsion fingerprints (default: True for these types)
- `includeChirality` (bool): Include chirality information (default: False)
- `targetSize` (int): Target size for torsion fingerprints (default: 4)

#### `get_fp_list(smiles: Iterable[str], params: FingerprintParams, return_numpy: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[bool]]]`

Get fingerprints for a list of SMILES strings.

Args:
    smiles: Iterable of SMILES strings.
    params: Fingerprint parameters.
    return_numpy: If True (default), returns a tuple of (2D numpy array, 1D boolean array). 
                  If False, returns a tuple of (list of 1D numpy arrays, list of bools).
                  
Returns:
    A tuple (fingerprints, valid_mask). `valid_mask` is an array of booleans where `True` means the molecule was processed successfully and `False` means an error occurred (e.g., invalid SMILES).

### Module: `parallel_rdkit.screen_smarts`

#### `screen_smarts(smarts_list: List[str], smiles_file: Union[str, Path], mode: str = "direct", batch_size: int = 10000, cache_path: Optional[Union[str, Path]] = None, output_path: Optional[Union[str, Path]] = None) -> Union[np.ndarray, int]`

Screen molecules from a SMILES file against a list of SMARTS patterns.

**Modes:**
- `mode="direct"`: Loads all molecules into memory and returns an N x M boolean array
- `mode="streaming"`: Processes molecules in batches and writes results to output file

**Args:**
- `smarts_list`: List of SMARTS patterns to screen against
- `smiles_file`: Path to file containing SMILES (one per line)
- `mode`: Either "direct" or "streaming"
- `batch_size`: Number of molecules per batch (streaming mode only)
- `cache_path`: Path to cache file for skipping recomputation
- `output_path`: Required for streaming mode, path to output .npy file

**Returns:**
- Direct mode: N x M boolean numpy array (N molecules, M SMARTS patterns)
- Streaming mode: Number of molecules processed

### Clustering helpers

Similarity matrices can be partitioned with the generic clustering helpers
exported from the package root or `parallel_rdkit.clustering`. The historical
`parallel_rdkit.matrix_similarity` import path remains available for these two
helpers.

#### `butina_split(sim_matrix: np.ndarray, dist_threshold: float = 0.3) -> List[int]`

Perform Butina clustering using a distance cutoff of `dist_threshold`.

#### `umap_split(sim_matrix: np.ndarray, n_clusters: int = 10, random_state: int = 42, **umap_kwargs) -> List[int]`

Reduce a similarity matrix with UMAP and cluster the embedding with KMeans.
UMAP and scikit-learn are optional dependencies required only by this helper.

### Module: `parallel_rdkit.mol`

#### `sanitize_smiles(smiles: Iterable[str], batch_size: int = 1000) -> List[str]`

Sanitize a list of SMILES strings in parallel.

Args:
    smiles: Iterable of SMILES strings.
    batch_size: Ignored in this implementation as C++ handles batching.
    
Returns:
    List of sanitized SMILES strings.

#### `smiles_to_formula(smiles: Union[Iterable[str], pl.Series]) -> Union[np.ndarray, pl.Series]`

Compute molecular formulas for a list or polars Series of SMILES strings.

The C++ backend parallelizes formula computation over molecules using OpenMP. The backend returns a flattened `(n*12)` array of int64 counts, which Python reshapes into a 2D numpy array of shape `(n, 12)`.

Element counts are returned in the fixed order:
`["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]`

The return type depends on the input type:
- If `smiles` is a `polars.Series`, the 2D numpy array is ingested directly into a `polars.Series` of dtype `Array(Int64, 12)` (no Python list of arrays is created).
- Otherwise, a 2D `int64` numpy array of shape `(n, 12)` is returned.

Invalid SMILES produce a row of zeros. Atoms whose elements are not in the 12-element list above are silently ignored.

Args:
    smiles: An iterable of SMILES strings, or a polars Series of SMILES.

Returns:
    `np.ndarray` of shape `(n, 12)` for list/iterable input, or a `pl.Series` of dtype `Array(Int64, 12)` for polars Series input.

**Example (list input):**
```python
from parallel_rdkit import smiles_to_formula

smiles_to_formula(["CCO", "CF"])
# array([[6, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#        [3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)
```

**Example (polars input):**
```python
import polars as pl
from parallel_rdkit import smiles_to_formula

s = pl.Series(["CCO", "CF"])
smiles_to_formula(s)
# shape: (2,)
# Series: 'formula' [array[i64, 12]]
# [
#     [6, 2, 0, 1, 0, 0, ... 0]
#     [3, 1, 0, 0, 1, 0, ... 0]
# ]
```

### Module: `parallel_rdkit.stoned`

**Dependencies Required:**
- `selfies` - SELFIES molecular string representation

Install separately: `pip install selfies`

The `parallel_rdkit.stoned` module implements the STONED (Selfies TO NEw molecules with Decoding) algorithm with C++ parallel acceleration for the RDKit-heavy steps (SMILES randomization, sanitization, and fingerprint scoring). The SELFIES token manipulation remains in Python, as the `selfies` library is Python-only.

#### `generate_local_space(smiles: str, num_random_samples: int = 1000, num_mutation_ls: List[int] = None, fp_params: Optional[FingerprintParams] = None, return_scores: bool = False) -> Union[List[str], Tuple[List[str], List[float]]]`

Generate a local chemical space around a single starting SMILES using randomized SMILES orderings, SELFIES encoding, and token mutations.

**Accelerated steps (C++ OpenMP):**
- SMILES randomization (`randomize_smiles_parallel`)
- Sanitization (`sanitize_smiles_parallel`)
- Fingerprint scoring (`tanimoto_scores_parallel`) when `return_scores=True`

Args:
    smiles: Starting SMILES string.
    num_random_samples: Number of randomized SMILES orderings (default: 1000).
    num_mutation_ls: List of mutation depths to apply (default: [1, 2, 3, 4, 5]).
    fp_params: Fingerprint parameters for scoring. Required if `return_scores=True`.
    return_scores: If True, returns `(smiles_list, scores_list)`.

Returns:
    List of unique generated SMILES, or tuple of (SMILES, scores) if `return_scores=True`.

**Example:**
```python
from parallel_rdkit.stoned import generate_local_space
from parallel_rdkit.fingerprint import FingerprintParams

smiles, scores = generate_local_space(
    smiles="CCO",
    num_random_samples=500,
    num_mutation_ls=[1, 2, 3],
    fp_params=FingerprintParams(fp_type="morgan", radius=2, fpSize=2048),
    return_scores=True,
)
print(f"Generated {len(smiles)} unique molecules")
```

#### `generate_pair_paths(smiles_list: List[str], num_tries: int = 2, num_random_samples: int = 2, collect_bidirectional: bool = True) -> List[str]`

Generate chemical paths between exactly 2 SMILES by greedily flipping differing SELFIES tokens.

Args:
    smiles_list: Exactly 2 SMILES strings.
    num_tries: Path attempts per randomized pair (default: 2).
    num_random_samples: Random orderings per endpoint (default: 2).
    collect_bidirectional: Also generate paths in the reverse direction (default: True).

Returns:
    List of unique SMILES strings along all paths.

#### `generate_triplet_paths(smiles_list: List[str], num_paths: int = 100, num_random_samples: int = 1) -> List[str]`

Generate median molecules / generalized paths from all combinations of 3 SMILES.

Args:
    smiles_list: At least 3 SMILES strings.
    num_paths: Number of paths to attempt per triplet (default: 100).
    num_random_samples: Random orderings per molecule (default: 1).

Returns:
    List of unique median/path SMILES strings.

#### `get_random_smiles(smi: str, num_random_samples: int) -> List[str]`

Obtain random SMILES orderings of a single SMILES using the C++ parallel backend.

Args:
    smi: Input SMILES string.
    num_random_samples: Number of randomized variants to generate.

Returns:
    List of unique randomized SMILES strings.

#### `tanimoto_scores_parallel(smiles: List[str], target_smi: str, fp_params: FingerprintParams) -> List[float]`

**Low-level C++ backend.** Compute Tanimoto similarity scores between a list of SMILES and a target SMILES in parallel using OpenMP.

This is automatically used by `generate_local_space` when `return_scores=True`, but can be called directly for custom workflows.

Args:
    smiles: Query SMILES strings.
    target_smi: Target SMILES string.
    fp_params: Fingerprint parameters.

Returns:
    List of Tanimoto scores (0.0 for invalid molecules).

## Benchmarking

The reproducible rectangular similarity benchmark is available at
`benchmarks/benchmark_cross_similarity.py`.

## Rectangular cross similarity

Use `cross_similarity(left_smiles, right_smiles, *, threshold=None,
fp_params=None, assume_sanitized=False, output_path=None, overwrite=False,
batch_size=4096, tile_size=256)` for a rectangular binary Tanimoto operation.
The result has `shape`, `left_valid`, and `right_valid` plus exactly one payload:
`dense` (C-order float32), `coo` (int64 row/column and float32 value arrays), or
`output_path` (a completed dense `.npy`). Thresholding is inclusive after the
score is formed as float32, and valid zero/zero vectors score 0.0.

**Invalid input positions remain present; every dense pair involving one is
NaN.** Use the validity masks and NaN-aware reductions. COO excludes invalid
pairs but preserves original indices and row-major ordering. `batch_size` and
`tile_size` independently bound molecule and score tiles. The native
implementation supports Morgan, RDKit, AtomPair, TopologicalTorsion, and
canonical 167-bit MACCS binary fingerprints; count-vector methods are rejected.
`assume_sanitized=True` is a caller assertion and does not disable RDKit
molecule construction. File output uses an atomic transaction and requires
`overwrite=True` to replace an existing path.
