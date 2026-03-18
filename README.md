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

### Module: `parallel_rdkit.matrix_similarity`

**Dependencies Required:**
- `nvmolkit` - GPU-accelerated fingerprint generation and similarity calculation
- `torch` - PyTorch tensors for GPU computation

Install separately: `pip install nvmolkit torch`

#### `calculate_similarity_matrix(smiles: List[str], parquet_path: Union[str, Path], indices: Optional[np.ndarray] = None, fp_params: Optional[FingerprintParams] = None, similarity_metric: Literal["tanimoto", "cosine", "both"] = "tanimoto", threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Calculate similarity matrix for a list of SMILES strings using GPU acceleration.

Writes lower-triangular similarities to parquet file (memory efficient). Only parquet output mode is supported.

**How the Calculation Works:**
1. **Fingerprint Generation**: Uses nvmolkit's GPU-accelerated Morgan fingerprint generator
2. **Similarity Computation**: Computes the matrix in row chunks based on available memory
3. **Lower Triangular Storage**: Only stores pairs where `mol1_idx <= mol2_idx` to avoid redundancy
4. **Threshold Filtering**: Applied on GPU before transferring to CPU (optional)
5. **Chunked Writing**: Appends results incrementally to parquet file using PyArrow

Args:
    smiles: List of SMILES strings.
    parquet_path: Path to output parquet file (required).
    indices: Original molecule indices from the caller. If None, uses `range(len(smiles))`. Invalid SMILES are filtered out and their indices are removed from output.
    fp_params: Fingerprint parameters as `FingerprintParams` dataclass. Defaults to Morgan fingerprints with radius=2, fpSize=2048.
    similarity_metric: Type of similarity to compute - "tanimoto", "cosine", or "both".
    threshold: Minimum similarity threshold. Only similarities >= threshold are stored.
    memory_usage_fraction: Fraction of available CPU memory to use for chunks (0.0-1.0, default 0.5).
    log_path: Path to log progress. If None, uses `{parquet_path}.log`.
    
Returns:
    None (results written to `parquet_path`).

**Parquet Output Format:**
Three columns when metric is "tanimoto" or "cosine":
- `mol1_idx` (uint32), `mol2_idx` (uint32), `tanimoto` or `cosine` (float32)

Five columns when metric is "both":
- `mol1_idx` (uint32), `mol2_idx` (uint32), `tanimoto` (float32), `cosine` (float32)

- Only lower triangular pairs stored (mol1_idx <= mol2_idx)
- Snappy compression enabled
- Dictionary encoding for index columns

**Example:**
```python
from parallel_rdkit.matrix_similarity import calculate_similarity_matrix
from parallel_rdkit.fingerprint import FingerprintParams
import numpy as np

smiles = ["CCO", "CCCO", "CCCCO"]
indices = np.array([100, 200, 300], dtype=np.uint32)

# Calculate Tanimoto similarities with threshold filtering
calculate_similarity_matrix(
    smiles=smiles,
    parquet_path="/path/to/output.parquet",
    indices=indices,
    fp_params=FingerprintParams(fp_type="morgan", radius=2, fpSize=2048),
    similarity_metric="tanimoto",
    threshold=0.5,  # Only store similarities >= 0.5
    memory_usage_fraction=0.5
)
# Creates: output.parquet and output.parquet.log

# Calculate both Tanimoto and Cosine similarities
calculate_similarity_matrix(
    smiles=smiles,
    parquet_path="/path/to/output_both.parquet",
    indices=indices,
    similarity_metric="both",
    threshold=0.5
)
# Creates parquet with both tanimoto and cosine columns
```

#### `calculate_tanimoto_matrix(smiles: List[str], parquet_path: Union[str, Path], indices: Optional[np.ndarray] = None, fp_params: Optional[FingerprintParams] = None, threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Backward compatibility alias for `calculate_similarity_matrix()` with `similarity_metric="tanimoto"`.

#### `calculate_cosine_matrix(smiles: List[str], parquet_path: Union[str, Path], indices: Optional[np.ndarray] = None, fp_params: Optional[FingerprintParams] = None, threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Alias for `calculate_similarity_matrix()` with `similarity_metric="cosine"`.

#### `calculate_similarity_matrix_streaming(parquet_path: Union[str, Path], output_parquet_path: Union[str, Path], smiles_column: str = "smiles", index_column: str = "index", fp_params: Optional[FingerprintParams] = None, similarity_metric: Literal["tanimoto", "cosine", "both"] = "tanimoto", threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Calculate similarity matrix by reading SMILES from a parquet file using polars.

Reads SMILES and their corresponding indices from an input parquet file, then computes similarities using `calculate_similarity_matrix()`.

Args:
    parquet_path: Path to input parquet file containing SMILES and indices.
    output_parquet_path: Path for output parquet file (required).
    smiles_column: Name of column containing SMILES strings (default: "smiles").
    index_column: Name of column containing molecule indices (default: "index").
    fp_params: Fingerprint parameters as `FingerprintParams` dataclass.
    similarity_metric: "tanimoto", "cosine", or "both".
    threshold: Minimum similarity threshold.
    memory_usage_fraction: Fraction of available CPU memory to use.
    log_path: Path to log file. Auto-generated from output_parquet_path if not provided.
    
Returns:
    None (results written to `output_parquet_path`).

#### `calculate_tanimoto_matrix_streaming(parquet_path: Union[str, Path], output_parquet_path: Union[str, Path], smiles_column: str = "smiles", index_column: str = "index", fp_params: Optional[FingerprintParams] = None, threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Backward compatibility alias for streaming calculation with `similarity_metric="tanimoto"`.

#### `calculate_cosine_matrix_streaming(parquet_path: Union[str, Path], output_parquet_path: Union[str, Path], smiles_column: str = "smiles", index_column: str = "index", fp_params: Optional[FingerprintParams] = None, threshold: Optional[float] = None, memory_usage_fraction: float = 0.5, log_path: Optional[Union[str, Path]] = None) -> None`

Streaming calculation with `similarity_metric="cosine"`.

#### `butina_split(sim_matrix: np.ndarray, dist_threshold: float = 0.3) -> List[int]`

Perform Butina clustering on the similarity matrix.

Args:
    sim_matrix: Dense similarity matrix (N x N).
    dist_threshold: Distance threshold (1 - similarity) for clustering.
    
Returns:
    List of cluster IDs for each molecule in the matrix.

#### `umap_split(sim_matrix: np.ndarray, n_clusters: int = 10, random_state: int = 42) -> List[int]`

Perform UMAP reduction followed by KMeans clustering for splitting.

Args:
    sim_matrix: Similarity matrix.
    n_clusters: Number of clusters for KMeans.
    random_state: Seed for reproducibility.
    **umap_kwargs: Additional arguments for UMAP (e.g., n_neighbors, min_dist).
    
Returns:
    List of cluster labels.

### Module: `parallel_rdkit.mol`

#### `sanitize_smiles(smiles: Iterable[str], batch_size: int = 1000) -> List[str]`

Sanitize a list of SMILES strings in parallel.

Args:
    smiles: Iterable of SMILES strings.
    batch_size: Ignored in this implementation as C++ handles batching.
    
Returns:
    List of sanitized SMILES strings.

