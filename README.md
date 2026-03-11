# Parallel RDKit

## Function Documentation

### Module: `parallel_rdkit`

#### `msready_smiles(smiles: str) -> str`

Transforms a SMILES string into an MS-Ready SMILES string.

#### `msready_smiles_parallel(smiles: Iterable[str]) -> List[str]`

Parallel MS-Ready transformation of SMILES.

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

### Module: `parallel_rdkit.fingerprint`

#### `get_fp_list(smiles: Iterable[str], params: FingerprintParams, return_numpy: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[bool]]]`

Get fingerprints for a list of SMILES strings.

Args:
    smiles: Iterable of SMILES strings.
    params: Fingerprint parameters.
    return_numpy: If True (default), returns a tuple of (2D numpy array, 1D boolean array). 
                  If False, returns a tuple of (list of 1D numpy arrays, list of bools).
                  
Returns:
    A tuple (fingerprints, valid_mask). `valid_mask` is an array of booleans where `True` means the molecule was processed successfully and `False` means an error occurred (e.g., invalid SMILES).

### Module: `parallel_rdkit.matrix_tanimoto`

#### `calculate_tanimoto_matrix(smiles: List[str], fp_radius: int = 2, fp_size: int = 2048, save_path: Optional[Union[str, Path]] = None, log_path: Optional[Union[str, Path]] = None) -> np.ndarray`

Calculate a dense Tanimoto similarity matrix for a list of SMILES strings using GPU.

Args:
    smiles: List of SMILES strings.
    fp_radius: Morgan fingerprint radius.
    fp_size: Morgan fingerprint bit size.
    save_path: Path to save the resulting matrix as a .npy file.
    log_path: Path to log progress.
    
Returns:
    A NumPy array of shape (N, N) containing similarity scores.

#### `calculate_tanimoto_matrix_streaming(parquet_path: Union[str, Path], smiles_column: str = 'smiles', save_path: Optional[Union[str, Path]] = None, log_path: Optional[Union[str, Path]] = None, fp_radius: int = 2, fp_size: int = 2048) -> np.ndarray`

Calculate Tanimoto matrix by reading SMILES from a parquet file using polars streaming.

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

