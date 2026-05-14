"""
STONED (Selfies TO NEw molecules with Decoding) generation utilities.

This module provides molecular structure generation using the STONED algorithm
with C++ parallel acceleration for the compute-intensive steps:
- SMILES randomization (C++ OpenMP)
- Batch sanitization (C++ OpenMP)
- Fingerprint scoring (C++ OpenMP)

The SELFIES encode/decode uses selfies_rs (Rust) for performance.
"""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

try:
    import selfies_rs as selfies
    _SELFIES_AVAILABLE = True
except ImportError:
    try:
        import selfies
        _SELFIES_AVAILABLE = True
    except ImportError:
        _SELFIES_AVAILABLE = False

from .parallel_rdkit_backend import (
    randomize_smiles_parallel as _randomize_smiles_parallel,
    tanimoto_scores_parallel as _tanimoto_scores_parallel,
)
from . import sanitize_smiles_parallel
from .fingerprint import FingerprintParams


def _require_selfies():
    if not _SELFIES_AVAILABLE:
        raise ImportError(
            "The 'selfies_rs' (or 'selfies') package is required for STONED generation. "
            "Install it with: pip install selfies_rs"
        )


def get_selfie_chars(selfie: str) -> List[str]:
    """Obtain a list of all SELFIE characters in a SELFIES string."""
    chars_selfie = []
    while selfie != '':
        start = selfie.find('[')
        end = selfie.find(']')
        if start == -1 or end == -1:
            break
        chars_selfie.append(selfie[start:end + 1])
        selfie = selfie[end + 1:]
    return chars_selfie


def mutate_selfie(selfie: str, max_molecules_len: int) -> Tuple[str, str]:
    """Return a mutated SELFIE string (single mutation) and its canonical SMILES."""
    _require_selfies()
    valid = False
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        alphabet = list(selfies.get_semantic_robust_alphabet())
        choice_ls = [1, 2, 3]  # 1=Insert, 2=Replace, 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]
            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[1:]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1:]
                )
        else:  # Delete
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[1:]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + chars_selfie[random_index + 1:]
                )

        selfie_mutated = "".join(x for x in selfie_mutated_chars)

        try:
            smiles = selfies.decoder(selfie_mutated)
            mol = smi2mol(smiles, sanitize=True)
            if mol is not None:
                smiles_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
                if smiles_canon and len(selfie_mutated_chars) <= max_molecules_len:
                    valid = True
                else:
                    valid = False
            else:
                valid = False
        except Exception:
            valid = False

    return selfie_mutated, smiles_canon


def get_mutated_SELFIES(selfies_ls: List[str], num_mutations: int) -> List[str]:
    """Mutate all SELFIES in 'selfies_ls' 'num_mutations' times."""
    _require_selfies()
    for _ in range(num_mutations):
        selfie_ls_mut_ls = []
        for str_ in selfies_ls:
            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls


def generate_local_space(
    smiles: str,
    num_random_samples: int = 1000,
    num_mutation_ls: Optional[List[int]] = None,
    fp_params: Optional[FingerprintParams] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], List[float]]]:
    """
    Generate a local chemical space around a single SMILES using STONED.

    Uses C++ parallel backends for SMILES randomization, sanitization,
    and optional fingerprint scoring. Uses selfies_rs for encode/decode.

    Parameters
    ----------
    smiles : str
        Starting SMILES string.
    num_random_samples : int
        Number of randomized SMILES orderings to generate.
    num_mutation_ls : List[int], optional
        Mutation depths to apply (default: [1, 2, 3, 4, 5]).
    fp_params : FingerprintParams, optional
        If provided together with return_scores=True, computes Tanimoto
        similarity scores of each generated molecule back to the starting SMILES.
    return_scores : bool
        If True and fp_params is provided, also returns similarity scores.

    Returns
    -------
    List[str] or Tuple[List[str], List[float]]
        Unique generated SMILES, optionally with their Tanimoto scores.
    """
    _require_selfies()

    if num_mutation_ls is None:
        num_mutation_ls = [1, 2, 3, 4, 5]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid starting SMILES: {smiles}")

    # Use C++ parallel backend for randomization
    raw_randomized = _randomize_smiles_parallel([smiles], num_random_samples)
    randomized_smile_orderings = list(set([s for s in raw_randomized if s]))

    # Convert to SELFIES using batch encoder (selfies_rs)
    if hasattr(selfies, 'encoder_batch'):
        selfies_ls = selfies.encoder_batch(randomized_smile_orderings)
    else:
        selfies_ls = [selfies.encoder(x) for x in randomized_smile_orderings]

    # Mutate and decode (Python loop - SELFIES manipulation)
    all_smiles = []
    for num_mutations in num_mutation_ls:
        selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)
        # Batch decode (selfies_rs)
        if hasattr(selfies, 'decoder_batch'):
            smiles_back = selfies.decoder_batch(selfies_mut)
        else:
            smiles_back = [selfies.decoder(x) for x in selfies_mut]
        all_smiles.extend(smiles_back)

    # Use C++ parallel backend for sanitization and deduplication
    canon_smi_ls = [s for s in sanitize_smiles_parallel(all_smiles) if s]
    canon_smi_ls = list(set(canon_smi_ls))

    if return_scores and fp_params is not None:
        scores = _tanimoto_scores_parallel(
            canon_smi_ls, smiles, fp_params.to_backend_opts()
        )
        return canon_smi_ls, scores

    return canon_smi_ls


def get_random_smiles(smi: str, num_random_samples: int) -> List[str]:
    """Obtain random SMILES orderings of a single SMILES."""
    raw = _randomize_smiles_parallel([smi], num_random_samples)
    return list(set([s for s in raw if s]))


def obtain_path(starting_smile: str, target_smile: str) -> List[str]:
    """
    Obtain a path from starting_smile to target_smile by greedily flipping
    differing SELFIES characters.
    """
    _require_selfies()
    starting_selfie = selfies.encoder(starting_smile)
    target_selfie = selfies.encoder(target_smile)

    starting_selfie_chars = get_selfie_chars(starting_selfie)
    target_selfie_chars = get_selfie_chars(target_selfie)

    # Pad the shorter string
    if len(starting_selfie_chars) < len(target_selfie_chars):
        for _ in range(len(target_selfie_chars) - len(starting_selfie_chars)):
            starting_selfie_chars.append(' ')
    else:
        for _ in range(len(starting_selfie_chars) - len(target_selfie_chars)):
            target_selfie_chars.append(' ')

    indices_diff = [
        i for i in range(len(starting_selfie_chars))
        if starting_selfie_chars[i] != target_selfie_chars[i]
    ]

    path_members = [starting_selfie_chars.copy()]
    current = starting_selfie_chars.copy()

    for _ in range(len(indices_diff)):
        if not indices_diff:
            break
        idx = np.random.choice(indices_diff, 1)[0]
        indices_diff.remove(idx)
        current[idx] = target_selfie_chars[idx]
        path_members.append(current.copy())

    # Collapse to SELFIES strings and decode
    path_smiles = []
    for member in path_members:
        selfie_str = ''.join(x for x in member).replace(' ', '')
        try:
            smi = selfies.decoder(selfie_str)
            mol = smi2mol(smi, sanitize=True)
            if mol is not None:
                smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
                if smi_canon:
                    path_smiles.append(smi_canon)
        except Exception:
            continue

    return path_smiles


def generate_pair_paths(
    smiles_list: List[str],
    num_tries: int = 2,
    num_random_samples: int = 2,
    collect_bidirectional: bool = True,
) -> List[str]:
    """
    Generate paths between exactly 2 SMILES.
    """
    if len(smiles_list) != 2:
        raise ValueError("Pair mode requires exactly 2 SMILES")

    start, target = smiles_list[0], smiles_list[1]

    start_rand = get_random_smiles(start, num_random_samples)
    target_rand = get_random_smiles(target, num_random_samples)

    all_smiles = []

    for smi_start in start_rand:
        for smi_target in target_rand:
            for _ in range(num_tries):
                path = obtain_path(smi_start, smi_target)
                all_smiles.extend(path)

    if collect_bidirectional:
        start_rand_2 = get_random_smiles(target, num_random_samples)
        target_rand_2 = get_random_smiles(start, num_random_samples)
        for smi_start in start_rand_2:
            for smi_target in target_rand_2:
                for _ in range(num_tries):
                    path = obtain_path(smi_start, smi_target)
                    all_smiles.extend(path)

    return list(set(all_smiles))


def form_joint_path(
    starting_selfie_chars: List[str],
    struct_2_selfie_chars: List[str],
    struct_3_selfie_chars: List[str],
) -> List[str]:
    """
    Create a generalized chemical path between three structures.
    """
    _require_selfies()
    best_median = starting_selfie_chars.copy()

    indices_diff_1 = [
        i for i in range(len(starting_selfie_chars))
        if starting_selfie_chars[i] != struct_2_selfie_chars[i]
    ]
    indices_diff_2 = [
        i for i in range(len(starting_selfie_chars))
        if starting_selfie_chars[i] != struct_3_selfie_chars[i]
    ]

    path_smiles = []

    while len(indices_diff_1) > 0 or len(indices_diff_2) > 0:
        # Mutation towards struct_2
        try:
            idx_1 = np.random.choice(indices_diff_1, 1)[0]
            indices_diff_1.remove(idx_1)
            median_1_sf = best_median.copy()
            median_1_sf[idx_1] = struct_2_selfie_chars[idx_1]
            median_1 = selfies.decoder(''.join(x for x in median_1_sf).strip())
            mol = smi2mol(median_1, sanitize=True)
            if mol is not None:
                smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
                if smi_canon:
                    path_smiles.append(smi_canon)
                    best_median = median_1_sf
                else:
                    indices_diff_1.append(idx_1)
            else:
                indices_diff_1.append(idx_1)
        except Exception:
            pass

        # Mutation towards struct_3
        try:
            idx_2 = np.random.choice(indices_diff_2, 1)[0]
            indices_diff_2.remove(idx_2)
            median_2_sf = best_median.copy()
            median_2_sf[idx_2] = struct_3_selfie_chars[idx_2]
            median_2 = selfies.decoder(''.join(x for x in median_2_sf).strip())
            mol = smi2mol(median_2, sanitize=True)
            if mol is not None:
                smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
                if smi_canon:
                    path_smiles.append(smi_canon)
                    best_median = median_2_sf
                else:
                    indices_diff_2.append(idx_2)
            else:
                indices_diff_2.append(idx_2)
        except Exception:
            pass

    return path_smiles


def generate_triplet_paths(
    smiles_list: List[str],
    num_paths: int = 100,
    num_random_samples: int = 1,
) -> List[str]:
    """
    Generate median molecules / generalized paths from all triplets.
    """
    if len(smiles_list) < 3:
        raise ValueError("Triplet mode requires at least 3 SMILES")

    all_smiles = []

    triplets = list(itertools.combinations(smiles_list, 3))

    for triplet in triplets:
        triplet = list(triplet)

        for _ in range(num_paths):
            triplet_rand = (
                get_random_smiles(triplet[0], num_random_samples)[0],
                get_random_smiles(triplet[1], num_random_samples)[0],
                get_random_smiles(triplet[2], num_random_samples)[0],
            )

            random_choice = np.random.choice([0, 1, 2], 3, replace=False)
            start = selfies.encoder(triplet_rand[random_choice[0]])
            s2 = selfies.encoder(triplet_rand[random_choice[1]])
            s3 = selfies.encoder(triplet_rand[random_choice[2]])

            start_chars = get_selfie_chars(start)
            s2_chars = get_selfie_chars(s2)
            s3_chars = get_selfie_chars(s3)

            max_len = max(len(start_chars), len(s2_chars), len(s3_chars))
            for chars in (start_chars, s2_chars, s3_chars):
                while len(chars) < max_len:
                    chars.append(' ')

            path = form_joint_path(start_chars.copy(), s2_chars.copy(), s3_chars.copy())
            all_smiles.extend(path)

    return list(set(all_smiles))
