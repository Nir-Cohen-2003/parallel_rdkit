from typing import List, Iterable
from itertools import batched, chain
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit import RDLogger
import polars as pl
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*') # type: ignore

def msready_smiles(smiles: str) -> str:
    """
    Transforms a SMILES string into an MS-Ready SMILES string.

    Workflow:
    1. Sanitize/Cleanup
    2. Metal Disconnection
    3. Normalization
    4. Reionization
    5. Fragment Removal (Salt Stripping)
    6. Charge Neutralization
    7. Tautomer Canonicalization
    """

    # 1. Load and Sanitize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 2. Metal Disconnection & Normalization & Reionization
    # The Cleanup function handles: MetalDisconnector, Normalizer, MetalDisconnector (again), and Reionizer
    clean_mol = rdMolStandardize.Cleanup(mol)

    # 3. Fragment Parent (Salt Stripping)
    # Returns the largest organic fragment
    frag_parent = rdMolStandardize.FragmentParent(clean_mol)

    # 4. Charge Parent (Neutralization)
    # Attempts to neutralize the molecule (handling zwitterions, quaternary ammoniums, etc.)
    # Note: This calls the Uncharger internally
    charge_parent = rdMolStandardize.ChargeParent(frag_parent)

    # 5. Tautomer Canonicalization
    # Enumerates tautomers and picks the canonical one using the scoring system
    te = rdMolStandardize.TautomerEnumerator()

    te.SetRemoveBondStereo(True)
    te.SetRemoveSp3Stereo(True)

    # Canonicalize returns the canonical tautomer molecule
    ms_ready_mol = te.Canonicalize(charge_parent)

    # 6. Final SMILES generation
    # isomericSmiles=True ensures stereochemistry is preserved if it survived the process
    return Chem.MolToSmiles(ms_ready_mol, isomericSmiles=false, canonical=True)


def sanitize_smiles_polars(smiles: pl.Series, batch_size: int = 10000) -> pl.Series:
    sanitized = sanitize_smiles(smiles, batch_size)
    return pl.Series(sanitized,dtype=pl.String)


def sanitize_smiles(smiles: Iterable[str], batch_size: int = 10000) -> List[str]:
    batches = list(batched(smiles, batch_size))
    with ProcessPoolExecutor() as executor:
        sanitized_batches = list(executor.map(_sanitize_smiles_batch, batches))
    return list(chain(*sanitized_batches))


def _sanitize_smiles_batch(smiles: List[str]) -> List[str]:
    RDLogger.DisableLog('rdApp.*') # type: ignore

    sanitized: list[str] = []
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s, sanitize=True)
            if mol is None:
                clean_smile = ''
            else:
                clean_smile = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except Exception as e:
            print(f"Error sanitizing SMILES {s}: {e}")
            clean_smile = ''
        sanitized.append(clean_smile)
    return sanitized


def inchi_to_smiles_polars(inchi: List[str], batch_size: int = 10000) -> pl.Series:
    smiles = inchi_to_smiles_list(inchi, batch_size)
    return pl.Series(smiles)


def inchi_to_smiles_list(inchi_list: List[str], batch_size: int = 10000) -> List[str]:
    """Convert a list of InChI strings to SMILES."""
    batches = list(batched(inchi_list, batch_size))
    with ProcessPoolExecutor() as executor:
        smiles_batches = list(executor.map(_inchi_to_smiles_batch, batches))
    return list(chain(*smiles_batches))


def _inchi_to_smiles_batch(inchi_list: List[str]) -> List[str]:
    """Convert a list of InChI strings to SMILES."""
    RDLogger.DisableLog('rdApp.*') # type: ignore

    smiles_list = []
    for inchi in inchi_list:
        try:
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                smiles = ''
            else:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except Exception as e:
            print(f"Error converting InChI {inchi}: {e}")
            smiles = ''
        smiles_list.append(smiles)
    return smiles_list


def smiles_to_inchi_polars(smiles: List[str], batch_size: int = 10000) -> pl.Series:
    inchi = smiles_to_inchi_list(smiles, batch_size)
    return pl.Series(inchi)


def smiles_to_inchi_list(smiles_list: List[str], batch_size: int = 10000) -> List[str]:
    """Convert a list of SMILES strings to InChI."""
    batches = list(batched(smiles_list, batch_size))
    with ProcessPoolExecutor() as executor:
        inchi_batches = list(executor.map(_smiles_to_inchi_batch, batches))
    return list(chain(*inchi_batches))


def _smiles_to_inchi_batch(smiles_list: List[str]) -> List[str]:
    """Convert a list of SMILES strings to InChI."""
    RDLogger.DisableLog('rdApp.*') # type: ignore

    inchi_list: List[str] = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                inchi = ''
            else:
                inchi = Chem.MolToInchi(mol)[0]
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {e}")
            inchi = ''
        inchi_list.append(inchi)
    return inchi_list



def smiles_to_inchikey_polars(smiles: List[str], batch_size: int = 10000) -> pl.Series:
    inchikey = smiles_to_inchikey_list(smiles, batch_size)
    return pl.Series(inchikey)


def smiles_to_inchikey_list(smiles_list: List[str], batch_size: int = 10000) -> List[str]:
    """Convert a list of SMILES strings to InChIKey."""
    batches = list(batched(smiles_list, batch_size))
    with ProcessPoolExecutor() as executor:
        inchikey_batches = list(executor.map(_smiles_to_inchikey_batch, batches))
    return list(chain(*inchikey_batches))


def _smiles_to_inchikey_batch(smiles_list: List[str]) -> List[str]:
    """Convert a list of SMILES strings to InChIKey."""
    RDLogger.DisableLog('rdApp.*') # type: ignore

    inchikey_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                inchikey = ''
            else:
                inchikey = Chem.MolToInchiKey(mol)
        except Exception as e:
            print(f"Error converting SMILES {smiles} to InChIKey: {e}")
            inchikey = ''
        inchikey_list.append(inchikey)
    return inchikey_list
