import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from parallel_rdkit import sanitize_smiles_parallel
from parallel_rdkit.fingerprint import FingerprintParams

try:
    from parallel_rdkit.stoned import (
        generate_local_space,
        generate_pair_paths,
        generate_triplet_paths,
        get_random_smiles,
        mutate_selfie,
        get_mutated_SELFIES,
        obtain_path,
        form_joint_path,
    )
    from parallel_rdkit.parallel_rdkit_backend import (
        randomize_smiles_parallel,
        tanimoto_scores_parallel,
    )
    _HAS_SELFIES = True
except ImportError:
    _HAS_SELFIES = False


pytestmark = pytest.mark.skipif(not _HAS_SELFIES, reason="selfies not installed")


class TestRandomizeSmilesParallel:
    def test_basic_randomization(self):
        smiles = ["CCO", "c1ccccc1", "CC(C)CCO"]
        num_samples = 10
        result = randomize_smiles_parallel(smiles, num_samples)
        assert len(result) == len(smiles) * num_samples

        # Every result should be a valid SMILES or empty string
        for smi in result:
            if smi:
                mol = Chem.MolFromSmiles(smi)
                assert mol is not None, f"Invalid SMILES produced: {smi}"

    def test_single_input(self):
        result = randomize_smiles_parallel(["CCO"], 5)
        assert len(result) == 5
        valid = [s for s in result if s]
        assert len(valid) > 0

    def test_invalid_smiles(self):
        result = randomize_smiles_parallel(["NOT_A_SMILES"], 3)
        assert len(result) == 3
        assert all(s == "" for s in result)

    def test_randomness(self):
        # With a real molecule, randomized variants should differ from each other
        result = randomize_smiles_parallel(["c1ccccc1"], 20)
        valid = [s for s in result if s]
        assert len(set(valid)) > 1, "Randomized SMILES should produce different strings"


class TestTanimotoScoresParallel:
    def _reference_scores(self, smiles_list, target_smi, fp_type="morgan", radius=2, fpSize=2048):
        """Compute reference scores using pure RDKit."""
        target = Chem.MolFromSmiles(target_smi)
        if fp_type == "morgan":
            fp_target = AllChem.GetMorganFingerprintAsBitVect(target, radius, nBits=fpSize)
        elif fp_type == "rdkit":
            fp_target = Chem.RDKFingerprint(target)
        else:
            raise ValueError(f"Unsupported fp_type for reference: {fp_type}")

        scores = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scores.append(0.0)
                continue
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fpSize)
            elif fp_type == "rdkit":
                fp = Chem.RDKFingerprint(mol)
            scores.append(TanimotoSimilarity(fp, fp_target))
        return scores

    def test_morgan_tanimoto(self):
        target = "c1ccccc1"
        queries = ["c1ccccc1", "CCO", "c1ccccc1O", "CC(C)C", "INVALID"]

        fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
        scores = tanimoto_scores_parallel(
            queries, target, fp_params.to_backend_opts()
        )

        ref_scores = self._reference_scores(queries, target, fp_type="morgan", radius=2, fpSize=2048)

        assert len(scores) == len(queries)
        for i, (s, r) in enumerate(zip(scores, ref_scores)):
            assert abs(s - r) < 1e-5, f"Score mismatch at index {i}: got {s}, expected {r}"

    def test_self_similarity(self):
        # A molecule should have perfect Tanimoto similarity with itself
        target = "CC(C)CCO"
        queries = [target]
        fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
        scores = tanimoto_scores_parallel(
            queries, target, fp_params.to_backend_opts()
        )
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_invalid_target(self):
        scores = tanimoto_scores_parallel(
            ["CCO"], "INVALID_SMILES", FingerprintParams().to_backend_opts()
        )
        assert scores[0] == 0.0

    def test_empty_queries(self):
        scores = tanimoto_scores_parallel(
            [], "CCO", FingerprintParams().to_backend_opts()
        )
        assert scores == []


class TestStonedGeneration:
    def test_generate_local_space(self):
        np.random.seed(42)
        result = generate_local_space(
            "CCO",
            num_random_samples=50,
            num_mutation_ls=[1, 2],
        )
        assert isinstance(result, list)
        assert len(result) > 0
        # All results should be valid canonical SMILES
        for smi in result:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None

    def test_generate_local_space_with_scores(self):
        np.random.seed(42)
        fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
        smiles, scores = generate_local_space(
            "CCO",
            num_random_samples=50,
            num_mutation_ls=[1],
            fp_params=fp_params,
            return_scores=True,
        )
        assert len(smiles) == len(scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_generate_pair_paths(self):
        np.random.seed(42)
        result = generate_pair_paths(
            ["CCO", "CCCO"],
            num_tries=2,
            num_random_samples=2,
            collect_bidirectional=True,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        # Both endpoints should ideally be present or at least close variants
        for smi in result:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None

    def test_generate_triplet_paths(self):
        np.random.seed(42)
        result = generate_triplet_paths(
            ["CCO", "CCCO", "CCCCO"],
            num_paths=5,
            num_random_samples=1,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        for smi in result:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None

    def test_mutate_selfie(self):
        np.random.seed(42)
        selfie = selfies.encoder("CCO")
        mutated, smiles_canon = mutate_selfie(selfie, max_molecules_len=20)
        assert mutated != selfie or smiles_canon == "CCO"  # May mutate or stay same by chance
        assert Chem.MolFromSmiles(smiles_canon) is not None

    def test_obtain_path(self):
        np.random.seed(42)
        path = obtain_path("CCO", "CCCO")
        assert len(path) > 0
        # Path should contain valid SMILES
        for smi in path:
            assert Chem.MolFromSmiles(smi) is not None

    def test_get_random_smiles(self):
        result = get_random_smiles("CCO", 20)
        assert len(result) > 0
        assert len(result) <= 20
        for smi in result:
            assert Chem.MolFromSmiles(smi) is not None


# Ensure selfies is imported for the test classes above
if _HAS_SELFIES:
    import selfies
