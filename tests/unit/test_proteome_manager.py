import pytest
import logging
from virus_interactome.proteome_manager import ProteomeManager


# --- Loading tests ---

def test_load_proteome_missing_file():
    with pytest.raises(FileNotFoundError):
        ProteomeManager("non_existent_file.fasta")


def test_load_proteome_empty_file(tmp_path):
    pm = ProteomeManager()
    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("")
    result = pm.load_proteome(str(empty_file))
    assert result == {}


def test_load_proteome_basic(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    expected = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected
    assert pm.file_path == str(dummy_fasta_path)


def test_load_proteome_with_repeated_protein_id(dummy_fasta_with_repeated_protein_id_path, caplog):
    caplog.set_level(logging.WARNING)
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_fasta_with_repeated_protein_id_path))
    assert "Duplicate protein ID 'Protein1_isoformB' found" in caplog.text
    expected = {
        "Protein1_isoformB": "RQDILDMKTAYIAKQRQISFVKSHFS",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected


def test_load_non_fasta_file(dummy_non_fasta_proteome):
    pm = ProteomeManager(str(dummy_non_fasta_proteome))
    expected = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected


def test_mixed_sequences(dummy_proteome_with_invalid_sequences, caplog):
    caplog.set_level(logging.WARNING)
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_proteome_with_invalid_sequences))
    assert "Invalid amino acids in Protein1_isoformB. Sequence skipped." in caplog.text
    expected = {
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert pm.sequences == expected
    assert {"Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILDZZZ"} == pm.invalid_sequences


# --- Identity matrix tests ---

def test_compute_identity_matrix_empty_proteome():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_identity_matrix()


def test_compute_identity_matrix_single_protein():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "MKTAYIAKQR"}
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == pytest.approx(1.0)


def test_compute_identity_matrix_two_proteins():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT"}
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == pytest.approx(1.0)
    assert df.iloc[1, 1] == pytest.approx(1.0)
    assert df.iloc[0, 1] == pytest.approx(df.iloc[1, 0])
    assert 0.75 <= df.iloc[0, 1] <= 1.0


@pytest.mark.slow
def test_compute_identity_matrix_multiprocessing():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity_matrix(n_jobs=2)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == pytest.approx(1.0) for i in range(3))
    assert df.loc["Protein1", "Protein3"] == pytest.approx(0.0)
    assert df.loc["Protein3", "Protein1"] == pytest.approx(0.0)
    assert df.loc["Protein1", "Protein2"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein1"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein3"] == pytest.approx(0.25)
    assert df.loc["Protein3", "Protein2"] == pytest.approx(0.25)


@pytest.mark.slow
def test_compute_identity_matrix_with_similarity_threshold():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity_matrix(n_jobs=2, similarity_threshold=0.7)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == pytest.approx(1.0) for i in range(3))
    assert df.loc["Protein1", "Protein3"] == pytest.approx(0.0)
    assert df.loc["Protein1", "Protein2"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein3"] == pytest.approx(0.25)
    high_sim_pairs = pm.high_similarity_pairs
    assert ("Protein1", "Protein2", 0.75) in high_sim_pairs
