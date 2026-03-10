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


# --- ids property ---

def test_ids_insertion_order():
    pm = ProteomeManager()
    pm.sequences = {"C": "AAA", "A": "GGG", "B": "TTT"}
    assert pm.ids == ("C", "A", "B")


def test_ids_cached():
    pm = ProteomeManager()
    pm.sequences = {"P1": "AAA", "P2": "GGG"}
    first = pm.ids
    second = pm.ids
    assert first is second


# --- filter_by_regex ---

def test_filter_by_regex_ids_only(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"isoform")
    assert isinstance(result, list)
    assert "Protein1_isoformB" in result
    assert "Protein4_isoformC" in result
    assert "Protein2" not in result


def test_filter_by_regex_return_sequences(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"variant", return_sequences=True)
    assert isinstance(result, dict)
    assert "Protein3_variantA" in result
    assert result["Protein3_variantA"] == "LLKSDGQVLKAV"


def test_filter_by_regex_no_matches(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"NOMATCH_XYZ_999")
    assert result == []


# --- compute_properties ---

def test_compute_properties_empty_raises():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_properties()


def test_compute_properties_columns(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    df = pm.compute_properties()
    expected_cols = {"length", "molecular_weight", "isoelectric_point",
                     "instability_index", "gravy", "aromaticity"}
    assert expected_cols.issubset(set(df.columns))


def test_compute_properties_values(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    df = pm.compute_properties()
    assert df.index.name == "id"
    assert set(df.index) == set(pm.sequences.keys())
    # Sanity: lengths match sequence lengths
    for pid, seq in pm.sequences.items():
        assert df.loc[pid, "length"] == len(seq)
    # MW and pI are positive numbers
    assert (df["molecular_weight"] > 0).all()
    assert (df["isoelectric_point"] > 0).all()


# --- summary ---

def test_summary_dict(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.summary()
    assert isinstance(result, dict)
    expected_keys = {"total_sequences", "total_residues", "average_length",
                     "min_length", "max_length", "invalid_sequences", "high_similarity_pairs"}
    assert expected_keys == set(result.keys())
    assert result["total_sequences"] == 4
    assert result["min_length"] <= result["average_length"] <= result["max_length"]


def test_summary_empty_proteome():
    pm = ProteomeManager()
    result = pm.summary()
    assert result["total_sequences"] == 0
    assert result["total_residues"] == 0
    assert result["min_length"] == 0


# --- get_sequence ---

def test_get_sequence_found(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    seq = pm.get_sequence("Protein2")
    assert seq == "GVALSKGEEAVRLFK"


def test_get_sequence_not_found(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    with pytest.raises(KeyError):
        pm.get_sequence("NONEXISTENT_PROTEIN")


# --- __str__ ---

def test_str_representation(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    s = str(pm)
    assert "ProteomeManager Summary" in s
    assert "Total sequences" in s


# --- sorted ids mode ---

def test_ids_sorted_order():
    pm = ProteomeManager()
    pm.sequences = {"C": "AAA", "A": "GGG", "B": "TTT"}
    pm._order_mode = "sorted"
    pm._ids_cache = None  # force recompute
    assert pm.ids == ("A", "B", "C")


# --- __len__ ---

def test_len_returns_sequence_count(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    assert len(pm) == 4


def test_len_empty_proteome():
    pm = ProteomeManager()
    assert len(pm) == 0


# --- get_ids ---

def test_get_ids(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    ids = pm.get_ids()
    assert isinstance(ids, list)
    assert set(ids) == {"Protein1_isoformB", "Protein2", "Protein3_variantA", "Protein4_isoformC"}


# --- _compute_identity static method ---

def test_compute_identity_identical():
    i, j, score = ProteomeManager._compute_identity((0, 1, "AAAA", "AAAA"))
    assert i == 0
    assert j == 1
    assert score == pytest.approx(1.0)


def test_compute_identity_different():
    i, j, score = ProteomeManager._compute_identity((0, 1, "AAAA", "TTTT"))
    assert score == pytest.approx(0.0)
