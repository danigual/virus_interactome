import pytest
from pathlib import Path
import logging
# from virus_interactome.proteome_input import load_proteome, create_af3_input_json_v2

from virus_interactome.proteome_manager import ProteomeManager
@pytest.fixture
def dummy_fasta_path():
    return Path(__file__).parent / "data" / "dummy_proteome.fasta"

@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return Path(__file__).parent / "data" / "dummy_proteome_repeated_protein_id.fasta"

@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return Path(__file__).parent / "data" / "dummy_proteome_repeated_protein_id.fasta"

@pytest.fixture
def dummy_non_fasta_proteome():
    return Path(__file__).parent / "data" / "dummy_proteome_non_fasta.fasta"

@pytest.fixture
def dummy_proteome_with_invalid_sequences():
    return Path(__file__).parent / "data" / "dummy_proteome_invalid_sequences.fasta"

## Testing load_proteome function

def test_load_proteome_missing_file():
    with pytest.raises(FileNotFoundError):
        pm = ProteomeManager("non_existent_file.fasta")

def test_load_proteome_empty_file(tmp_path):
    pm = ProteomeManager()
    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("")
    result =  pm.load_proteome(str(empty_file))
    assert result == {}

def test_load_proteome_basic(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    expected_proteome = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected_proteome
    assert pm.file_path == str(dummy_fasta_path)

def test_load_proteome_with_repeated_protein_id(dummy_fasta_with_repeated_protein_id_path, caplog):
    caplog.set_level(logging.WARNING)  # Capture WARNING level logs
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_fasta_with_repeated_protein_id_path))
    assert "Duplicate protein ID 'Protein1_isoformB' found" in caplog.text
    expected_proteome = {
        "Protein1_isoformB": "RQDILDMKTAYIAKQRQISFVKSHFS",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected_proteome

def test_load_non_fasta_file(dummy_non_fasta_proteome):
    pm = ProteomeManager(str(dummy_non_fasta_proteome))
    expected_proteome = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected_proteome


def test_mixed_sequences(dummy_proteome_with_invalid_sequences, caplog):
    caplog.set_level(logging.WARNING)  # Capture WARNING level logs
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_proteome_with_invalid_sequences))
    assert "Invalid amino acids in Protein1_isoformB. Sequence skipped." in caplog.text
    expected_proteome = {
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert pm.sequences == expected_proteome
    assert {"Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILDZZZ"} == pm.invalid_sequences

## Testing computing identity matrix
def test_compute_identity_matrix_empty_proteome():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_identity_matrix()


def test_compute_identity_matrix_single_protein():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "MKTAYIAKQR"}
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == 1.0


def test_compute_identity_matrix_two_proteins():
    pm = ProteomeManager()
    pm.sequences = {
        "Protein1": "AAAA",
        "Protein2": "AAAT"
    }
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == 1.0
    assert df.iloc[1, 1] == 1.0
    assert df.iloc[0, 1] == df.iloc[1, 0]
    assert 0.75 <= df.iloc[0, 1] <= 1.0

def test_compute_identity_matrix_multiprocessing():
    pm = ProteomeManager()
    pm.sequences = {
        "Protein1": "AAAA",
        "Protein2": "AAAT",
        "Protein3": "TTTT"
    }
    df = pm.compute_identity_matrix(n_jobs=2)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == 1.0 for i in range(3))
    assert all(df.iloc[i, i] == 1.0 for i in range(3))
    assert(df.loc["Protein1", "Protein3"] == 0.0)
    assert(df.loc["Protein3", "Protein1"] == 0.0)
    assert(df.loc["Protein1", "Protein2"] == 0.75)
    assert(df.loc["Protein2", "Protein1"] == 0.75)
    assert(df.loc["Protein2", "Protein3"] == 0.25)
    assert(df.loc["Protein3", "Protein2"] == 0.25)

def test_compute_identity_matrix_multiprocessing_with_similarity_threshold():
    pm = ProteomeManager()
    pm.sequences = {
        "Protein1": "AAAA",
        "Protein2": "AAAT",
        "Protein3": "TTTT"
    }
    df = pm.compute_identity_matrix(n_jobs=2, similarity_threshold=0.7)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == 1.0 for i in range(3))
    assert all(df.iloc[i, i] == 1.0 for i in range(3))
    assert(df.loc["Protein1", "Protein3"] == 0.0)
    assert(df.loc["Protein3", "Protein1"] == 0.0)
    assert(df.loc["Protein1", "Protein2"] == 0.75)
    assert(df.loc["Protein2", "Protein1"] == 0.75)
    assert(df.loc["Protein2", "Protein3"] == 0.25)
    assert(df.loc["Protein3", "Protein2"] == 0.25)
    high_sim_pairs = pm.high_similarity_pairs
    assert ("Protein1", "Protein2", 0.75) in high_sim_pairs

# ## Testing create_af3_input_json_v2 function

# def test_missing_protein_id_raises_keyerror():
#     proteome = {"Protein1": "SEQ1"}
#     with pytest.raises(KeyError):
#         create_af3_input_json_v2("Protein1", "ProteinX", proteome_dict=proteome)

# def test_empty_proteome_dict_raises_keyerror():
#     with pytest.raises(KeyError):
#         create_af3_input_json_v2("Protein1", proteome_dict={})

# def test_non_string_proteome_label_raises_typeerror():
#     proteome = {"Protein1": "SEQ1"}
#     with pytest.raises(TypeError):
#         create_af3_input_json_v2("Protein1", proteome_dict=proteome, proteome_label=["a", "b"])

# def test_invalid_copy_number_raises_valueerror():
#     proteome = {"Protein1": "SEQ1"}
#     with pytest.raises(ValueError):
#         create_af3_input_json_v2("Protein1", 0, proteome_dict=proteome)
    
# def test_two_consecutive_integers_raises_valueerror():
#     proteome = {"Protein1": "SEQ1"}
#     with pytest.raises(ValueError):
#         create_af3_input_json_v2("Protein1", 2, 3, proteome_dict=proteome)

# ### Testing correctness of output
# def test_single_orf_default_copy():
#     proteome = {"Protein1": "SEQ1"}
#     result = create_af3_input_json_v2("Protein1", proteome_dict=proteome)
#     assert result["name"] == "Protein1"
#     assert result["sequences"] == [{
#         "proteinChain": {
#             "count": 1,
#             "sequence": "SEQ1"
#         }
#     }]

# def test_multiple_orfs_with_copies():
#     proteome = {"Protein1": "SEQ1", "Protein2": "SEQ2"}
#     result = create_af3_input_json_v2("Protein1", 2, "Protein2", 3, proteome_dict=proteome)
#     assert result["name"] == "Protein1__Protein2"
#     assert result["sequences"] == [
#         {"proteinChain": {"count": 2, "sequence": "SEQ1"}},
#         {"proteinChain": {"count": 3, "sequence": "SEQ2"}}
#     ]

# def test_proteome_label_in_name():
#     proteome = {"Protein1": "SEQ1"}
#     result = create_af3_input_json_v2("Protein1", proteome_dict=proteome, prefix="TestLabel")
#     assert result["name"] == "TestLabel_Protein1"

# def test_mixed_orfs_with_and_without_copies():
#     proteome = {"Protein1": "SEQ1", "Protein2": "SEQ2", "Protein3": "SEQ3"}
#     result = create_af3_input_json_v2("Protein1", "Protein2", 2, "Protein3", proteome_dict=proteome)
#     assert result["name"] == "Protein1__Protein2__Protein3"
#     assert result["sequences"] == [
#         {"proteinChain": {"count": 1, "sequence": "SEQ1"}},
#         {"proteinChain": {"count": 2, "sequence": "SEQ2"}},
#         {"proteinChain": {"count": 1, "sequence": "SEQ3"}}
#     ]
