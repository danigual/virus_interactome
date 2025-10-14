import pytest
from pathlib import Path
from virus_interactome.proteome_input import load_proteome

@pytest.fixture
def dummy_fasta_path():
    return Path(__file__).parent / "data" / "dummy_proteome.fasta"

@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return Path(__file__).parent / "data" / "dummy_proteome_repeated_protein_id.fasta"

def test_load_proteome_missing_file():
    with pytest.raises(FileNotFoundError):
        load_proteome("non_existent_file.fasta")

def test_load_proteome_empty_file(tmp_path):
    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("")
    result = load_proteome(str(empty_file))
    assert result == {}

def test_load_proteome_basic(dummy_fasta_path):
    proteome = load_proteome(str(dummy_fasta_path))
    expected_proteome = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert isinstance(proteome, dict)
    assert proteome == expected_proteome

def test_load_proteome_with_repeated_protein_id(dummy_fasta_with_repeated_protein_id_path):
    
    with pytest.warns(UserWarning, match="Duplicate protein ID 'Protein1_isoformB' found. Overwriting previous entry."):
        proteome = load_proteome(str(dummy_fasta_with_repeated_protein_id_path))
    expected_proteome = {
        "Protein1_isoformB": "RQDILDMKTAYIAKQRQISFVKSHFS",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK"
    }
    assert isinstance(proteome, dict)
    assert proteome == expected_proteome



