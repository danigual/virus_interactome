import pytest
from pathlib import Path
from virus_interactome.proteome_input import load_proteome, create_af3_input_json_v2

@pytest.fixture
def dummy_fasta_path():
    return Path(__file__).parent / "data" / "dummy_proteome.fasta"

@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return Path(__file__).parent / "data" / "dummy_proteome_repeated_protein_id.fasta"

@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return Path(__file__).parent / "data" / "dummy_proteome_repeated_protein_id.fasta"

## Testing load_proteome function

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

## Testing create_af3_input_json_v2 function

def test_missing_protein_id_raises_keyerror():
    proteome = {"Protein1": "SEQ1"}
    with pytest.raises(KeyError):
        create_af3_input_json_v2("Protein1", "ProteinX", proteome_dict=proteome)

def test_empty_proteome_dict_raises_keyerror():
    with pytest.raises(KeyError):
        create_af3_input_json_v2("Protein1", proteome_dict={})

def test_non_string_proteome_label_raises_typeerror():
    proteome = {"Protein1": "SEQ1"}
    with pytest.raises(TypeError):
        create_af3_input_json_v2("Protein1", proteome_dict=proteome, proteome_label=["a", "b"])

def test_invalid_copy_number_raises_valueerror():
    proteome = {"Protein1": "SEQ1"}
    with pytest.raises(ValueError):
        create_af3_input_json_v2("Protein1", 0, proteome_dict=proteome)
    
def test_two_consecutive_integers_raises_valueerror():
    proteome = {"Protein1": "SEQ1"}
    with pytest.raises(ValueError):
        create_af3_input_json_v2("Protein1", 2, 3, proteome_dict=proteome)

### Testing corretness of output

def test_single_orf_default_copy():
    proteome = {"Protein1": "SEQ1"}
    result = create_af3_input_json_v2("Protein1", proteome_dict=proteome)
    assert result["name"] == "Protein1"
    assert result["sequences"] == [{
        "proteinChain": {
            "count": 1,
            "sequence": "SEQ1"
        }
    }]

def test_multiple_orfs_with_copies():
    proteome = {"Protein1": "SEQ1", "Protein2": "SEQ2"}
    result = create_af3_input_json_v2("Protein1", 2, "Protein2", 3, proteome_dict=proteome)
    assert result["name"] == "Protein1__Protein2"
    assert result["sequences"] == [
        {"proteinChain": {"count": 2, "sequence": "SEQ1"}},
        {"proteinChain": {"count": 3, "sequence": "SEQ2"}}
    ]

def test_proteome_label_in_name():
    proteome = {"Protein1": "SEQ1"}
    result = create_af3_input_json_v2("Protein1", proteome_dict=proteome, proteome_label="TestLabel")
    assert result["name"] == "TestLabel_Protein1"

def test_mixed_orfs_with_and_without_copies():
    proteome = {"Protein1": "SEQ1", "Protein2": "SEQ2", "Protein3": "SEQ3"}
    result = create_af3_input_json_v2("Protein1", "Protein2", 2, "Protein3", proteome_dict=proteome)
    assert result["name"] == "Protein1__Protein2__Protein3"
    assert result["sequences"] == [
        {"proteinChain": {"count": 1, "sequence": "SEQ1"}},
        {"proteinChain": {"count": 2, "sequence": "SEQ2"}},
        {"proteinChain": {"count": 1, "sequence": "SEQ3"}}
    ]
