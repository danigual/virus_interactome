import pytest
from pathlib import Path
from virus_interactome.utils import process_full_data_af3, process_full_data_boltz
import json
import numpy as np

## AF3 dummy data fixtures
@pytest.fixture
def dummy_mol_path():
    return Path(__file__).parent / "data" / "dummy_dimer_model_0.cif"

@pytest.fixture
def dummy_summary_confidences_af3():
    return Path(__file__).parent / "data" / "dummy_dimer_full_data_0.json"

@pytest.fixture
def dummy_full_data_af3():
    return Path(__file__).parent / "data" / "dummy_dimer_full_data_0.json"

## Boltz dummy data fixtures
@pytest.fixture
def dummy_confidences_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "confidence_pvi__protease_model_0.json"

@pytest.fixture
def dummy_cif_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "pvi__protease_model_0.cif"

@pytest.fixture
def dummy_pae_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "pae_pvi__protease_model_0.npz"

@pytest.fixture
def dummy_bad_pae_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "bad_pae_pvi__protease_model_0.npz"

@pytest.fixture
def dummy_plddt_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "plddt_pvi__protease_model_0.npz"

@pytest.fixture
def dummy_bad_plddt_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "bad_plddt_pvi__protease_model_0.npz"

@pytest.fixture
def dummy_pde_boltz():
    return Path(__file__).parent / "data" / "boltz_dummy_example" / "pde_pvi__protease_model_0.npz"

# --- Test: Missing CIF file ---
def test_missing_cif_file(dummy_confidences_boltz, dummy_pae_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz("nonexistent.cif", pae_file=str(dummy_pae_boltz), plddt_file=str(dummy_plddt_boltz), pde_file=str(dummy_pde_boltz), confidence_file=str(dummy_confidences_boltz))

# --- Test: Missing PAE file ---
def test_missing_pae_file(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(dummy_cif_boltz, pae_file="missing.npz", plddt_file=dummy_plddt_boltz, pde_file=dummy_pde_boltz, confidence_file=dummy_confidences_boltz)

# # --- Test: Missing PLDDT file ---
def test_missing_pae_file(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(str(dummy_cif_boltz), pae_file=str(dummy_pae_boltz), plddt_file="missing.npz", pde_file=str(dummy_pde_boltz), confidence_file=str(dummy_confidences_boltz))

# --- Test: Missing PDE file ---
def test_missing_pde_file(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(str(dummy_cif_boltz), pae_file=str(dummy_pae_boltz), plddt_file=str(dummy_plddt_boltz), pde_file="missing.npz", confidence_file=str(dummy_confidences_boltz))

# --- Test: Missing JSON file ---
def test_missing_confidence_file(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(str(dummy_cif_boltz), pae_file=str(dummy_pae_boltz), plddt_file=str(dummy_plddt_boltz), pde_file=str(dummy_pde_boltz), confidence_file="missing.json")

# --- Test: Mismatched PAE shapes ---
def test_mismatched_pae_shapes(dummy_cif_boltz, dummy_bad_pae_boltz):
    with pytest.raises(ValueError, match="PAE and PDE shapes do not match"):
        process_full_data_boltz(str(dummy_cif_boltz), pae_file=str(dummy_bad_pae_boltz))

# --- Test: Mismatched PLDDT shapes ---
def test_mismatched_plddt_shapes(dummy_cif_boltz, dummy_bad_plddt_boltz):
    with pytest.raises(ValueError, match="pLDDT length"):
        process_full_data_boltz(str(dummy_cif_boltz), plddt_file=str(dummy_bad_plddt_boltz))

def test_process_full_data_boltz_with_dummy_data(
    dummy_cif_boltz,
    dummy_confidences_boltz,
    dummy_pae_boltz,
    dummy_plddt_boltz,
    dummy_pde_boltz
):
    data = process_full_data_boltz(
        str(dummy_cif_boltz),
        confidence_file=str(dummy_confidences_boltz),
        pae_file=str(dummy_pae_boltz),
        plddt_file=str(dummy_plddt_boltz),
        pde_file=str(dummy_pde_boltz)
    )

    expected_pae = np.load(str(dummy_pae_boltz))["pae"]
    expected_res_plddt = np.load(str(dummy_plddt_boltz))["plddt"]
    expected_chain_boundaries_by_res = {"A": (0, 249), "B": (250, 453)}    
    expected_chain_boundaries_by_atom = {"A": (0, 1898), "B": (1899, 3517)}    

    assert "pae" in data
    assert np.array_equal(data["pae"], expected_pae)
    assert "atom_plddts" in data
    assert np.array_equal(data["res_plddts"], expected_res_plddt)
    assert "res_plddts" in data
    assert "chain_boundaries_by_res" in data
    assert "token_chain_ids" in data    
    assert data["chain_boundaries_by_res"] == expected_chain_boundaries_by_res
    assert data["chain_boundaries_by_atom"] == expected_chain_boundaries_by_atom
    assert "ptm" in data
    assert "iptm" in data
    
# def test_process_full_data_af3_with_empty_file(tmp_path):
#     pass

# def test_process_full_data_af3_with_dummy_data(dummy_full_data_af3):
#     pass

# def test_process_full_data_boltz_with_empty_file(tmp_path):
#     pass

# def test_process_full_data_boltz_with_dummy_data(dummy_mol_path)::
    # pass