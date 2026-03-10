import json
import pytest
import numpy as np
from virus_interactome.utils import (
    process_full_data_af3,
    process_full_data_boltz,
    process_full_data_colabfold,
    parse_msa_metrics,
    load_json,
    check_sequence_validity,
)


# ---------------------------------------------------------------------------
# Boltz — missing file tests
# ---------------------------------------------------------------------------

def test_missing_cif_file_boltz(dummy_confidences_boltz, dummy_pae_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(
            "nonexistent.cif",
            pae_file=str(dummy_pae_boltz),
            plddt_file=str(dummy_plddt_boltz),
            pde_file=str(dummy_pde_boltz),
            confidence_file=str(dummy_confidences_boltz),
        )


def test_missing_pae_file_boltz(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(
            str(dummy_cif_boltz),
            pae_file="missing.npz",
            plddt_file=str(dummy_plddt_boltz),
            pde_file=str(dummy_pde_boltz),
            confidence_file=str(dummy_confidences_boltz),
        )


def test_missing_plddt_file_boltz(dummy_cif_boltz, dummy_confidences_boltz, dummy_pae_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(
            str(dummy_cif_boltz),
            pae_file=str(dummy_pae_boltz),
            plddt_file="missing.npz",
            pde_file=str(dummy_pde_boltz),
            confidence_file=str(dummy_confidences_boltz),
        )


def test_missing_pde_file_boltz(dummy_cif_boltz, dummy_confidences_boltz, dummy_plddt_boltz, dummy_pae_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(
            str(dummy_cif_boltz),
            pae_file=str(dummy_pae_boltz),
            plddt_file=str(dummy_plddt_boltz),
            pde_file="missing.npz",
            confidence_file=str(dummy_confidences_boltz),
        )


def test_missing_confidence_file_boltz(dummy_cif_boltz, dummy_pae_boltz, dummy_plddt_boltz, dummy_pde_boltz):
    with pytest.raises(FileNotFoundError):
        process_full_data_boltz(
            str(dummy_cif_boltz),
            pae_file=str(dummy_pae_boltz),
            plddt_file=str(dummy_plddt_boltz),
            pde_file=str(dummy_pde_boltz),
            confidence_file="missing.json",
        )


# ---------------------------------------------------------------------------
# Boltz — shape mismatch tests
# ---------------------------------------------------------------------------

def test_mismatched_pae_shapes_boltz(dummy_cif_boltz, dummy_bad_pae_boltz):
    with pytest.raises(ValueError, match="PAE and PDE shapes do not match"):
        process_full_data_boltz(str(dummy_cif_boltz), pae_file=str(dummy_bad_pae_boltz))


def test_mismatched_plddt_shapes_boltz(dummy_cif_boltz, dummy_bad_plddt_boltz):
    with pytest.raises(ValueError, match="pLDDT length"):
        process_full_data_boltz(str(dummy_cif_boltz), plddt_file=str(dummy_bad_plddt_boltz))


# ---------------------------------------------------------------------------
# Boltz — full processing
# ---------------------------------------------------------------------------

def test_process_full_data_boltz_with_dummy_data(
    dummy_cif_boltz, dummy_confidences_boltz, dummy_pae_boltz, dummy_plddt_boltz, dummy_pde_boltz
):
    data = process_full_data_boltz(
        str(dummy_cif_boltz),
        confidence_file=str(dummy_confidences_boltz),
        pae_file=str(dummy_pae_boltz),
        plddt_file=str(dummy_plddt_boltz),
        pde_file=str(dummy_pde_boltz),
    )

    expected_pae = np.load(str(dummy_pae_boltz))["pae"]
    expected_res_plddt = np.load(str(dummy_plddt_boltz))["plddt"] * 100
    expected_chain_boundaries_by_res = {"A": (0, 249), "B": (250, 453)}
    expected_chain_boundaries_by_atom = {"A": (0, 1898), "B": (1899, 3517)}

    assert "pae" in data
    assert np.array_equal(data["pae"], expected_pae)
    assert "ca_plddts" in data
    assert "cb_plddts" in data
    assert np.array_equal(data["ca_plddts"], expected_res_plddt)
    assert "chain_boundaries_by_res" in data
    assert "token_chain_ids" in data
    assert data["chain_boundaries_by_res"] == expected_chain_boundaries_by_res
    assert data["chain_boundaries_by_atom"] == expected_chain_boundaries_by_atom
    assert "ptm" in data
    assert "iptm" in data


# ---------------------------------------------------------------------------
# AF3 — missing file tests
# ---------------------------------------------------------------------------

def test_missing_cif_file_af3(dummy_full_data_af3, dummy_summary_confidences_af3):
    with pytest.raises(FileNotFoundError):
        process_full_data_af3(
            "nonexistent.cif",
            json_path=str(dummy_full_data_af3),
            summary_json_path=str(dummy_summary_confidences_af3),
        )


def test_missing_full_data_json_af3(dummy_cif_af3, dummy_summary_confidences_af3):
    with pytest.raises(FileNotFoundError):
        process_full_data_af3(
            str(dummy_cif_af3),
            json_path="non_existent.json",
            summary_json_path=str(dummy_summary_confidences_af3),
        )


def test_missing_summary_json_af3(dummy_cif_af3, dummy_full_data_af3):
    with pytest.raises(FileNotFoundError):
        process_full_data_af3(
            str(dummy_cif_af3),
            json_path=str(dummy_full_data_af3),
            summary_json_path="non_existent.json",
        )


# ---------------------------------------------------------------------------
# AF3 — full processing
# ---------------------------------------------------------------------------

def test_process_full_data_af3_with_dummy_data(dummy_full_data_af3, dummy_summary_confidences_af3, dummy_cif_af3):
    data = process_full_data_af3(
        str(dummy_cif_af3),
        json_path=str(dummy_full_data_af3),
        summary_json_path=str(dummy_summary_confidences_af3),
    )

    full_data = load_json(dummy_full_data_af3)
    expected_pae = full_data["pae"]
    expected_atom_plddt = full_data["atom_plddts"]
    expected_chain_boundaries_by_res = {"A": (0, 249), "B": (250, 453)}
    expected_chain_boundaries_by_atom = {"A": (0, 1899), "B": (1900, 3519)}

    assert "pae" in data
    assert np.array_equal(data["pae"], expected_pae)
    assert "atom_plddts" in data
    assert np.array_equal(data["atom_plddts"], expected_atom_plddt)
    assert "cb_plddts" in data
    assert "chain_boundaries_by_res" in data
    assert data["chain_boundaries_by_res"] == expected_chain_boundaries_by_res
    assert data["chain_boundaries_by_atom"] == expected_chain_boundaries_by_atom
    assert "ptm" in data
    assert data["ptm"] == pytest.approx(0.54)
    assert "iptm" in data
    assert data["iptm"] == pytest.approx(0.86)
    assert "token_chain_ids" in data


# ---------------------------------------------------------------------------
# check_sequence_validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq,expected", [
    ("ACDEFGHIKLMNPQRSTVWY", True),
    ("ACDE", True),
    ("ACDZ", False),
    ("ACD*E", False),
    ("", False),
    ("acde", False),
])
def test_check_sequence_validity(seq, expected):
    assert check_sequence_validity(seq) == expected


# ---------------------------------------------------------------------------
# parse_msa_metrics
# ---------------------------------------------------------------------------

def test_parse_msa_metrics_missing_file():
    result = parse_msa_metrics("nonexistent.a3m")
    assert result == {"msa_depth": 0, "msa_coverage": 0.0}


def test_parse_msa_metrics_empty_file(tmp_path):
    a3m = tmp_path / "empty.a3m"
    a3m.write_text("")
    result = parse_msa_metrics(str(a3m))
    assert result == {"msa_depth": 0, "msa_coverage": 0.0}


def test_parse_msa_metrics_query_only(tmp_path):
    a3m = tmp_path / "query_only.a3m"
    a3m.write_text(">query\nMKTAYIAKQR\n")
    result = parse_msa_metrics(str(a3m))
    assert result["msa_depth"] == 0
    assert result["msa_coverage"] == 0.0


def test_parse_msa_metrics_with_hits(tmp_path):
    # query: 10 aa; hit1 has 2 gaps -> 8 non-gap / 10 = 0.8; hit2 same
    a3m = tmp_path / "hits.a3m"
    a3m.write_text(">query\nMKTAYIAKQR\n>hit1\nMKTAY--KQR\n>hit2\n--TAYIAKQR\n")
    result = parse_msa_metrics(str(a3m))
    assert result["msa_depth"] == 2
    assert result["msa_coverage"] == pytest.approx(0.8)


def test_parse_msa_metrics_full_coverage_hit(tmp_path):
    a3m = tmp_path / "full.a3m"
    a3m.write_text(">query\nAAAA\n>hit1\nAAAA\n>hit2\nAAA-\n")
    result = parse_msa_metrics(str(a3m))
    assert result["msa_depth"] == 2
    # hit1: 4/4=1.0, hit2: 3/4=0.75 → mean = 0.875
    assert result["msa_coverage"] == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# process_full_data_colabfold — missing file errors
# ---------------------------------------------------------------------------

def test_process_full_data_colabfold_missing_mol_file(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text("{}")
    with pytest.raises(FileNotFoundError):
        process_full_data_colabfold("nonexistent_unrelaxed_rank_001.pdb", str(scores))


def test_process_full_data_colabfold_missing_scores_json(dummy_cif_af3):
    with pytest.raises(FileNotFoundError):
        process_full_data_colabfold(str(dummy_cif_af3), "nonexistent_scores.json")


# ---------------------------------------------------------------------------
# process_full_data_colabfold — full processing
# ---------------------------------------------------------------------------

@pytest.fixture
def colabfold_scores_json(tmp_path):
    """Synthetic ColabFold scores JSON matching dummy_cif_af3 (454 CA residues)."""
    n_res = 454  # Chain A: 250, Chain B: 204
    rng = np.random.default_rng(0)
    scores = {
        "pae": rng.uniform(0, 15, (n_res, n_res)).tolist(),
        "plddt": (np.ones(n_res) * 75.0).tolist(),
        "ptm": 0.6,
        "iptm": 0.72,
    }
    path = tmp_path / "colabfold_scores.json"
    path.write_text(json.dumps(scores))
    return path


def test_process_full_data_colabfold_returns_expected_keys(dummy_cif_af3, colabfold_scores_json):
    data = process_full_data_colabfold(str(dummy_cif_af3), str(colabfold_scores_json))
    for key in ("pae", "atom_plddts", "ca_plddts", "cb_plddts",
                "chain_boundaries_by_res", "chain_boundaries_by_atom",
                "token_chain_ids", "ptm", "iptm", "iptm_chain_pair"):
        assert key in data, f"Missing key: {key}"


def test_process_full_data_colabfold_chain_boundaries(dummy_cif_af3, colabfold_scores_json):
    data = process_full_data_colabfold(str(dummy_cif_af3), str(colabfold_scores_json))
    assert data["chain_boundaries_by_res"] == {"A": (0, 249), "B": (250, 453)}


def test_process_full_data_colabfold_confidences(dummy_cif_af3, colabfold_scores_json):
    data = process_full_data_colabfold(str(dummy_cif_af3), str(colabfold_scores_json))
    assert data["ptm"] == pytest.approx(0.6)
    assert data["iptm"] == pytest.approx(0.72)
    assert data["ca_plddts"].shape == (454,)
    assert data["pae"].shape == (454, 454)
