"""Shared fixtures for virus_interactome test suite."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def dummy_fasta_path():
    return DATA_DIR / "dummy_proteome.fasta"


@pytest.fixture
def dummy_fasta_with_repeated_protein_id_path():
    return DATA_DIR / "dummy_proteome_repeated_protein_id.fasta"


@pytest.fixture
def dummy_non_fasta_proteome():
    return DATA_DIR / "dummy_proteome_non_fasta.fasta"


@pytest.fixture
def dummy_proteome_with_invalid_sequences():
    return DATA_DIR / "dummy_proteome_invalid_sequences.fasta"


# AF3 dummy data
@pytest.fixture
def dummy_cif_af3():
    return DATA_DIR / "af3_dummy_example" / "fold_adv5_pvi_protease_model_0.cif"


@pytest.fixture
def dummy_full_data_af3():
    return DATA_DIR / "af3_dummy_example" / "fold_adv5_pvi_protease_full_data_0.json"


@pytest.fixture
def dummy_summary_confidences_af3():
    return DATA_DIR / "af3_dummy_example" / "fold_adv5_pvi_protease_summary_confidences_0.json"


# Boltz dummy data
@pytest.fixture
def dummy_cif_boltz():
    return DATA_DIR / "boltz_dummy_example" / "pvi__protease_model_0.cif"


@pytest.fixture
def dummy_confidences_boltz():
    return DATA_DIR / "boltz_dummy_example" / "confidence_pvi__protease_model_0.json"


@pytest.fixture
def dummy_pae_boltz():
    return DATA_DIR / "boltz_dummy_example" / "pae_pvi__protease_model_0.npz"


@pytest.fixture
def dummy_bad_pae_boltz():
    return DATA_DIR / "boltz_dummy_example" / "bad_pae_pvi__protease_model_0.npz"


@pytest.fixture
def dummy_plddt_boltz():
    return DATA_DIR / "boltz_dummy_example" / "plddt_pvi__protease_model_0.npz"


@pytest.fixture
def dummy_bad_plddt_boltz():
    return DATA_DIR / "boltz_dummy_example" / "bad_plddt_pvi__protease_model_0.npz"


@pytest.fixture
def dummy_pde_boltz():
    return DATA_DIR / "boltz_dummy_example" / "pde_pvi__protease_model_0.npz"


# Generic CIF/JSON for metrics tests
@pytest.fixture
def dummy_mol_path():
    return DATA_DIR / "dummy_dimer_model_0.cif"


@pytest.fixture
def dummy_json_path():
    return DATA_DIR / "dummy_dimer_full_data_0.json"


# ---------------------------------------------------------------------------
# Synthetic DataFrames for InteractomeAnalyzer
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_interactome_df():
    """Synthetic interactome DataFrame with ~10 PPIs and realistic metric ranges."""
    np.random.seed(42)
    ppis = [
        "ProtA__ProtB", "ProtA__ProtC", "ProtA__ProtD",
        "ProtB__ProtC", "ProtB__ProtD", "ProtB__ProtE",
        "ProtC__ProtD", "ProtC__ProtE",
        "ProtD__ProtE", "ProtD__ProtF",
    ]
    n = len(ppis)
    return pd.DataFrame({
        "PPI": ppis,
        "Folder": [f"/fake/path/{p}" for p in ppis],
        "ipSAE_AB": np.round(np.random.uniform(0.1, 0.9, n), 3),
        "pDockQ2_AB": np.round(np.random.uniform(0.05, 0.6, n), 3),
        "pLDDT_mean": np.round(np.random.uniform(40, 95, n), 1),
        "msa_depth": np.random.randint(1, 100, n),
        "ipTM": np.round(np.random.uniform(0.1, 0.9, n), 3),
        "pTM": np.round(np.random.uniform(0.3, 0.9, n), 3),
        "Best_iLIS": np.round(np.random.uniform(0.05, 0.65, n), 3),
    })


@pytest.fixture
def dummy_interactome_csv(tmp_path, dummy_interactome_df):
    """Writes dummy_interactome_df to a temp CSV and returns the path."""
    csv_path = tmp_path / "interactome_data.csv"
    dummy_interactome_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def dummy_cluster_df():
    """Synthetic cluster DataFrame with ~15 rows."""
    rows = []
    ppis = ["ProtA__ProtB", "ProtA__ProtC", "ProtB__ProtC", "ProtC__ProtD", "ProtD__ProtE"]
    for i, ppi in enumerate(ppis):
        for cid in range(3):
            rows.append({
                "PPI": ppi,
                "path": f"/fake/models/{ppi}/model_{cid}.cif",
                "cluster_id": cid,
                "cluster_ratio": np.round(np.random.uniform(1.0, 15.0), 2),
                "n_residues": np.random.randint(10, 200),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_cluster_csv(tmp_path, dummy_cluster_df):
    """Writes dummy_cluster_df to a temp CSV and returns the path."""
    csv_path = tmp_path / "clusters_data.csv"
    dummy_cluster_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def analyzer_with_data(dummy_interactome_csv, dummy_cluster_csv):
    """Returns an InteractomeAnalyzer with both datasets auto-loaded from tmp_path."""
    from virus_interactome.interactome_analyzer import InteractomeAnalyzer
    return InteractomeAnalyzer(output_path=dummy_interactome_csv.parent)
