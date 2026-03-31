import pytest
import json
import numpy as np
import pandas as pd
from virus_interactome.metrics import (
    calculate_pdockq, calculate_pdockq2,
    calculate_LIS, calculate_LIS_family, calculate_ipsae,
)
from virus_interactome.utils import process_full_data_af3


def test_calculate_pdockq(dummy_mol_path):
    full_data = process_full_data_af3(str(dummy_mol_path))
    plddt_by_res = np.array(full_data["cb_plddts"])
    result = calculate_pdockq(str(dummy_mol_path), plddt_by_res)

    expected = pd.DataFrame({"chain1": ["A"], "chain2": ["B"], "pDockQ": [0.4685032816442164]})
    pd.testing.assert_frame_equal(result, expected, atol=1e-6)


def test_calculate_pdockq2(dummy_mol_path):
    full_data = process_full_data_af3(str(dummy_mol_path))
    plddt_by_res = np.array(full_data["cb_plddts"])
    pae = np.array(full_data["pae"])
    result = calculate_pdockq2(str(dummy_mol_path), plddt_by_res, pae)

    expected = pd.DataFrame({
        "chain1": ["A", "B"],
        "chain2": ["B", "A"],
        "pDockQ2": [0.42800951458731734, 0.38320326568380014],
    })
    pd.testing.assert_frame_equal(result, expected, atol=1e-6)


def test_calculate_LIS(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, "r") as f:
        full_data = json.load(f)
    pae = np.array(full_data["pae"])
    result = calculate_LIS(str(dummy_mol_path), pae)

    expected = pd.DataFrame({
        "chain1": ["A", "B"],
        "chain2": ["B", "A"],
        "LIS": [0.46360432330827067, 0.4772722012190417],
    })
    pd.testing.assert_frame_equal(result, expected, atol=1e-6)


def test_calculate_LIS_family_columns(dummy_mol_path, dummy_json_path):
    """calculate_LIS_family returns all expected columns."""
    with open(dummy_json_path) as f:
        full_data = json.load(f)
    pae = np.array(full_data["pae"])
    result = calculate_LIS_family(str(dummy_mol_path), pae)

    expected_cols = {"chain1", "chain2", "LIS", "LIA", "cLIS", "cLIA", "iLIS", "iLIA", "LIR", "cLIR"}
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == 2  # A→B and B→A


def test_calculate_LIS_family_values(dummy_mol_path, dummy_json_path):
    """Numeric values for LIS family match expected output."""
    with open(dummy_json_path) as f:
        full_data = json.load(f)
    pae = np.array(full_data["pae"])
    result = calculate_LIS_family(str(dummy_mol_path), pae)

    ab = result.loc[(result.chain1 == "A") & (result.chain2 == "B")].iloc[0]
    ba = result.loc[(result.chain1 == "B") & (result.chain2 == "A")].iloc[0]

    # LIS matches backwards-compatible wrapper
    assert abs(ab["LIS"] - 0.463604) < 1e-4
    assert abs(ba["LIS"] - 0.477272) < 1e-4

    # LIA (count of PAE≤12 pairs) is positive
    assert ab["LIA"] == 13832
    assert ba["LIA"] == 12961

    # cLIA ≤ LIA (contact filter is stricter)
    assert ab["cLIA"] <= ab["LIA"]
    assert ba["cLIA"] <= ba["LIA"]
    assert ab["cLIA"] == 77
    assert ba["cLIA"] == 76

    # cLIS ≥ LIS (contact pairs have lower PAE on average → higher score)
    assert ab["cLIS"] >= ab["LIS"]
    assert ba["cLIS"] >= ba["LIS"]

    # iLIS = sqrt(LIS * cLIS)
    assert abs(ab["iLIS"] - np.sqrt(ab["LIS"] * ab["cLIS"])) < 1e-8
    assert abs(ba["iLIS"] - np.sqrt(ba["LIS"] * ba["cLIS"])) < 1e-8

    # iLIA = sqrt(LIA * cLIA)
    assert abs(ab["iLIA"] - np.sqrt(ab["LIA"] * ab["cLIA"])) < 1e-6


def test_calculate_LIS_family_ilis_zero_when_no_contacts(dummy_mol_path):
    """iLIS is 0 when PAE matrix has no contacts (all PAE > cutoff)."""
    with open(dummy_mol_path.parent / (dummy_mol_path.stem.replace("_model_", "_full_data_") + ".json")) as f:
        full_data = json.load(f)
    pae_no_contacts = np.full_like(np.array(full_data["pae"]), fill_value=20.0)
    result = calculate_LIS_family(str(dummy_mol_path), pae_no_contacts)
    assert (result["iLIS"] == 0.0).all()
    assert (result["LIS"] == 0.0).all()
    assert (result["LIA"] == 0).all()


def test_calculate_LIS_backwards_compat(dummy_mol_path, dummy_json_path):
    """calculate_LIS wrapper returns same LIS values as calculate_LIS_family."""
    with open(dummy_json_path) as f:
        full_data = json.load(f)
    pae = np.array(full_data["pae"])
    old = calculate_LIS(str(dummy_mol_path), pae)
    new = calculate_LIS_family(str(dummy_mol_path), pae)

    assert list(old.columns) == ["chain1", "chain2", "LIS"]
    for _, row in old.iterrows():
        new_val = new.loc[(new.chain1 == row.chain1) & (new.chain2 == row.chain2), "LIS"].values[0]
        assert abs(row["LIS"] - new_val) < 1e-10


def test_calculate_ipsae(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, "r") as f:
        full_data = json.load(f)
    pae = np.array(full_data["pae"])
    result = calculate_ipsae(str(dummy_mol_path), pae)

    expected = pd.DataFrame({
        "chain1": ["A", "B"],
        "chain2": ["B", "A"],
        "ipSAE": [0.49339637364650085, 0.4620967390211816],
        "ipSAE_d0chn": [0.641075, 0.638146],
        "ipSAE_d0dom": [0.637747, 0.630472],
    })
    pd.testing.assert_frame_equal(result, expected, atol=1e-4)
