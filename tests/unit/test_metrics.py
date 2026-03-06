import pytest
import json
import numpy as np
import pandas as pd
from virus_interactome.metrics import calculate_pdockq, calculate_pdockq2, calculate_LIS, calculate_ipsae
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
