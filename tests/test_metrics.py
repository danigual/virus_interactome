import pytest
from pathlib import Path
from virus_interactome.metrics import calculate_pdockq, calculate_pdockq2, calculate_LIS, calculate_ipsae
import json
import numpy as np
import pandas as pd

@pytest.fixture
def dummy_mol_path():
    return Path(__file__).parent / "data" / "dummy_dimer_model_0.cif"

@pytest.fixture
def dummy_json_path():
    return Path(__file__).parent / "data" / "dummy_dimer_full_data_0.json"

def test_calculate_pdockq(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, 'r') as f:
        full_data = json.load(f)
    # Ensure the dummy full_data has the expected structure
    from virus_interactome.utils import process_full_data_af3
    full_data = process_full_data_af3(str(dummy_json_path))
    plddt_by_res = np.array(full_data['res_plddts'])
    calculated_pdockq = calculate_pdockq(str(dummy_mol_path), plddt_by_res)

    expected_pdockq = pd.DataFrame({"chain1": ["A"], "chain2": ["B"], "pDockQ": [0.4685032816442164]})
    pd.testing.assert_frame_equal(calculated_pdockq, expected_pdockq)

def test_calculate_pdockq2(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, 'r') as f:
        full_data = json.load(f)
    # Modify the pLDDT values to test different scenarios
    from virus_interactome.utils import process_full_data_af3
    full_data = process_full_data_af3(str(dummy_json_path))
    plddt_by_res = np.array(full_data['res_plddts'])
    pae = np.array(full_data['pae'])
    calculated_pdockq2 = calculate_pdockq2(str(dummy_mol_path), plddt_by_res, pae)

    expected_pdockq2 = pd.DataFrame({"chain1": ["A", "B"], "chain2": ["B", "A"], 
                                    "pDockQ2": [0.42800951458731734, 0.38320326568380014]})
    print(calculated_pdockq2)
    pd.testing.assert_frame_equal(calculated_pdockq2, expected_pdockq2)

def test_calculate_LIS(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, 'r') as f:
        full_data = json.load(f)
    # Modify the pLDDT values to test different scenarios
    pae = np.array(full_data['pae'])
    calculated_LIS = calculate_LIS(str(dummy_mol_path), pae)

    expected_LIS = pd.DataFrame({"chain1": ["A", "B"], "chain2": ["B", "A"], 
                                    "LIS": [0.46360432330827067, 0.4772722012190417]})
    # import pdb;pdb.set_trace()
    pd.testing.assert_frame_equal(calculated_LIS, expected_LIS)


def test_calculate_ipsae(dummy_mol_path, dummy_json_path):
    with open(dummy_json_path, 'r') as f:
        full_data = json.load(f)
    # Modify the pLDDT values to test different scenarios
    pae = np.array(full_data['pae'])
    calculated_IPSAE = calculate_ipsae(str(dummy_mol_path), pae)

    expected_IPSAE = pd.DataFrame({"chain1": ["A", "B"], "chain2": ["B", "A"], 
                                    "ipSAE": [0.49339637364650085, 0.4620967390211816],
                                    "ipSAE_d0chn": [0.641075, 0.638146],
                                    "ipSAE_d0dom": [0.637747, 0.630472]})
    pd.testing.assert_frame_equal(calculated_IPSAE, expected_IPSAE)
