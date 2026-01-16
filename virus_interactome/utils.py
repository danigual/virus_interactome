import json
import yaml
from collections import OrderedDict
import numpy as np
from typing import Union
from glob import glob
import numpy as np
import os
from moleculekit.molecule import Molecule

def check_sequence_validity(seq: str) -> bool:
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    return all(residue in VALID_AMINO_ACIDS for residue in seq)

def load_json(json_path: str)-> Union[dict, list]:
    """
    Loads and parses a JSON file from the specified path.

    This function opens a JSON file and returns its contents as a Python dictionary
    or list, depending on the structure of the JSON.

    Parameters
    ----------
    json_path : str
        Path to the JSON file to be loaded.

    Returns
    -------
    dict or list
        Parsed content of the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not a valid JSON.
    """

    with open (json_path,"r") as j:
        data = json.load(j)
    return data

def load_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def load_boltz_input(yaml_path: str, job_name: str | None  = None):
    """
    This has to follow the same convention as af3 JSONs

    [{
        "name": <name>,
        "sequences": [{
            "proteinChain": {
                "count": <count>,
                "sequence": <sequence>
            }
        }]
    }] 
    It is only going to be one, but for compatibility with AF3, we put it in a list.
    """
    data = load_yaml(yaml_path)

    if job_name is None:
        job_name = os.path.splitext(os.path.basename(yaml_path))[0]

    sequence_info = []
    for seq in data["sequences"]:
        count = seq.get("protein").get("multiple_chains")
        count = count.count(",")+2 if count is not None else 1
        tmp_sequence_info = {
            "proteinChain":{
                "sequence": seq.get("protein").get("sequence"),
                "count": count,
            }
        }
        sequence_info.append(tmp_sequence_info)
    job = [
        {
            "name": job_name,
            "sequences": sequence_info
        }
    ]
    return job

def process_full_data_boltz(mol_file: str, 
                            pae_file: str | None = None,
                            plddt_file: str | None = None,
                            pde_file: str | None = None,
                            confidence_file: str | None = None,
                            )-> dict:
    """
    Process Boltz output files and return structural confidence data.

    Parameters
    ----------
    mol_file : str
        Path to the main CIF file (mandatory).
    confidence_file : str, optional
        Path to the confidence JSON file. If None, it will be inferred.
    pae_file : str, optional
        Path to the PAE NPZ file. If None, it will be inferred.
    plddt_file : str, optional
        Path to the pLDDT NPZ file. If None, it will be inferred.
    pde_file : str, optional
        Path to the PDE NPZ file. If None, it will be inferred.
    confidence_file : str, optional
        Path to the summary confidences JSON file. If None, it will be inferred.
    Returns
    -------
    dict
        Dictionary containing:
        - 'pae': numpy.ndarray, PAE matrix
        - 'plddt': numpy.ndarray, pLDDT scores
        - 'ptm': float, predicted TM-score (if available)
        - 'iptm': float, interface predicted TM-score (if available)

    Raises
    ------
    FileNotFoundError
        If any required file cannot be found.
    ValueError
        If the NPZ files do not contain expected arrays.
    """

    base_name = os.path.splitext(os.path.basename(mol_file))[0]  # pvi__protease_model_0
    dir_name = os.path.dirname(mol_file)

    # Deduce rutas si no se pasan
    if confidence_file is None:
        confidence_file = os.path.join(dir_name, f"confidence_{base_name}.json")
    if pae_file is None:
        pae_file = os.path.join(dir_name, f"pae_{base_name}.npz")
    if plddt_file is None:
        plddt_file = os.path.join(dir_name, f"plddt_{base_name}.npz")
    if pde_file is None:
        pde_file = os.path.join(dir_name, f"pde_{base_name}.npz")

    # Validate file existence
    for file_path, description in [
        (mol_file, "molecule CIF/PDB"),
        (confidence_file, "confidence JSON"),
        (pae_file, "PAE NPZ"),
        (plddt_file, "pLDDT NPZ"),
        (pde_file, "PDE NPZ"),
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {description} file: {file_path}")

    mol = Molecule(mol_file)
    pae_data = np.load(pae_file)["pae"]
    pde_data = np.load(pde_file)["pde"]
    plddt_data = np.load(plddt_file)["plddt"] * 100 ## In AF3 we have range 1-100
    confidence_data = load_json(confidence_file)
    chain_by_res = mol.chain[mol.name == "CA"]


    if pae_data.shape != pde_data.shape:
        raise ValueError(f"PAE and PDE shapes do not match: {pae_data.shape} vs {pde_data.shape}")

    if pae_data.shape[0] != len(plddt_data):
        raise ValueError(f"pLDDT length ({len(plddt_data)}) does not match PAE size ({pae_data.shape[0]})")

    if len(chain_by_res) != len(plddt_data):
        raise ValueError(f"pLDDT length ({len(plddt_data)}) does not match chain_by_res size ({len(chain_by_res)})")

    ## Check dimensions 
    chain_boundaries = {}
    chain_boundaries_by_atom = {}
  
    for chain_id in np.unique(mol.chain):
        chain_indexes = np.where(np.array(chain_by_res) == chain_id)
        chain_boundaries[chain_id] = (np.min(chain_indexes), np.max(chain_indexes))
        atom_chain_indexes = np.where(np.array(mol.chain) == chain_id)
        chain_boundaries[chain_id] = (np.min(chain_indexes), np.max(chain_indexes))
        chain_boundaries_by_atom[chain_id] = (np.min(atom_chain_indexes), np.max(atom_chain_indexes))
    
    iptm_by_chain = confidence_data.get("pair_chains_iptm", None)
    ndim = len(iptm_by_chain)
    iptm_by_chain_as_list = np.zeros((ndim, ndim))
    for i in range(len(iptm_by_chain)):
        for j in range(len(iptm_by_chain)):
            iptm_by_chain_as_list[i,j] = iptm_by_chain[str(i)].get(str(j), None)

    return {"pae": pae_data, 
            "atom_plddts": mol.beta,
            "cb_plddts": plddt_data,
            "ca_plddts": plddt_data,
            "chain_boundaries_by_res": chain_boundaries,
            "chain_boundaries_by_atom": chain_boundaries_by_atom,
            "token_chain_ids": chain_by_res,
            "ptm": confidence_data.get("ptm", None),
            "iptm": confidence_data.get("iptm", None),
            "iptm_chain_pair": iptm_by_chain_as_list}

def process_full_data_af3(mol_file: str,
                          json_path: str | None = None,
                          summary_json_path: str | None = None,)-> dict: ## Maybe this should be a mol_file also?
    
    """
    Processes AlphaFold3 full data JSON and extracts structural metadata.

    This function loads a JSON file containing AlphaFold3 output, computes chain lengths,
    boundaries (both residue and atom-level), and converts key fields to NumPy arrays for
    efficient downstream analysis. It enriches the original data with additional structural
    annotations.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing AlphaFold3 full data output.

    Returns
    -------
    dict
        Dictionary containing the original data plus:
        - "chain_lengths": OrderedDict with residue counts per chain.
        - "chain_boundaries": List of tuples with residue index ranges per chain.
        - "chain_boundaries_by_atom": List of tuples with atom index ranges per chain.
        - "pae", "atom_plddts", "contact_probs": Converted to NumPy arrays.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    KeyError
        If expected keys are missing in the JSON structure.
    """
    from moleculekit.molecule import Molecule

    if json_path is None:
        json_path = mol_file.replace("_model_", "_full_data_").replace(".cif", ".json")
    if summary_json_path is None:
        summary_json_path = mol_file.replace("_model_", "_summary_confidences_").replace(".cif", ".json")

    # Validate file existence
    for file_path, description in [
        (mol_file, "molecule CIF/PDB"),
        (json_path, "full data JSON"),
        (summary_json_path, "confidence JSON"),
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
    full_data = load_json(json_path)
    summary_data = load_json(summary_json_path)
    token_chain_ids = np.array(full_data.get("token_chain_ids"))
    atom_chain_ids = np.array(full_data.get("atom_chain_ids"))
    
    mol = Molecule(mol_file)
    chain_by_res = mol.chain[mol.name == "CA"]

    # mol_path = json_path.replace("full_data", "model").replace(".json", ".cif")
    mol = Molecule(mol_file)
    ca_mask = mol.name == "CA"
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    cb_plddt = np.array(full_data.get("atom_plddts"))[cb_mask]
    ca_plddt = np.array(full_data.get("atom_plddts"))[ca_mask]
    
    chain_boundaries = {}
    chain_boundaries_by_atom = {}
    for chain_id in np.unique(token_chain_ids):
        chain_indexes = np.where(np.array(chain_by_res) == chain_id)
        chain_boundaries[str(chain_id)] = (int(np.min(chain_indexes)), int(np.max(chain_indexes)))
        
        atom_chain_indexes = np.where(atom_chain_ids == chain_id)
        chain_boundaries_by_atom[str(chain_id)] = (np.min(atom_chain_indexes), np.max(atom_chain_indexes))

    ## Convert pae, atom_plddts and contact_probs to np arrays
    full_data["pae"] = np.array(full_data["pae"])
    full_data["ptm"] = summary_data["ptm"]
    full_data["iptm"] = summary_data["iptm"]
    full_data["iptm_chain_pair"] = np.array(summary_data["chain_pair_iptm"])
    full_data["atom_plddts"] = np.array(full_data["atom_plddts"])
    full_data["cb_plddts"] = np.array(cb_plddt)
    full_data["ca_plddts"] = np.array(ca_plddt)
    # full_data["contact_probs"] = np.array(full_data["contact_probs"])
    full_data["token_chain_ids"] = token_chain_ids
    
    # full_data["chain_lengths"] = chain_lengths 
    full_data["chain_boundaries_by_res"] = chain_boundaries
    full_data["chain_boundaries_by_atom"] = chain_boundaries_by_atom
    return full_data
