import json
from collections import OrderedDict
import numpy as np
from typing import Union

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

def process_full_data_af3(json_path: str)-> dict:
    
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

    full_data = load_json(json_path)
    token_chain_ids = full_data["token_chain_ids"]
    atom_chain_ids = full_data["atom_chain_ids"] 
  
    
    chain_lengths = OrderedDict()
    for chain_id in token_chain_ids:
        if chain_id not in chain_lengths:
            chain_lengths[chain_id] = 0
        chain_lengths[chain_id] += 1
    
    atom_chain_lengths = OrderedDict()
    for atom_id in atom_chain_ids:
        if atom_id not in atom_chain_lengths:
            atom_chain_lengths[atom_id] = 0
        atom_chain_lengths[atom_id] += 1

    chain_boundaries = []
  
    for chain_id in chain_lengths.keys():
        
        chain_indexes = np.where(np.array(token_chain_ids) == chain_id)
        
        chain_boundaries.append((np.min(chain_indexes), np.max(chain_indexes)))
   
    
    chain_boundaries_by_atom = []
   
    for atom_id in atom_chain_lengths.values():
    
        atom_chain_indexes = np.where(np.array(atom_chain_ids) == atom_id)
        
        chain_boundaries_by_atom.append((np.min(atom_chain_indexes), np.max(atom_chain_indexes)))
   
    
    ## Convert pae, atom_plddts and contact_probs to np arrays
    full_data["pae"] = np.array(full_data["pae"])
    full_data ["atom_plddts"] = np.array(full_data["atom_plddts"])
    full_data["contact_probs"] = np.array(full_data["contact_probs"])
    
    full_data["chain_lengths"] = chain_lengths 
    full_data["chain_boundaries"] = chain_boundaries
    full_data["chain_boundaries_by_atom"] = chain_boundaries_by_atom
    return full_data










