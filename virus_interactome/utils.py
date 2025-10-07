import json
from collections import OrderedDict
import numpy as np

def load_json(json_path: str):
    with open (json_path,"r") as j:
        data = json.load(j)
    return data

def process_full_data_af3(json_path: str):
    full_data = load_json(json_path)
    token_chain_ids = full_data["token_chain_ids"]
    atom_chain_ids = full_data["atom_chain_ids"] 
    # pae_matrix = full_data["pae"]
    
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
    start = 0
    for length in chain_lengths.values():
        end = start + length
        chain_boundaries.append((start, end))
        start = end
    
    chain_boundaries_by_atom = []
    start = 0
    for length in atom_chain_lengths.values():
        end = start + length
        chain_boundaries_by_atom.append((start, end))
        start = end
    
    ## Convert pae, atom_plddts and contact_probs to np arrays
    full_data["pae"] = np.array(full_data["pae"])
    full_data ["atom_plddts"] = np.array(full_data["atom_plddts"])
    full_data["contact_probs"] = np.array(full_data["contact_probs"])
    
    full_data["chain_lengths"] = chain_lengths 
    full_data["chain_boundaries"] = chain_boundaries
    full_data["chain_boundaries_by_atom"] = chain_boundaries_by_atom
    return full_data

