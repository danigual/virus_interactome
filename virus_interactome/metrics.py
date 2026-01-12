import numpy as np
import json
import os, sys
from moleculekit.molecule import Molecule
import pandas as pd
import itertools
import math

residue_set= {"ALA", "ARG", "ASN", "ASP", "CYS",
              "GLN", "GLU", "GLY", "HIS", "ILE",
              "LEU", "LYS", "MET", "PHE", "PRO",
              "SER", "THR", "TRP", "TYR", "VAL",
              "DA", "DC", "DT", "DG", "A", "C", "U", "G"}

nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

# Define the ptm and d0 functions
def ptm_func(x,d0):
    return 1.0/(1+(x/d0)**2.0)  
ptm_func_vec=np.vectorize(ptm_func)  # vector version

# Define the d0 functions for numbers and arrays; minimum value = 1.0; from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702–710 (2004)
def calc_d0(L,pair_type):
    L=float(L)
    if L<27: L=27
    min_value=1.0
    if pair_type=='nucleic_acid': min_value=2.0
    d0=1.24*(L-15)**(1.0/3.0) - 1.8
    return max(min_value, d0)

def calc_d0_array(L,pair_type):
    # Convert L to a NumPy array if it isn't already one (enables flexibility in input types)
    L = np.array(L, dtype=float)
    L = np.maximum(27,L)
    min_value=1.0

    if pair_type=='nucleic_acid': min_value=2.0

    # Calculate d0 using the vectorized operation
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0/3.0) - 1.8)

# Initializes a nested dictionary with all values set to 0
def init_chainpairdict_zeros(chainlist):
    return {chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with NumPy arrays of zeros of a specified size
def init_chainpairdict_npzeros(chainlist, arraysize):
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with empty sets.
def init_chainpairdict_set(chainlist):
    return {chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

def classify_chains(chains, residue_types):
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    
    # Get unique chains and iterate over them
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        # Find indices where the current chain is located
        indices = np.where(chains == chain)[0]
        # Get the residues for these indices
        chain_residues = residue_types[indices]
        # Count nucleic acid residues
        nuc_count = sum(residue in nuc_residue_set for residue in chain_residues)
        
        # Determine if the chain is a nucleic acid or protein
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'
    
    return chain_types

def calculate_pdockq(mol_file, plddt_by_res, pDockQ_cutoff=8.0):
    mol = None
    if isinstance(mol_file, Molecule):
        mol = mol_file
    elif isinstance(mol_file, str):
        try:
            mol = Molecule(mol_file)
        except:
            raise FileNotFoundError(f"File {mol_file} does not exists")
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    # cb_plddt = plddt_by_atom[cb_mask]
    cb_plddt = plddt_by_res
    unique_chains = [str(i) for i in np.unique(mol.chain)]

    coordinates = mol.coords[cb_mask].reshape(-1,3) 
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    chains = mol.chain[cb_mask]

    pDockQ = pd.DataFrame({"chain1": [], "chain2": [], "pDockQ": []})
    for chain1, chain2 in itertools.combinations(unique_chains, 2):
        interface_distances = distances[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_matrix = interface_distances<=pDockQ_cutoff
        npairs = np.sum(interface_matrix)

        tmp_pdockq = 0.0          
        if npairs > 0:
            res_chain1_indices, res_chain2_indices = np.where(interface_matrix)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)

            cb_plddt_chain1 = cb_plddt[chains==chain1][unique_res_chain1_indexes]
            cb_plddt_chain2 = cb_plddt[chains==chain2][unique_res_chain2_indexes]
            mean_plddt = np.concatenate((cb_plddt_chain1, cb_plddt_chain2)).mean()
            # mean_plddt = cb_plddt[ list(pDockQ_unique_residues[chain1][chain2])].mean()
            x = mean_plddt * math.log10(npairs)
            tmp_pdockq = 0.724 / (1 + math.exp(-0.052*(x-152.611)))+0.018
        pDockQ = pd.concat([pDockQ, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], "pDockQ": [tmp_pdockq]})], ignore_index=True)

    return pDockQ

def calculate_pdockq2(mol_file, plddt_by_res, pae_matrix, pDockQ_cutoff=8.0):
    mol = None
    if isinstance(mol_file, Molecule):
        mol = mol_file
    elif isinstance(mol_file, str):
        try:
            mol = Molecule(mol_file)
        except:
            raise FileNotFoundError(f"File {mol_file} does not exists")
        
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    # cb_plddt = plddt_by_atom[cb_mask]
    cb_plddt = plddt_by_res
    unique_chains = [str(i) for i in np.unique(mol.chain)]

    coordinates = mol.coords[cb_mask].reshape(-1,3) 
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    chains = mol.chain[cb_mask]

    pDockQ2 = pd.DataFrame({"chain1": [], "chain2": [], "pDockQ2": []})
    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        interface_distances = distances[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_pae = pae_matrix[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_dist_matrix = interface_distances<=pDockQ_cutoff
        npairs = np.sum(interface_dist_matrix)

        tmp_pdockq2 = 0.0          
        if npairs > 0:
            pae_ptm_sum = np.sum(ptm_func_vec(interface_pae[interface_dist_matrix], 10))
            res_chain1_indices, res_chain2_indices = np.where(interface_dist_matrix)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)

            cb_plddt_chain1 = cb_plddt[chains==chain1][unique_res_chain1_indexes]
            cb_plddt_chain2 = cb_plddt[chains==chain2][unique_res_chain2_indexes]
            mean_plddt = np.concatenate((cb_plddt_chain1, cb_plddt_chain2)).mean()
            mean_ptm = pae_ptm_sum / npairs
            x = mean_plddt * mean_ptm
            tmp_pdockq2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        pDockQ2 = pd.concat([pDockQ2, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], "pDockQ2": [tmp_pdockq2]})], ignore_index=True)

    return pDockQ2

def calculate_LIS(mol_file, pae_matrix, threshold=12.0):
    mol = None
    if isinstance(mol_file, Molecule):
        mol = mol_file
    elif isinstance(mol_file, str):
        try:
            mol = Molecule(mol_file)
        except:
            raise FileNotFoundError(f"File {mol_file} does not exists")
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    unique_chains = [str(i) for i in np.unique(mol.chain)]

    chains = mol.chain[cb_mask]
    unique_chains = [str(i) for i in np.unique(chains)]
    # LIS = init_chainpairdict_zeros(list(unique_chains))
    LIS = pd.DataFrame({"chain1": [], "chain2": [], "LIS": []})

    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        # import pdb; pdb.set_trace() 
        mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)  # Select residues for (chain1, chain2)
        selected_pae = pae_matrix[mask]  # Get PAE values for this pair
        
        if selected_pae.size > 0:  # Ensure we have values
            valid_pae = selected_pae[selected_pae <= threshold]  # Apply the threshold
            if valid_pae.size > 0:
                scores = (threshold - valid_pae) / threshold  # Compute scores
                avg_score = np.mean(scores)  # Average score for (chain1, chain2)
                tmp_lis = avg_score
            else:
                tmp_lis = 0.0  # No valid values
        else:
            tmp_lis = 0.0     
        LIS = pd.concat([LIS, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], "LIS": [tmp_lis]})], ignore_index=True)
    return LIS

def calculate_ipsae(mol_file, pae_matrix, pae_cutoff=10, dist_cutoff=10.0):
    mol = None
    if isinstance(mol_file, Molecule):
        mol = mol_file
    elif isinstance(mol_file, str):
        try:
            mol = Molecule(mol_file)
        except:
            raise FileNotFoundError(f"File {mol_file} does not exists")
        
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    numres = cb_mask.sum()
    # mol_residues = mol.resid[cb_mask]
    unique_chains = [str(i) for i in np.unique(mol.chain)]

    # coordinates = mol.coords[cb_mask].reshape(-1,3) 
    # distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    chains = mol.chain[cb_mask]
    
    ipsae = pd.DataFrame({"chain1": [], "chain2": [], "ipSAE": [], "ipSAE_d0chn": [], "ipSAE_d0dom": []})

    # d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    # d0_nucleic_acid = 2.0
    
    chain_dict = classify_chains(chains, mol.resname)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1==chain2: continue
            if chain_dict[chain1] == 'nucleic_acid' or chain_dict[chain2] == 'nucleic_acid':
                chain_pair_type[chain1][chain2]='nucleic_acid'
            else:
                chain_pair_type[chain1][chain2]='protein'

    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        n0chn = np.sum(chains==chain1) + np.sum(chains==chain2) # total number of residues in chain1 and chain2
        d0chn = calc_d0(n0chn, chain_pair_type[chain1][chain2])
        ptm_matrix_d0chn=ptm_func_vec(pae_matrix, d0chn)

        # valid_pairs_iptm = (chains == chain2)
        # valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)
        interface_pae = pae_matrix[chains == chain1][:, chains == chain2]
        interface_pae_mask = interface_pae < pae_cutoff

        if interface_pae_mask.any():
            interface_ptm_d0chn = ptm_matrix_d0chn[chains == chain1][:, chains == chain2]
            interface_ptm_d0chn[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0chn_byres = np.nanmean(interface_ptm_d0chn, axis=1)
            
            tmp_ipSAE_d0chn = np.nanmax(ipsae_d0chn_byres)

            res_chain1_indices, res_chain2_indices = np.where(interface_pae_mask)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)
            n0dom = len(unique_res_chain1_indexes) + len(unique_res_chain2_indexes)

            d0dom = calc_d0(n0dom, chain_pair_type[chain1][chain2])
            ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom)
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)
            interface_ptm_d0dom = ptm_matrix_d0dom[chains == chain1][:, chains == chain2]
            interface_ptm_d0dom[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0dom_byres = np.nanmean(interface_ptm_d0dom, axis=1)
            tmp_ipSAE_d0dom = np.nanmax(ipsae_d0dom_byres)
            
            
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])
            ptm_matrix_d0res = np.array([ptm_func_vec(pae_matrix[i], d0res_byres[i]) for i in range(numres)])
            interface_ptm_d0res = ptm_matrix_d0res[chains == chain1][:, chains == chain2]
            interface_ptm_d0res[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0res_byres = np.nanmean(interface_ptm_d0res, axis=1)
            print(ipsae_d0res_byres.shape)
            tmp_ipSAE_d0res = np.nanmax(ipsae_d0res_byres)
            ipsae = pd.concat([ipsae, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], 
                                                    "ipSAE": [tmp_ipSAE_d0res], "ipSAE_d0chn": [tmp_ipSAE_d0chn],
                                                    "ipSAE_d0dom": [tmp_ipSAE_d0dom]})])
        else: 
            ipsae = pd.concat([ipsae, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], 
                                                    "ipSAE": [0], "ipSAE_d0chn": [0],
                                                    "ipSAE_d0dom": [0]})])
    return ipsae.reset_index(drop=True)


def calculate_all_metrics(mol_file, all_metrics):
    mol = None
    if isinstance(mol_file, Molecule):
        mol = mol_file
    elif isinstance(mol_file, str):
        try:
            mol = Molecule(mol_file)
        except:
            raise FileNotFoundError(f"File {mol_file} does not exists")

    pae = all_metrics["pae"]
    # plddt_by_atom = all_metrics["atom_plddts"]
    plddt_by_residue = all_metrics["cb_plddts"]
    chain_by_res = np.array(all_metrics["token_chain_ids"])

    # import pdb;pdb.set_trace()

    ipsae = calculate_ipsae(mol, pae)

    ipSAE_AB = ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE"].values[0],
    ipSAE_BA = ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE"].values[0],
    LIS = calculate_LIS(mol, pae)
    # pdockq = calculate_pdockq(mol, plddt_by_atom=plddt_by_atom)
    # pdockq2 = calculate_pdockq2(mol, plddt_by_atom=plddt_by_atom, pae_matrix=pae)
    pdockq = calculate_pdockq(mol, plddt_by_res=plddt_by_residue)
    pdockq2 = calculate_pdockq2(mol, plddt_by_res=plddt_by_residue, pae_matrix=pae)

    return {
        # "pLDDT_mean": np.mean(plddt_by_atom),
        # "pLDDT_mean_A": np.mean(plddt_by_atom[mol.chain == "A"]),
        # "pLDDT_mean_B": np.mean(plddt_by_atom[mol.chain == "B"]),
        "pLDDT_mean": np.mean(plddt_by_residue),
        "pLDDT_mean_A": np.mean(plddt_by_residue[chain_by_res == "A"]),
        "pLDDT_mean_B": np.mean(plddt_by_residue[chain_by_res == "B"]),
        "pae_mean": np.mean(pae),
        "pae_mean_A": np.mean(pae[chain_by_res == "A"][:, chain_by_res == "A"]),
        "pae_mean_B": np.mean(pae[chain_by_res == "B"][:, chain_by_res == "B"]),
        "pae_mean_AB": np.mean([np.mean(pae[chain_by_res == "A"][:, chain_by_res == "B"]), 
                                np.mean(pae[chain_by_res == "B"][:, chain_by_res == "A"])
                        ]),
        "pDockQ": pdockq.loc[(pdockq.chain1 == "A") & (pdockq.chain2 == "B"), "pDockQ"].values[0],
        "pDockQ2_AB": pdockq2.loc[(pdockq2.chain1 == "A") & (pdockq2.chain2 == "B"), "pDockQ2"].values[0],
        "pDockQ2_BA": pdockq2.loc[(pdockq2.chain1 == "B") & (pdockq2.chain2 == "A"), "pDockQ2"].values[0],
        "LIS_AB": LIS.loc[(LIS.chain1 == "A") & (LIS.chain2 == "B"), "LIS"].values[0],
        "LIS_BA": LIS.loc[(LIS.chain1 == "B") & (LIS.chain2 == "A"), "LIS"].values[0],
        "ipSAE_AB": ipSAE_AB,
        "ipSAE_BA": ipSAE_BA, 
        "max_ipSAE": np.max([ipSAE_AB, ipSAE_BA]),
        "ipSAE_d0chn_AB": ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE_d0chn"].values[0],
        "ipSAE_d0chn_BA": ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE_d0chn"].values[0],
        "ipSAE_d0dom_AB": ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE_d0dom"].values[0],
        "ipSAE_d0dom_BA": ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE_d0dom"].values[0],
    }