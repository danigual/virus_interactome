from itertools import combinations
import warnings
import os
import json
from typing import Union
from pathlib import Path
from Bio import SeqIO

def load_proteome (fasta_file:str): ## Moving this to proteome_input
    ''' Receives a fasta file and extracts the ID-sequence
            ARGS IN:
                fasta_file (.fasta)
            ARGS OUT:
                proteins_dic (dictionary): key-value, id-aas_sequence
    '''

    if not Path(fasta_file).is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    proteome_dict = {}
    for protein in SeqIO.parse(fasta_file,"fasta"):
        complete_id = protein.id
        fractioned_id = complete_id.split('|')
        short_id = fractioned_id[0]

        if short_id in proteome_dict:
            warnings.warn(f"Duplicate protein ID '{short_id}' found. Overwriting previous entry.")

        sequence = str(protein.seq)
        proteome_dict[short_id] = sequence
    return proteome_dict

def create_af3_input_json_v2(*args, proteome_dict:dict, prefix=None, suffix=None):
    orf_list = []
    orf_num_copies = []
    idx = 0
    while idx < len(args):
        if isinstance(args[idx], str):
            # Check if next argument is an integer (copy count)
            orf_list.append(args[idx])
            if idx + 1 < len(args) and isinstance(args[idx + 1], int):
                orf_num_copies.append(args[idx + 1])
                idx += 2
            else:
                orf_num_copies.append(1)
                idx += 1
        else:
            raise ValueError(f"Expected string at position {idx}, got {type(args[idx])}")
    
    # Check all orfs are in the proteome_dict
    for orf in orf_list:
        if orf not in proteome_dict:
            raise KeyError(f"ORF {orf} is missing from proteome_dict")

    # Check all num_copies are integers and > 0
    for num in orf_num_copies:
        if not isinstance(num, int) or num <= 0:
            raise ValueError(f"Number of copies must be a positive integer, got {num}")

    # Generate the json
    if prefix is not None:
        prefix = prefix + "_" 
    else:
        prefix = ""
    
    if suffix is not None:
        suffix = "___" + suffix  
    else:
        suffix = ""

    header = prefix + "__".join(orf_list) + suffix

    return {
        "name": header,
        "sequences": 
            [
                {
                    "proteinChain": {
                        "count": num,
                        "sequence": proteome_dict[orf]
                    }
                } for orf, num in zip(orf_list, orf_num_copies)
            ]
    }

# def create_af3_input_json(orf1, orf2, proteome_dict:dict):
#     return {
#             "name": f"AdV5_{orf1}__{orf2}",
#             "sequences":[
#                 {
#                     "proteinChain": {
#                         "count": 1,
#                         "sequence": f"{proteome_dict[orf1]}"
#                     }
#                 }, {
#                     "proteinChain": {
#                         "count": 1,
#                         "sequence": f"{proteome_dict[orf2]}"
#                                 }
#                 }
#             ]
#         }

def proteome_json (proteome_dict: dict, outputdir:str, batch_size=30):
    '''Receives a dictionary and returns a list of dictionaries as .json file
            ARGS IN:
                key_value_dict (dictionary): key-protein ID, value-aas_seq
            ARGS OUT:
                writes ".json" files in the outputdir           
    '''
    os.makedirs(outputdir, exist_ok= True) 
   
    keys = list (proteome_dict.keys())
    orf_combinations = combinations(keys, 2)
  
    tmp_batch = []
    file_idx = 0
    for orf1, orf2 in orf_combinations:
   
        if (len(tmp_batch) == batch_size):
            tmp_output_name = os.path.join(outputdir, f"{file_idx}.json")
            with open(tmp_output_name, "w") as f:
                json.dump(tmp_batch, f, indent=4)
            
            tmp_batch = []
            file_idx += 1
        
        tmp_job = create_af3_input_json_v2(orf1, orf2, proteome_dict=proteome_dict)
        tmp_batch.append(tmp_job)
    
    ## Writing the remaining
    if len(tmp_batch) > 0:
        tmp_output_name = os.path.join(outputdir, f"{file_idx}.json")
        with open(tmp_output_name, "w") as f:
            json.dump(tmp_batch, f, indent=4)

def generate_heterodimers_jobs(proteome_dict:dict):
    proteome_ids = list (proteome_dict.keys())
    orf_combinations = combinations(proteome_ids, 2)
  
    for orf1, orf2 in orf_combinations:
        yield create_af3_input_json_v2(orf1, orf2, proteome_dict=proteome_dict)

def generate_n_homo_mers_jobs(proteome_dict:dict, max_n_homo_mers:int=0, **kwargs):
    proteome_ids = list (proteome_dict.keys())

    for orf in proteome_ids:
        for number_proteins in range(2, max_n_homo_mers+1):
            yield create_af3_input_json_v2(orf, number_proteins, proteome_dict=proteome_dict, **kwargs)
            # yield create_af3_input_json_v2(orf, number_proteins, proteome_dict=proteome_dict, suffix=f"{number_proteins}mer")

def write_batch(job_generator, outputdir:str, label:str="", batch_size:int=30):
    file_idx = 0
    tmp_jobs = []
    for job in job_generator:
        tmp_jobs.append(job)

        if len(tmp_jobs) == batch_size:
            tmp_output_name = os.path.join(outputdir, f"{label}_{file_idx}.json")
            with open(tmp_output_name, "w") as f:
                json.dump(tmp_jobs, f, indent=4)
            tmp_jobs = []
            file_idx += 1

    tmp_output_name = os.path.join(outputdir, f"{label}_{file_idx}.json")
    with open(tmp_output_name, "w") as f:
        json.dump(tmp_jobs, f, indent=4)
    tmp_jobs = []

def generate_interactome_jsons(proteome:Union[str, dict], outputdir:str, batch_size:int=30):
    if isinstance(proteome, dict):
        proteome_dict = proteome
    elif isinstance(proteome, str):
        proteome_dict = load_proteome(proteome)
    os.makedirs(outputdir, exist_ok= True) 
    
    ## Generate heterodimers jsons
    heterodimer_jobs = generate_heterodimers_jobs(proteome_dict)
    write_batch(heterodimer_jobs, outputdir, label="HETERO", batch_size=batch_size)
    homo_mers_jobs = generate_n_homo_mers_jobs(proteome_dict, 6)
    write_batch(homo_mers_jobs, outputdir, label="HOMO", batch_size=batch_size)
