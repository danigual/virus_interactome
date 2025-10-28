from itertools import combinations
import warnings
import os
import json
from typing import Union
from pathlib import Path
from Bio import SeqIO

def load_proteome (fasta_file:str): ## Moving this to proteome_input
    
    """
    Loads a FASTA file and extracts protein sequences into a dictionary.

    This function parses a FASTA file and builds a dictionary mapping protein IDs to their
    amino acid sequences. It uses the first segment of the ID (before the first '|') as the key.
    Duplicate IDs will trigger a warning and overwrite previous entries.

    Parameters
    ----------
    fasta_file : str
        Path to the FASTA file containing protein sequences.

    Returns
    -------
    dict
        Dictionary where keys are protein IDs and values are amino acid sequences.

    Raises
    ------
    FileNotFoundError
        If the FASTA file does not exist.
    UserWarning
        If duplicate protein IDs are found.
    """

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

<<<<<<< HEAD
def create_af3_input_json_v2(*args, proteome_dict:dict, prefix=None, suffix=None):
=======

def create_af3_input_json_v2(*args, proteome_dict:dict, proteome_label=None):
    
    """
    Creates a JSON-compatible dictionary for AlphaFold3 input based on selected ORFs and copy counts.

    This function receives a variable number of arguments representing ORF names and optional copy counts.
    It validates the input, checks for ORF presence in the proteome dictionary, and constructs a structured
    dictionary suitable for AlphaFold3 input. If no copy count is provided for an ORF, it defaults to 1.

    Parameters
    ----------
    *args : str and int
        A sequence of ORF names followed optionally by integers indicating the number of copies.
    proteome_dict : dict
        Dictionary mapping ORF names to amino acid sequences.
    proteome_label : str, optional
        Label to prefix the generated input name.

    Returns
    -------
    dict
        A dictionary containing the input name and a list of protein chains with sequence and copy count.

    Raises
    ------
    ValueError
        If the argument sequence is malformed or copy counts are invalid.
    KeyError
        If any ORF is not found in the proteome dictionary.
    """

>>>>>>> 55a59b3c6a11cdc74aff4ad760ebd9c072fb38fc
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


def proteome_json (proteome_dict: dict, outputdir:str, batch_size=30):
    
    """
    Generates JSON files containing pairwise ORF combinations for AlphaFold3 input.

    This function takes a proteome dictionary and creates input jobs for all possible
    heterodimer combinations (pairs of ORFs). Jobs are batched and written to JSON files
    in the specified output directory.

    Parameters
    ----------
    proteome_dict : dict
        Dictionary mapping ORF names to amino acid sequences.
    outputdir : str
        Directory where the JSON files will be saved.
    batch_size : int, optional
        Number of jobs per JSON file (default is 30).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the output directory cannot be created.
    """

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
    
    """
    Yields AlphaFold3 input jobs for all pairwise heterodimer combinations in a proteome.

    Parameters
    ----------
    proteome_dict : dict
        Dictionary mapping ORF names to amino acid sequences.

    Yields
    ------
    dict
        AlphaFold3 input job dictionary for each ORF pair.
    """

    proteome_ids = list (proteome_dict.keys())
    orf_combinations = combinations(proteome_ids, 2)
  
    for orf1, orf2 in orf_combinations:
        yield create_af3_input_json_v2(orf1, orf2, proteome_dict=proteome_dict)

<<<<<<< HEAD
def generate_n_homo_mers_jobs(proteome_dict:dict, max_n_homo_mers:int=0, **kwargs):
=======
def generate_n_homo_mers_jobs(proteome_dict:dict, max_n_homo_mers:int=0):
    
    """
    Yields AlphaFold3 input jobs for homomeric assemblies of each ORF up to a specified size.

    Parameters
    ----------
    proteome_dict : dict
        Dictionary mapping ORF names to amino acid sequences.
    max_n_homo_mers : int, optional
        Maximum number of copies per homomer (default is 0, which yields nothing).

    Yields
    ------
    dict
        AlphaFold3 input job dictionary for each homomer configuration.
    """

>>>>>>> 55a59b3c6a11cdc74aff4ad760ebd9c072fb38fc
    proteome_ids = list (proteome_dict.keys())

    for orf in proteome_ids:
        for number_proteins in range(2, max_n_homo_mers+1):
            yield create_af3_input_json_v2(orf, number_proteins, proteome_dict=proteome_dict, **kwargs)
            # yield create_af3_input_json_v2(orf, number_proteins, proteome_dict=proteome_dict, suffix=f"{number_proteins}mer")

def write_batch(job_generator, outputdir:str, label:str="", batch_size:int=30):
    
    """
    Writes batches of AlphaFold3 input jobs to JSON files.

    This function collects jobs from a generator and writes them in batches to JSON files
    in the specified output directory. Each file is labeled with a prefix and an index.

    Parameters
    ----------
    job_generator : generator
        Generator yielding AlphaFold3 input job dictionaries.
    outputdir : str
        Directory where the JSON files will be saved.
    label : str, optional
        Prefix label for the output filenames.
    batch_size : int, optional
        Number of jobs per JSON file (default is 30).

    Returns
    -------
    None
    """

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
    """
    Generates JSON files for heterodimer and homomer AlphaFold3 input jobs from a proteome.

    This function accepts either a proteome dictionary or a path to a FASTA file, and creates
    input jobs for all pairwise heterodimers and homomers (up to 5 copies). Jobs are batched
    and saved as JSON files in the specified output directory.

    Parameters
    ----------
    proteome : Union[str, dict]
        Either a path to a FASTA file or a dictionary mapping ORF names to sequences.
    outputdir : str
        Directory where the JSON files will be saved.
    batch_size : int, optional
        Number of jobs per JSON file (default is 30).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the FASTA file or output directory is invalid.
    """
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
