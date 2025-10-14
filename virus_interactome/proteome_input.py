import warnings
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

def create_af3_input_json_v2(*args, proteome_dict:dict, proteome_label = None):
    orf_list = []
    orf_num_copies = []

    for idx, i in enumerate(args): 
        if isinstance(i, str):
            orf_list.append(i)
        
            if isinstance(args[idx + 1], int):
                orf_num_copies.append(args[idx + 1])
                idx += 2
            else:
                orf_num_copies.append(1)
        
        else:
            raise ValueError(f"Expected string at position {i}, got {type(args[i])}")
    
    # Check all orfs are in the proteome_dict
    for orf in orf_list:
        if orf not in proteome_dict:
            raise KeyError(f"ORF {orf} is missing from proteome_dict")

    # Check all num_copies are integers and > 0
    for num in orf_num_copies:
        if not isinstance(num, int) or num <= 0:
            raise ValueError(f"Number of copies must be a positive integer, got {num}")

    # Generate the json
    if proteome_label is not None:
        proteome_label += "_" 
    else:
        proteome_label = ""
    header = proteome_label + "__".join(orf_list)

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