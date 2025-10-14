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