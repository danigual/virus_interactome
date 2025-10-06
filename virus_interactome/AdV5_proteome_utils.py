
#------------------------------ IMPORTS -------------------------------------

from Bio import SeqIO
from itertools import combinations
import json
import os

#------------------------------FUNCTIONS -----------------------------

def load_proteome (fasta_file:str):
    ''' Receives a fasta file and extracts the ID-sequence
            ARGS IN:
                fasta_file (.fasta)
            ARGS OUT:
                proteins_dic (dictionary): key-value, id-aas_sequence
     '''

    proteins_dic = {}
    for protein in SeqIO.parse(fasta_file,"fasta"):
        complete_id = protein.id
        fractioned_id = complete_id.split('|')
        short_id = fractioned_id[0]
        sequence = str (protein.seq)
        proteins_dic[short_id] = sequence
    return proteins_dic


def get_af3_input(orf1, orf2, proteome_dict):
    return {
            "name": f"AdV5_{orf1}__{orf2}",
            "sequences":[
                {
                    "proteinChain": {
                        "count": 1,
                        "sequence": f"{proteome_dict[orf1]}"
                    }
                }, {
                    "proteinChain": {
                        "count": 1,
                        "sequence": f"{proteome_dict[orf2]}"
                                }
                }
            ]
        }

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
        
        tmp_job = get_af3_input(orf1, orf2, proteome_dict=proteome_dict)
        tmp_batch.append(tmp_job)
    
    ## Writing the remaining
    tmp_output_name = os.path.join(outputdir, f"{file_idx}.json")
    with open(tmp_output_name, "w") as f:
        json.dump(tmp_batch, f, indent=4)
    
   
    






