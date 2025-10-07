
#------------------------------ IMPORTS -------------------------------------
import json
import os
import numpy as np
import concurrent.futures
import pandas as pd
from glob import glob
import tqdm
from Bio import SeqIO
from itertools import combinations
from utils import process_full_data_af3

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
    
def process_cif_file (cif_file: str):
    
    # Extract folder name and model number
    folder_path = "/".join(cif_file.split("/")[1:-1])
    ppi_id = cif_file.split("/")[-2].replace("adv5_", "")
    model_number = int(cif_file.split("/")[-1].split("_")[-1].replace(".cif", ""))
    # Name for summary confidences file
    summary_confidences_file = cif_file.replace(".cif",".json").replace("model","summary_confidences")
    # Name for full data file 
    full_data_file = summary_confidences_file.replace("summary_confidences","full_data")
    # Open summary confidences file
    with open (summary_confidences_file,"r") as sc:
        # Load json data
        summary_data = json.load(sc)
        # Extract fraction disordered, iPTM and pTM
        fraction_disordered = summary_data["fraction_disordered"]
        iPTM = summary_data["iptm"]
        pTM = summary_data["ptm"]
    # Process full data file (json file)
    full_data = process_full_data_af3(full_data_file)
  
    # Extract the length of chain A and chains B
    chain_lenghts = full_data["chain_lengths"]
    chain_lenght_A = chain_lenghts["A"]
    chain_lenght_B = chain_lenghts["B"]

    ## Add plddt mean 
    plddt_mean = np.mean(full_data["atom_plddts"])

    ## Add plddt_mean_chain_A
    ## Extract the start and end of atoms boundaries for chain A
    start_a_atoms, end_a_atoms = full_data["chain_boundaries_by_atom"][0]
    end_a_atoms += 1
    ## Indexing in atom_plddts using the boundaries in oder to calculate the mean
    plddt_mean_chain_A = np.mean(full_data["atom_plddts"][start_a_atoms:end_a_atoms])
    
    ## Add plddt_mean_chain_B
    ## Extract the start and end of atoms boundaries for chain B
    start_b_atoms, end_b_atoms = full_data["chain_boundaries_by_atom"][1]
    end_b_atoms += 1
    ## Indexing in atom_plddts using the boundaries in oder to calculate the mean
    plddt_mean_chain_B = np.mean(full_data["atom_plddts"][start_b_atoms:end_b_atoms])
    
    ## Add mean PAE
    mean_pae = np.mean(full_data["pae"])
    
    ## Add mean PAE chain A
    ## Extract the start and end of residues boundaries for chain A
    start_a_residues, end_a_residues = full_data["chain_boundaries"][0]
    end_a_residues += 1
    ## Generate the pae matrix for chain A 
    pae_chain_A = full_data["pae"][start_a_residues:end_a_residues,start_a_residues:end_a_residues]
    mean_pae_chain_A = np.mean(pae_chain_A)

    ## Add mean PAE chain B
    ## Extract the start and end of residues boundaries for chain B
    start_b_residues, end_b_residues = full_data["chain_boundaries"][1]
    end_b_residues += 1
    ## Generate the pae matrix for chain B
    pae_chain_B = full_data["pae"][start_b_residues:end_b_residues,start_b_residues:end_b_residues]
    mean_pae_chain_B = np.mean(pae_chain_B)

    
    ## Add mean PAE pair A-B
    ## Generate the pae matrix for chain A-B
    pae_chain_A_B = full_data["pae"][start_a_residues:end_a_residues,start_b_residues:end_b_residues]
    mean_pae_chain_A_B = np.mean(pae_chain_A_B)   
    ## Add minimun PAE pair A-B
    
    # Append info to the list
    return [ppi_id, folder_path, model_number, fraction_disordered, iPTM, 
                        pTM, chain_lenght_A, chain_lenght_B, 
                        # contact_probs,
                         plddt_mean, plddt_mean_chain_A, plddt_mean_chain_B,
                          mean_pae, mean_pae_chain_A, mean_pae_chain_B,
                        #   min_pae_chain_A, min_pae_chain_B, 
                          mean_pae_chain_A_B]
   
    

def process_interactome(folder_path:str, output_name:str, **kwargs):    
    # Extract the .cif files
    all_cif_files = glob(f"{folder_path}/*/*cif")

    # Paralelización

     # lista de tuplas
    results = []
    with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
         # map devuelve los resultados en orden de la lista
        for res in tqdm.tqdm(executor.map(process_cif_file, all_cif_files)): #For testing
             results.append(res)
     # Create the df
    df = pd.DataFrame(results,
                     columns=["PPI", "Folder", "Model_num","Fraction_disordered","iPTM","pTM",
                             "chain_lenght_A","chain_lenght_B", 
                             #    "contact_probs",
                             "plddt_mean", "plddt_mean_chain_A","plddt_mean_chain_B",
                             "mean_pae", "mean_pae_chain_A", "mean_pae_chain_B",
                             #    "min_pae_chain_A", "min_pae_chain_B",
                             "mean_pae_chain_A_B"])
    df.reset_index(inplace=True, drop=True)
    # Save df to .csv
    df.to_csv(output_name)
    return df 




