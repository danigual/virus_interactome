
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
from .utils import process_full_data_af3
from .plotting import plot_boxplots, plot_iptm_vs_ptm
#------------------------------FUNCTIONS -----------------------------

def process_cif_file (cif_file: str):
    
    # Extract folder name and model number
    folder_path = "/".join(cif_file.split("/")[1:-1])
    ppi_id = cif_file.split("/")[-2].replace("adv5_", "")
    orf_a, orf_b = ppi_id.split("__")
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
    return [ppi_id, orf_a, orf_b, folder_path, model_number, fraction_disordered, iPTM, 
                        pTM, chain_lenght_A, chain_lenght_B, 
                        # contact_probs,
                         plddt_mean, plddt_mean_chain_A, plddt_mean_chain_B,
                          mean_pae, mean_pae_chain_A, mean_pae_chain_B,
                        #   min_pae_chain_A, min_pae_chain_B, 
                          mean_pae_chain_A_B]

## Creating simplfied dataframe with info of individual proteins
def get_info_for_proteins(df):
    info_from_chain_a = df.loc[:, ["ORF_A", "PPI", "Model_num", 
                                   "plddt_mean_chain_A", "mean_pae_chain_A"]]

    info_from_chain_b = df.loc[:, ["ORF_B", "PPI", "Model_num", 
                                   "plddt_mean_chain_B", "mean_pae_chain_B"]]

    info_from_chain_a.columns = ["ORF", "PPI", "Model_num", 
                                 "plddt_mean_ORF", "pae_mean_ORF"]

    info_from_chain_b.columns = ["ORF", "PPI", "Model_num", 
                                 "plddt_mean_ORF", "pae_mean_ORF"]
    by_protein_df = pd.concat([info_from_chain_a, info_from_chain_b], 
                              ignore_index=True) 
    return by_protein_df   
   

def process_interactome(folder_path:str, output_path:str, **kwargs):    
    ## TODO:Check and create the output path

    # Extract the .cif files
    all_cif_files = glob(f"{folder_path}/*/*cif")

    ## TODO:Check we are working with a non-empty list

    # Paralelización

    #List of tuples
    ##TODO: skip this is .csv exists
    results = []
    with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
         # map devuelve los resultados en orden de la lista
        for res in tqdm.tqdm(executor.map(process_cif_file, all_cif_files)): #For testing
             results.append(res)
    # Create the df with the info of all PPIs
    interactome_df = pd.DataFrame(results,
                     columns=["PPI", "ORF_A", "ORF_B", "Folder", "Model_num","Fraction_disordered","iPTM","pTM",
                             "chain_lenght_A","chain_lenght_B", 
                             #    "contact_probs",
                             "plddt_mean", "plddt_mean_chain_A","plddt_mean_chain_B",
                             "mean_pae", "mean_pae_chain_A", "mean_pae_chain_B",
                             #    "min_pae_chain_A", "min_pae_chain_B",
                             "mean_pae_chain_A_B"])
    interactome_df.reset_index(inplace=True, drop=True)
    # Save dfs to .csv
    interactome_df.to_csv(f'{output_path}/interactome_data.csv', index=False)
    by_protein_df = get_info_for_proteins(interactome_df)
    by_protein_df.to_csv(f'{output_path}/interactome_data_by_protein.csv', index=False) 

    ## Plotting boxplots
    output_folder = f"{output_path}/plots"
    plddt_array, labels_array1 = process_boxplot_data(by_protein_df, "plddt_mean")
    pae_array, labels_array2 = process_boxplot_data(by_protein_df, "mean_pae")
    plot_boxplots("plddt",plddt_array, labels_array1, output_path=output_folder)
    plot_boxplots("pae",pae_array,labels_array2, output_path=output_folder)

    ## Plotting scatterplots
    plot_iptm_vs_ptm(interactome_df, output_path=output_folder)



def process_boxplot_data(df, value_column:str):
        '''
        Receives a df, the orf column and the plddt column. It returns a list of 
        of lists of the different pae or plddts values by protein
            ARGS IN:
            -------
            df: pd.df
            value_column: pd.Series

            ARGS OUT:
            --------
            value_matrix: np.arrray (array of lists with values per ORF)
            labels_list: np.array (array of ORF names, ordered by mean value)


        '''
        grouped_values = []
        orf_labels = []
        for orf in df["ORF"].unique():
            tmp = list(df.loc[df["ORF"] == orf , value_column])
            grouped_values.append(tmp) 
            orf_labels.append(orf)
            
        mean_values = [np.mean(v) for v in grouped_values]
        sort_idx = np.argsort(mean_values)    
        value_matrix = np.array(grouped_values, dtype=object)[sort_idx]    
        orf_labels = np.array(orf_labels)[sort_idx]
    
        return value_matrix, orf_labels



        


   




