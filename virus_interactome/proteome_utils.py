
#------------------------------ IMPORTS -------------------------------------
# import json
# import subprocess
import os
import numpy as np
# from pathlib import Path
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from glob import glob
import tqdm
from sklearn.cluster import DBSCAN
from .utils import process_full_data_af3, load_json, process_full_data_boltz
from .plotting import plot_boxplots, plot_iptm_vs_ptm, plot_pae_clusters
from moleculekit.molecule import Molecule
from .metrics import calculate_all_metrics
#------------------------------FUNCTIONS -----------------------------

def cluster_pae (pae_submatrix, threshold:int=15, eps:int=10)-> tuple:
    
    """
    Clusters low PAE regions in a PAE submatrix using DBSCAN.

    This function identifies coordinates in the PAE matrix where the predicted aligned error
    is below a given threshold, and applies DBSCAN clustering to group them. It returns the
    coordinates of low PAE values and their corresponding cluster labels.

    Parameters
    ----------
    pae_submatrix : np.ndarray
        A 2D array representing a subset of the PAE matrix.
    threshold : int, optional
        Maximum PAE value to consider for clustering (default is 15).
    eps : int, optional
        Maximum distance between points in a cluster for DBSCAN (default is 10).

    Returns
    -------
    tuple
        A tuple containing:
        - low_pae_coords (np.ndarray): Coordinates of low PAE values.
        - labels (np.ndarray or list): Cluster labels assigned by DBSCAN.

    Raises
    ------
    ValueError
        If the input matrix is invalid or clustering fails.
    """

    low_pae_coords = np.column_stack(np.where(pae_submatrix < threshold))
    if len(low_pae_coords)>0: ##Maybe we want something higher so we get rid of shit
        #Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=5).fit(low_pae_coords)
        labels = clustering.labels_
    else:
        low_pae_coords, labels = [], []

    return  low_pae_coords, labels

def cluster_info (low_coords, cluster_labels)-> pd.DataFrame:
    
    """
    Extracts and summarizes geometric information from clustered low PAE coordinates.

    This function processes the output of a clustering algorithm (e.g., DBSCAN) applied to low PAE regions.
    For each cluster (excluding noise), it computes bounding box coordinates, percentiles to reduce outlier
    impact, and the cluster center. The results are returned as a pandas DataFrame.

    Parameters
    ----------
    low_coords : np.ndarray
        Array of coordinates (row, column) where PAE values are below a threshold.
    cluster_labels : np.ndarray
        Array of cluster labels assigned to each coordinate.

    Returns
    -------
    pd.DataFrame
        DataFrame containing geometric and statistical information for each cluster.

    Raises
    ------
    ValueError
        If input arrays are mismatched or improperly formatted.
    """

    cluster_info_list = []
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        if label == -1:
            continue  # Ignorar ruido

        cluster_coords = low_coords[cluster_labels == label]

        # Clustering functions (x 1 y 0)
        x_min = np.min(cluster_coords[:, 0])
        y_max = np.max(cluster_coords[:, 1])
        x_max = np.max(cluster_coords[:, 0])
        y_min = np.min(cluster_coords[:, 1])

        cluster_center = np.mean(cluster_coords, axis=0)

        #Percentiles to reduce the impact of outliers
        top_percentile = np.percentile(cluster_coords, 99.5, axis=0)
        lower_percentile = np.percentile(cluster_coords, .5, axis=0)
        per_x_min = lower_percentile[0]
        per_y_min = lower_percentile[1]
        per_x_max = top_percentile[0]
        per_y_max = top_percentile[1]

        cluster_info_list.append({
            "cluster_id": label,
            "num_points": len(cluster_coords),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "percentile_x_min": per_x_min,
            "percentile_x_max": per_x_max,
            "percentile_y_min": per_y_min,
            "percentile_y_max": per_y_max,
            "center_x": cluster_center[1],
        })

    # Generate the pd.df
    cluster_data_from_model = pd.DataFrame(cluster_info_list)


    
    return cluster_data_from_model
    
def process_ppi(model_file: str, model_type="af3", prefix="")-> tuple[list, pd.DataFrame]:
    """
    Processes an AlphaFold3 CIF model file and extracts structural and confidence metrics.

    This function parses metadata from the CIF file path, loads associated JSON files containing
    summary confidences and full structural data, and computes various metrics such as pLDDT means,
    PAE means (overall and per chain), and clustering information from the PAE submatrix between chains.
    It also generates a cluster plot and returns both summary metrics and cluster details.

    Parameters
    ----------
    model_file : str
        Path to the CIF file generated by AlphaFold3 or Boltz2.

    Returns
    -------
    list
        A list containing summary metrics:
        [ppi_id, orf_a, orf_b, folder_path, model_number, fraction_disordered, iPTM, pTM,
        chain_length_A, chain_length_B, plddt_mean, plddt_mean_chain_A, plddt_mean_chain_B,
        mean_pae, mean_pae_chain_A, mean_pae_chain_B, mean_pae_chain_A_B]
    pd.DataFrame
        DataFrame containing cluster information extracted from the PAE submatrix.

    Raises
    ------
    FileNotFoundError
        If associated JSON files are missing.
    KeyError
        If expected keys are missing in the JSON data.
    ValueError
        If the CIF file path format is invalid or model number cannot be extracted.
    """
    print(model_file)
    # Extract folder name and model number
    ppi_id = model_file.split("/")[-2].replace(prefix, "") ## This should be general
    model_number = int(model_file.split("/")[-1].split("_")[-1].replace(".cif", ""))
    orf_a, orf_b = ppi_id.split("__")
    folder_path = "/".join(model_file.split("/")[1:-1])
    if model_type == "af3":
        # Name for summary confidences file
        summary_confidences_file = model_file.replace(".cif",".json").replace("model","summary_confidences")
        # Name for full data file 
        full_data_file = summary_confidences_file.replace("summary_confidences","full_data")
        # Open summary confidences file
        summary_data = load_json(summary_confidences_file)
        
        # with open (summary_confidences_file,"r") as sc:
        #     # Load json data
        #     summary_data = json.load(sc)
        #     # Extract fraction disordered, iPTM and pTM
        #     fraction_disordered = summary_data["fraction_disordered"]
        # Process full data file (json file)
        full_data = process_full_data_af3(full_data_file)
    elif model_type == "boltz2":
        summary_file = model_file.replace(".cif", ".json").replace(f"{ppi_id}_model", f"confidence_{ppi_id}_model")
        summary_data = load_json(summary_file)
        full_data = process_full_data_boltz(os.path.dirname(model_file))
    
    # import pdb;pdb.set_trace()
    iPTM = summary_data["iptm"]
    pTM = summary_data["ptm"]

    mol = Molecule(model_file)
    all_metrics = calculate_all_metrics(mol, full_data)
    # Extract the length of chain A and chains B
    # chain_lenghts = full_data["chain_lengths"]
    # chain_lenght_A = chain_lenghts["A"]
    # chain_lenght_B = chain_lenghts["B"]

    # ## Add plddt mean 
    # plddt_mean = np.mean(full_data["atom_plddts"])

    # ## Add plddt_mean_chain_A
    # ## Extract the start and end of atoms boundaries for chain A
    # start_a_atoms, end_a_atoms = full_data["chain_boundaries_by_atom"][0]
    # end_a_atoms += 1
    # ## Indexing in atom_plddts using the boundaries in order to calculate the mean
    # plddt_mean_chain_A = np.mean(full_data["atom_plddts"][start_a_atoms:end_a_atoms])
    
    # ## Add plddt_mean_chain_B
    # ## Extract the start and end of atoms boundaries for chain B
    # start_b_atoms, end_b_atoms = full_data["chain_boundaries_by_atom"][1]
    # end_b_atoms += 1
    # ## Indexing in atom_plddts using the boundaries in oder to calculate the mean
    # plddt_mean_chain_B = np.mean(full_data["atom_plddts"][start_b_atoms:end_b_atoms])
    
    # ## Add mean PAE
    # mean_pae = np.mean(full_data["pae"])
    
    # ## Add mean PAE chain A
    # ## Extract the start and end of residues boundaries for chain A
    # start_a_residues, end_a_residues = full_data["chain_boundaries"][0]
    # end_a_residues += 1
    # ## Generate the pae matrix for chain A 
    # pae_chain_A = full_data["pae"][start_a_residues:end_a_residues,start_a_residues:end_a_residues]
    # mean_pae_chain_A = np.mean(pae_chain_A)

    # ## Add mean PAE chain B
    # ## Extract the start and end of residues boundaries for chain B
    # start_b_residues, end_b_residues = full_data["chain_boundaries"][1]
    # end_b_residues += 1
    # ## Generate the pae matrix for chain B
    # pae_chain_B = full_data["pae"][start_b_residues:end_b_residues,start_b_residues:end_b_residues]
    # mean_pae_chain_B = np.mean(pae_chain_B)

    # ## Add mean PAE pair A-B
    # ## Generate the pae matrix for chain A-B
    # pae_chain_A_B = full_data["pae"][start_a_residues:end_a_residues,start_b_residues:end_b_residues]
    # mean_pae_chain_A_B = np.mean(pae_chain_A_B)   
    

    ## Here, we do cluster of PAE AB submatrix, we already have the submatrix
    chain_by_res = full_data["token_chain_ids"]
    pae = full_data["pae"]
    pae_submatrix_1 = pae[chain_by_res == "A"][:, chain_by_res == "B"]
    pae_submatrix_2 = pae[chain_by_res == "B"][:, chain_by_res == "A"].T
    submatrix = np.mean([pae_submatrix_1, pae_submatrix_2], axis=0) ## Maybe we want the mean?
    # submatrix = np.min([pae_submatrix_1, pae_submatrix_2], axis=0) ## Maybe we want the mean?

    ## Clustering
    low_coords, cluster_labels = cluster_pae(submatrix)

    ## here we do the plot of the pae clusters
    plot_pae_clusters(submatrix,low_coords, cluster_labels, save_name=model_file.replace(".cif", "_cluster.png")) 
    
    cluster_data = cluster_info(low_coords=low_coords, cluster_labels=cluster_labels)
    # Incluir en el df de los clusters el ppi_id
    if len(cluster_data)>0:
        cluster_data.loc[:, "PPI"] = ppi_id 
        cluster_data.loc[:, "model_num"] = model_number 
        cluster_data.loc[:, "path"] = model_file 

    return {"PPI": ppi_id, "ORF_A": orf_a, "ORF_B":orf_b, "path": folder_path, 
            "model_num": model_number, 
            # "Fraction_disordered": fraction_disordered, 
            "iPTM": iPTM, 
            "pTM": pTM, "chain_length_A": np.sum(chain_by_res == "A"), "chain_length_B":np.sum(chain_by_res == "B"), 
            **all_metrics
            }, cluster_data

## Creating simplified dataframe with info of individual proteins
def get_info_for_proteins(df)-> pd.DataFrame:
    """
    Extracts per-protein metrics from a DataFrame containing chain-level information.

    This function takes a DataFrame with metrics for chain A and chain B of protein-protein interactions,
    renames the columns to unify the format, and concatenates the data to produce a per-ORF summary.
    Useful for downstream analysis of individual protein metrics across models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for ORF_A, ORF_B, PPI, Model_num, and chain-specific metrics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with unified columns:
        ["ORF", "PPI", "Model_num", "pLDDT_mean_ORF", "pae_mean_ORF"]

    Raises
    ------
    KeyError
        If expected columns are missing in the input DataFrame.
    """

    info_from_chain_a = df.loc[:, ["ORF_A", "PPI", "Model_num", "pLDDT_mean_A", "pae_mean_A"]]
    info_from_chain_b = df.loc[:, ["ORF_B", "PPI", "Model_num", "pLDDT_mean_B", "pae_mean_B"]]
    info_from_chain_a.columns = ["ORF", "PPI", "Model_num", "pLDDT_mean_ORF", "pae_mean_ORF"]
    info_from_chain_b.columns = ["ORF", "PPI", "Model_num", "pLDDT_mean_ORF", "pae_mean_ORF"]

    by_protein_df = pd.concat([info_from_chain_a, info_from_chain_b],  ignore_index=True) 

    return by_protein_df   
   

def process_interactome(folder_path:str, output_path:str, model_type="af3", prefix="", **kwargs):    
    
    """
    Processes an AlphaFold3 interactome folder and generates summary data, cluster analysis, and visualizations.

    This function scans a directory for CIF model files, extracts structural and confidence metrics from each,
    aggregates the results into summary DataFrames, and saves them as CSV files. It also generates boxplots
    and scatterplots for pLDDT, PAE, and ipTM vs pTM metrics.

    Parameters
    ----------
    folder_path : str
        Path to the root directory containing AlphaFold3 model subfolders with CIF files.
    output_path : str
        Directory where output CSVs and plots will be saved.
    **kwargs : dict
        Optional keyword arguments passed to the ProcessPoolExecutor (e.g., max_workers).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no CIF files are found in the specified folder.
    FileNotFoundError
        If required JSON files associated with CIF models are missing.
    """
    ## TODO:Check and create the output path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/plots/", exist_ok=True)

    # Extract the .cif files (list with all cif files)
    all_ppi_models = []
    if model_type == "af3":
        all_ppi_models = np.array(glob(f"{folder_path}/{prefix}*/*cif")) ## For testing  
    elif model_type == "boltz2":
        all_ppi_models = np.array(glob(f"{folder_path}/predictions/{prefix}*/*cif"))
    # all_cif_files = glob(f"{folder_path}/*/*cif")

    if len(all_ppi_models)==0:
        raise ValueError(f"""No files found in the directory {folder_path}.
                         Please make sure it follows the AF3 output directory convention""")
    
    ##Load output.csv and filter out processed data
    interactome_df = pd.DataFrame()
    clusters_df = pd.DataFrame()
    if os.path.exists(f'{output_path}/interactome_data.csv',):
        print("Loading existing data...")
        interactome_df = pd.read_csv(f'{output_path}/interactome_data.csv')
        folder_names = pd.Series([os.path.dirname(i) for i in all_ppi_models])
        all_ppi_models = all_ppi_models[~folder_names.isin(interactome_df.Folder)]
        clusters_df = pd.read_csv(f'{output_path}/clusters_data.csv')
 
    # Paralelización
    #List of tuples -> every tuple has list,df
    ##TODO: skip this is .csv exists
    interactome_df_list = []
    cluster_df_list = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
        worker = partial(process_ppi, model_type=model_type, prefix=prefix)
         # map devuelve los resultados en orden de la lista
        
        # all_args = [(model_file, model_type, prefix) for model_file in all_ppi_models]
        for res in tqdm.tqdm(executor.map(worker, all_ppi_models)): #For testing
        # for res in tqdm.tqdm(executor.map(process_ppi, all_args)): #For testing
            interactome_df_list.append(res[0])
            cluster_df_list.append(res[1])
    # Create the df with the info of all PPIs
    # interactome_df_list = [list[0] for list in results] 
    # interactome_df = pd.DataFrame(interactome_df_list,
    #                  columns=["PPI", "ORF_A", "ORF_B", "Folder", "Model_num","Fraction_disordered","iPTM","pTM",
    #                          "chain_lenght_A","chain_lenght_B", 
    #                          #    "contact_probs",
    #                          "plddt_mean", "plddt_mean_chain_A","plddt_mean_chain_B",
    #                          "mean_pae", "mean_pae_chain_A", "mean_pae_chain_B",
    #                          #    "min_pae_chain_A", "min_pae_chain_B",
    #                          "mean_pae_chain_A_B"])
    interactome_df = pd.concat([interactome_df, pd.DataFrame.from_dict(interactome_df_list)], ignore_index=True)
    # interactome_df = pd.DataFrame.from_dict(interactome_df_list)
    # interactome_df.reset_index(inplace=True, drop=True)

    # Join the dfs of the clusters in a single df
    # non_empty_cluster_info = filter(lambda x: len(x)>0, cluster_df_list)
    if len(cluster_df_list) > 0:
        filtered_cluster_df_list = [ i for i in cluster_df_list if i.shape[0]>0 ]
        clusters_df = pd.concat([clusters_df, pd.concat(filtered_cluster_df_list)], ignore_index=True)
    # clusters_df = pd.concat(non_empty_cluster_info, ignore_index=True)
    # clusters_df = pd.concat([df[1] for df in results], ignore_index=True)

    # Save dfs to .csv
    interactome_df.to_csv(f'{output_path}/interactome_data.csv', index=False)
    clusters_df.to_csv(f'{output_path}/clusters_data.csv', index=False)
    by_protein_df = get_info_for_proteins(interactome_df)
    by_protein_df.to_csv(f'{output_path}/interactome_data_by_protein.csv', index=False) 

    ## Plotting boxplots
    output_folder = f"{output_path}/plots"
    plddt_array, labels_array1 = process_boxplot_data(by_protein_df, "pLDDT_mean_ORF")
    pae_array, labels_array2 = process_boxplot_data(by_protein_df, "pae_mean_ORF")
    plot_boxplots("plddt",plddt_array, labels_array1, output_path=output_folder)
    plot_boxplots("pae",pae_array,labels_array2, output_path=output_folder)

    ## Plotting scatterplots
    plot_iptm_vs_ptm(interactome_df, output_path=output_folder)


def process_boxplot_data(df, value_column:str)->tuple[np.ndarray, np.ndarray]:

    """
    Prepares data for boxplot visualization by grouping values per ORF.

    This function takes a DataFrame and a column name (e.g., pLDDT or PAE values),
    groups the values by ORF, computes the mean for sorting, and returns the data
    in a format suitable for plotting: a matrix of values and a corresponding list of labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns "ORF" and the specified value column.
    value_column : str
        Name of the column containing the values to be grouped and plotted.

    Returns
    -------
    tuple
        A tuple containing:
        - value_matrix : np.ndarray
            Array of lists, each containing values for an ORF, sorted by mean value.
        - labels_list : np.ndarray
            Array of ORF names corresponding to the sorted value_matrix.

    Raises
    ------
    KeyError
        If the specified value_column or "ORF" is missing from the DataFrame.
    """

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