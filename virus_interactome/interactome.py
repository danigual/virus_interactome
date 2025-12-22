import os
import tqdm
import pandas as pd
import yaml
import json
import concurrent.futures
import warnings

from glob import glob
from sklearn.cluster import DBSCAN
from functools import partial

from .utils import load_json, load_boltz_input, check_sequence_validity
from .proteome_manager import ProteomeManager
# from .molecule import MoleculeModel
from .metrics import calculate_all_metrics
from .plotting import plot_boxplots, plot_iptm_vs_ptm, plot_pae_clusters

def check_input(seq_list, residue_threshold: int = 5000):
    total_res = 0
    if len(seq_list) == 0:
        return False, "Sequence list cannot be empty."
    
    for chain_id, seq, count in seq_list:
        if count < 1:
            return False, "Count needs to be at least 1."
        seq_clean = check_sequence_validity(seq)
        # _validate_seq(seq_clean, strict=strict)
        if not seq_clean:
            return False, f"{chain_id} is not a valid protein sequnce."
        total_res += len(seq) * count

    if total_res > residue_threshold:
        msg = (
            f"Total residues {total_res} exceed recommended maximum ({residue_threshold})."
        )
        warnings.warn(msg, category=UserWarning, stacklevel=2)
    
    return True, None

class InteractomeWriter:
    def __init__(self, proteome_a: str | ProteomeManager, proteome_b: str | ProteomeManager | None = None):
        self.proteome_a = None
        self.proteome_b = None
        self.mode = "intra" # intra (intrainteractome) or inter (interactome between two proteomes)

        if isinstance(proteome_a, str):
            self.proteome_a = ProteomeManager(proteome_a)
        elif isinstance(proteome_a, ProteomeManager):
            self.proteome_a = proteome_a
        else:
            raise ValueError("proteome_a must be a string path or a ProteomeManager instance")
        
        if isinstance(proteome_b, str):
            self.proteome_b = ProteomeManager(proteome_b)
            self.mode = "inter"
        elif isinstance(proteome_b, ProteomeManager):
            self.proteome_b = proteome_b
            self.mode = "inter"
        else:
            self.mode = "intra"
        
        self.job_info = self.check_run()
    
    @staticmethod
    def get_af3_input(seq_list: list, job_name: str = "AF3_job", residue_threshold: int = 5000, save_path: str | None = None):
        '''
        Docstring for get_af3_input
        
        Parameters
        ----------
        seq_list : list 
            List tuples of type [(id, seq, count)] 
            e.g., [("A", "MSE...", 1), ("B", "AAAA", 2)]
        param save_path : str
            Path to save the job in json format.
        
        Returns
        -------
        Dictionary of the format 
            {
            "name": <name>,
            "sequences": [{
                "proteinChain": {
                    "count": <count>,
                    "sequence": <sequence>
                    }
                }]
            }
        '''
        is_good_input, err_msg = check_input(seq_list, residue_threshold=residue_threshold)
        
        if is_good_input == False:
            raise ValueError(err_msg)
        
        sequences = []

        for chain_id, seq, count in seq_list:
            sequences.append(
                {"proteinChain": {"id": chain_id, "count": count, "sequence": seq}}
            )

        data = {
            "name": job_name,
            "sequences": sequences
            }
        
        if save_path is not None:
            with open(save_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)

        return data

    @staticmethod
    def get_boltz2_input(seq_list: list, residue_threshold=1600, save_path: str | None = None):
        '''
        Docstring for get_boltz2_input
        
        Parameters
        ----------
        seq_list : list 
            List tuples of type [(id, seq, count)] 
        param save_path : str
            Path to save the job in json format.
        '''
        is_good_input, err_msg = check_input(seq_list, residue_threshold=residue_threshold)
      
        if is_good_input == False:
            raise ValueError(err_msg)
        
        id_list = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
        seqs2yaml = []
        chain_idx = 0

        for chain_id, seq, counts in seq_list:
            tmp_job = {"protein": {"id": id_list[chain_idx], "sequence": seq, }}
            if counts>1:
                multiple_chains = ",".join(id_list[chain_idx + 1 : chain_idx + counts])
                tmp_job["protein"]["multiple_chains"] = multiple_chains
            chain_idx += counts
            seqs2yaml.append(tmp_job)

        data = {
            "version": 1,
            "sequences": seqs2yaml
        }

        if save_path is not None:
            with open(save_path, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

        return data

class InteractomeRunner:
    def __init__(self, path_of_inputs, path_of_outputs, mode="boltz2"):
        self._available_modes = ["af3", "boltz2"]
        if mode not in self._available_modes:
            raise ValueError(f"Mode should be in {' '.join(self._available_modes)}")
        else:
            self.mode = mode
    
        if self.mode == "af3":
            self.inputs = glob(f"{path_of_inputs}/*json")
        elif self.mode == "boltz2":
            self.inputs = glob(f"{path_of_inputs}/*yaml")

        self.path_of_inputs = path_of_inputs
        self.path_of_outputs = path_of_outputs
        self.outputs = glob(f"{path_of_outputs}/*/")

        self.parse_job_dictionary = {
            "af3": self._parse_af3_job,
            "boltz2": self._parse_boltz2_job,
        }
        self.status = self.check_run()

    def _parse_af3_job(self, input_json):
        return load_json(input_json)

    def _parse_boltz2_job(self, input_yaml):
        return load_boltz_input(input_yaml)
    
    def check_run(self):
        parse_job = self.parse_job_dictionary[self.mode]

        all_job_info = []
        for model_input in self.inputs:
            ## Parse de job
            batch_jobs = parse_job(model_input)
            ## AF3 may return several, but Boltz2 only one

            ## For each job we get: job_id, #proteins, #aa
            for tmp_job in batch_jobs:
                all_job_info.append(
                    [tmp_job.get("name"), ## Job name
                     sum([i.get("proteinChain").get("count") for i in tmp_job.get("sequences")]), ## Number of chains
                     sum([len(i.get("proteinChain").get("sequence")) for i in tmp_job.get("sequences")]), ## Total number of residues
                     ]
                )
        df = pd.DataFrame(all_job_info, columns = ["PPI", "num_chain", "num_aa"])

        ## Check number of models
        num_models = []
        for tmp_job_name, _, _ in all_job_info:
            tmp_num_models = len(glob(f"{self.path_of_outputs}/{tmp_job_name}/*model*cif"))
            num_models.append(tmp_num_models)
        
        df.loc[:, "num_models"] = num_models
        df.loc[:, "status"] = "PENDING"

        mode_num_models = int(df.num_models.mode().values)
        df.loc[df.num_models == mode_num_models, "status"] = "COMPLETED"
        df.loc[df.num_models != mode_num_models, "status"] = "PENDING"
        df.loc[df.num_models == 0, "status"] = "FAILED"

        custom_order = ['FAILED', 'PENDING', 'COMPLETED']
        df['status'] = pd.Categorical(df['status'], categories=custom_order, ordered=True)
        df.sort_values(['status', 'num_aa'], inplace=True)
        
        return df
    
    def write_status(self, file_name: str | None = None):
        if file_name is None:
            file_name = f"{self.path_of_inputs}/JOB_STATUS.csv"
        self.status.to_csv(file_name, index=False)## Write status 
    
    def write_missing_jobs(self, output_path: str | None = None):
        import shutil
        if self.mode == "af3":
            raise ValueError("This functions is only supported for Boltz2 runs... for the moment")
        
        tmp_jobs = self.status.loc[ self.status.status != "COMPLETED", "PPI"].values

        if len(tmp_jobs) == 0:
            raise Warning("No pending jobs. Exiting doing nothing")
        
        if output_path is None:
            output_path = f"{self.path_of_inputs}/../input_missing/"
        os.makedirs(output_path, exist_ok=True)
    
        print(f"Safe missing jobs to {output_path}")
        for ppi_id in tmp_jobs:
            shutil.copy(f"{self.path_of_inputs}/{ppi_id}.yaml", f"{output_path}/{ppi_id}.yaml")

class InteractomeProcessor:
    def __init__(self, model_list: list[str]):
        self.model_list = model_list
        self.df_het = None
        self.df_hom = None
        self.cluster_data = None

        self.process_models()
    
    def process_ppi(model_file: str, model_type: str = "AF3", prefix: str = "")-> tuple[dict, pd.DataFrame]:
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
        tuple
            A tuple containing:
            - A dictionary with summary metrics.
            - A DataFrame with cluster details.
        """
        # Parse metadata from the file path
        dir_name = os.path.dirname(model_file)
        base_name = os.path.basename(model_file).replace(".cif", "").replace(".pdb", "")
        folder_path = os.path.dirname(dir_name)
        ppi_id = os.path.basename(folder_path)
        model_number = int(base_name.split("_")[-1].replace("model_",""))

        # Create MoleculeModel instance
        if model_type == "AF3":
            molecule_model = MoleculeModel.from_af3(model_file)
        elif model_type == "boltz":
            molecule_model = MoleculeModel.from_boltz(model_file)
        
        # Calculate all metrics
        all_metrics = calculate_all_metrics(molecule_model)

        # Generate cluster plot
        plot_pae_clusters(molecule_model.pae, molecule_model.chain_by_res, output_path=dir_name, prefix=prefix)

        # Return summary metrics and cluster details
        return {"PPI": ppi_id, "ORF_A": ppi_id.split("_")[0], "ORF_B":ppi_id.split("_")[1], "Folder": folder_path, 
                "Model_num": model_number, 
                "iPTM": molecule_model.iptm, 
                "pTM": molecule_model.ptm, 
                # "chain_length_A": np.sum(chain_by_res == "A"), "chain_length_B":np.sum(chain_by_res == "B"), 
                **all_metrics
                }, all_metrics.get("cluster_info", pd.DataFrame())
    
    def process_models(self, output_path: str = ".", model_type: str = "AF3", prefix: str = "", **kwargs):
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

class InteractomeAnalyzer:
    def __init__(self):
        pass

    def calculate_network(self):
        pass

    def basic_plots(self):
        pass

    def protein_peptide_analysis(self):
        pass

    def cluster_analysis(self):
        pass
