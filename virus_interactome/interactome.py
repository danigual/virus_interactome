import os
import tqdm
import pandas as pd
import numpy as np
import yaml
import json
import concurrent.futures
import warnings
import csv

from glob import glob
from sklearn.cluster import DBSCAN
from functools import partial
from itertools import combinations, product
from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path
from moleculekit.molecule import Molecule

from .utils import load_json, load_boltz_input, check_sequence_validity, process_full_data_af3, process_full_data_boltz
from .proteome_manager import ProteomeManager
# from .proteome_utils import cluster_pae
# from .molecule import MoleculeModel
from .metrics import calculate_all_metrics
from .plotting import plot_boxplots, plot_iptm_vs_ptm, plot_pae_clusters, plot_paes, plot_plddt

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
        
        # self.job_info = self.check_run()
    
    def generate_intra_pairs(self):
        """
        Generate unordered heteromeric pairs (A,B) with A != B from proteome A.
        No homomers are included here; for homomers use `generate_homo_pairs()`.
        Order is canonical (i < j) to avoid duplicates.
        """

        if self.mode != "inter" or self.proteome_b is None:
            raise ValueError("generate_inter_pairs() requires 'inter' mode with a valid proteome_b.")

        ids = list(self.proteome_a.sequences.keys())
        # Unique unordered pairs without self-pairs
        return combinations(ids, 2)

    def generate_inter_pairs(self):
        """
        Generate the full cartesian product between proteome A and proteome B.
        Requires 'inter' mode (i.e., proteome_b is provided).
        """
        if self.mode != "inter" or self.proteome_b is None:
            raise ValueError("generate_inter_pairs() requires 'inter' mode with a valid proteome_b.")
        ids_a = list(self.proteome_a.sequences.keys())
        ids_b = list(self.proteome_b.sequences.keys())
        return product(ids_a, ids_b)

    def generate_homo_mers(self, nmin=2, nmax=6):
        """
        Only to be used in the "intra" mode
        
        :param nmin: Minimun number of copies
        :param nmax: Maximun number of copies
        """
        if self.mode == "inter":
            raise ValueError("Homo mers can only be computed in 'intra' mode.")
        if nmin < 2:
            raise ValueError("nmin can not be lower than 2. If you want just 1 copie use the generate_single_run method")
        
        if nmax < nmin:
            raise ValueError("nmax should be lower than nmin")
        ids_a = list(self.proteome_a.sequences.keys())

        for tmp_protein_id in ids_a:
            for num_copies in range(nmin, nmax+1):
                yield(tmp_protein_id, num_copies)
     
    def generate_single_run(
        self,
        source: str = "a",  # 'a' | 'b' | 'both'
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None
    ) -> Iterable[Tuple[str, str, int]]:
        """
        Single-protein jobs: yields (id, sequence, count=1).
        - source='a' -> proteome A
        - source='b' -> proteome B (requires inter mode)
        - source='both' -> A then B (if available)
        """
        if source not in {"a", "b", "both"}:
            raise ValueError("source must be 'a', 'b', or 'both'.")

        if source in {"a", "both"}:
            for pid in (ids_a or self.proteome_a.ids):
                yield (pid, self.proteome_a.sequences[pid], 1)

        if source in {"b", "both"}:
            if self.mode != "inter" or self.proteome_b is None:
                raise ValueError("source='b' or 'both' requires 'inter' mode with a proteome_b.")
            for pid in (ids_b or self.proteome_b.ids):
                yield (pid, self.proteome_b.sequences[pid], 1)

    def write_interactome_jobs(
        self,
        engine: str,  # 'af3' | 'boltz2'
        output_dir: str,
        *,
        mode: str = "intra_pairs",    # 'intra_pairs' | 'inter_pairs' | 'homomers' | 'single'
        include_homo: bool = False,   # optional for intra_pairs (if you want to fold homomers later)
        nmin: int = 2, nmax: int = 6, # for homomers
        counts_map: Optional[Dict[str, int]] = None,
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None,
        af3_threshold: int = 5000,
        af3_batch_size: int = 30,
        boltz_threshold: int = 1600,
        skip_over_threshold: bool = False,
        filename_fmt: str = "{engine}_{name}.{ext}",
        index_name: str = "index.csv",
    ) -> List[dict]:
        """
        Orchestrate job creation and file writing for different generation modes.
        Returns a list of metadata dicts (one per attempted job).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ext = "json" if engine.lower() == "af3" else "yaml" if engine.lower() == "boltz2" else None
        if ext is None:
            raise ValueError("engine must be 'af3' or 'boltz2'")

        # Select generator:
        if mode == "intra_pairs":
            pairs = self.generate_intra_pairs(ids_a=ids_a)  # (idA, idB)
            iterator = (("pair", p) for p in pairs)
        elif mode == "inter_pairs":
            pairs = self.generate_inter_pairs(ids_a=ids_a, ids_b=ids_b)  # (idA, idB)
            iterator = (("pair", p) for p in pairs)
        elif mode == "homomers":
            homes = self.generate_homo_mers(nmin=nmin, nmax=nmax)  # (id, copies)
            iterator = (("homo", h) for h in homes)
        elif mode == "single":
            singles = self.generate_single_run(source="both" if self.mode == "inter" else "a",
                                            ids_a=ids_a, ids_b=ids_b)  # (id, seq, 1)
            iterator = (("single", s) for s in singles)
        else:
            raise ValueError("Unknown mode")

        metas: List[dict] = []
        index_path = Path(output_dir) / index_name
        with open(index_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "engine","mode","name","idA","idB","countA","countB","total_residues","warnings","file_path"
            ])
            w.writeheader()

            for kind, entry in iterator:
                # Build seq_list and job_name
                name = ""
                if kind == "pair":
                    idA, idB = entry
                    # seq_list = seq_list_for_pair(self.proteome_a, self.proteome_b if self.mode == "inter" else None,
                    #                             (idA, idB), counts_map=counts_map)
                    ## TODO: sort idA idB alphabetically or somehow
                    name = f"{idA}__{idB}"
                elif kind == "homo":
                    pid, copies = entry
                    seq_list = [(pid, self.proteome_a.sequences[pid], copies)]
                    name = f"{pid}__{copies}"
                    idA, idB = pid, ""  # single species
                elif kind == "single":
                    pid, seq, cnt = entry
                    seq_list = [(pid, seq, cnt)]
                    name = f"{pid}"
                    idA, idB = pid, ""
                else:
                    raise RuntimeError("Unexpected iterator kind")

                total_res = sum(len("".join(s.split())) * c for _, s, c in seq_list)
                threshold = af3_threshold if engine.lower() == "af3" else boltz_threshold
                over = total_res > threshold

                base_name = filename_fmt.format(engine=engine.lower(), name=name, ext=ext)
                save_path = str(Path(output_dir) / base_name) if not (skip_over_threshold and over) else ""

                if save_path:
                    if engine.lower() == "af3":
                        payload = self.get_af3_input(seq_list, job_name=name,
                                                    save_path=save_path, 
                                                    residue_threshold=af3_threshold)
                    else:
                        payload = self.get_boltz2_input(seq_list, save_path=save_path,
                                                        residue_threshold=boltz_threshold)
                    # warns = payload.get("_warnings", [])
                # else:
                warns = []
                if total_res > threshold:
                    warns = [f"Skipped: total residues {total_res} exceed {threshold}"]

                meta = {
                    "engine": engine.lower(),
                    "mode": mode,
                    "name": name,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                    "warnings": "|".join(warns),
                    "file_path": save_path,
                }
                metas.append(meta)
                w.writerow(meta)
            
        ## TODO: write launch scripts for AF3 and Boltz2

        return metas

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
            id_str = "[" + ','.join(id_list[chain_idx:chain_idx+counts]) + "]"
            # id_str = f"[{','.join(id_list[chain_idx:chain_idx+counts])}]"
            tmp_job = {"protein": {"id": id_str, "sequence": seq, }}
            # if counts>1:
            #     multiple_chains = ",".join(id_list[chain_idx + 1 : chain_idx + counts])
            #     tmp_job["protein"]["multiple_chains"] = multiple_chains
            chain_idx += counts
            seqs2yaml.append(tmp_job)

        data = {
            "version": 1,
            "sequences": seqs2yaml
        }

        if save_path is not None:
            with open("/tmp/boltz.yaml", 'w') as outfile:
                yaml.dump(data, outfile, default_style=None, default_flow_style=False)

            #Ugly patch to avoid PyYAML putting quotes around the id strings
            os.system(f"sed \"s/'//g\" /tmp/boltz.yaml > {save_path}")
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
    def __init__(self, model_list: list[str], engine : str = "AF3"
                #  , mode:str = "heteromers"
                 ):
        self.model_list = model_list
        self.engine = engine.lower()
        if self.engine.lower() not in ["af3", "boltz"]:
            raise ValueError("Engine should be 'AF3' or 'Boltz'")
        self.df_het = None
        self.df_hom = None
        self.cluster_data = None
        # if mode not in ["heteromers", "homoromers"]:
        #     raise ValueError("Mode should be 'heteromers' or 'homoromers'")
        # self.mode = mode
        # self.process_models()
    
    @staticmethod
    def cluster_pae(pae_submatrix, threshold:int=15, eps:int=10)-> tuple:
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

    @staticmethod
    def cluster_info(low_coords, cluster_labels)-> pd.DataFrame:
        
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
            # x_min = np.min(cluster_coords[:, 1])
            # y_max = np.max(cluster_coords[:, 0])
            # x_max = np.max(cluster_coords[:, 1])
            # y_min = np.min(cluster_coords[:, 0])
            ## we are going crazy over this
            x_min = np.min(cluster_coords[:, 0])
            y_max = np.max(cluster_coords[:, 1])
            x_max = np.max(cluster_coords[:, 0])
            y_min = np.min(cluster_coords[:, 1])

            cluster_center = np.mean(cluster_coords, axis=0)
            # import pdb;pdb.set_trace()

            #Percentiles to reduce the impact of outliers
            # top_percentile = np.percentile(cluster_coords, 99.5, axis=0)
            # lower_percentile = np.percentile(cluster_coords, .5, axis=0)
            # per_x_min = lower_percentile[0]
            # per_y_min = lower_percentile[1]
            # per_x_max = top_percentile[0]
            # per_y_max = top_percentile[1]
            x_len = x_max - x_min
            y_len = y_max - y_min
            cluster_ratio = max(x_len, y_len) / min(x_len, y_len) if min(x_len, y_len) > 0 else 0
            cluster_info_list.append({
                "cluster_id": label,
                "num_points": len(cluster_coords),
                "x_len": x_len,
                "y_len": y_len,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                # "percentile_x_min": per_x_min,
                # "percentile_x_max": per_x_max,
                # "percentile_y_min": per_y_min,
                # "percentile_y_max": per_y_max,
                ## TODO: include Cluster_ratio, x_len, y_len, peptide_start, peptide_end
                "center_x": round(cluster_center[1], 2),
                "center_y": round(cluster_center[0], 2),
                "Cluster_ratio": round(cluster_ratio, 2)
            })

        # Generate the pd.df
        cluster_data_from_model = pd.DataFrame(cluster_info_list)
        
        return cluster_data_from_model
    
    @staticmethod
    def process_ppi(model_file: str, model_type: str = "AF3", mode: str = "heteromers", prefix: str = "")-> tuple[dict, pd.DataFrame]:
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
        print(f"Processing {model_file}...")
        # Parse metadata from the file path
        dir_name = os.path.dirname(model_file)
        base_name = os.path.basename(model_file).replace(".cif", "").replace(".pdb", "")
        # folder_path = os.path.dirname(dir_name)
        ppi_id = dir_name.split("/")[-1].replace(prefix, "")
        orf_a, orf_b = ppi_id.split("__")
        model_number = int(base_name.split("_")[-1].replace("model_",""))

        # Create MoleculeModel instance
        if model_type.lower() == "af3":
            full_data = process_full_data_af3(model_file)
            # molecule_model = MoleculeModel.from_af3(model_file)
        elif model_type.lower() == "boltz":
            full_data = process_full_data_boltz(model_file)
            # molecule_model = MoleculeModel.from_boltz(model_file)
        else:
            raise ValueError("model_type should be 'AF3' or 'Boltz'")

        ## Plotting
        # plot plddt
        plddt_save_name = model_file.replace(".cif", "_plddt.png").replace(".pdb", "_plddt.png")
        plot_plddt(full_data["ca_plddts"], full_data["chain_boundaries_by_res"],
                   full_data["token_chain_ids"], plddt_save_name)
        # plot pae
        pae_save_name = model_file.replace(".cif", "_pae.png").replace(".pdb", "_pae.png")
        plot_paes(full_data["pae"], full_data["chain_boundaries_by_res"], set(full_data["token_chain_ids"]),
                  f"{full_data.get('iptm', 'N/A')} ipTM - {full_data.get('ptm', 'N/A')} pTM",
                  pae_save_name)

        ## Do this only for protein pairs, not for homomers
        ## Only if I have two chains
        all_metrics = {}
        if len(set(full_data["token_chain_ids"])) == 2: ## We have two chains
            # Calculate all metrics
            all_metrics = calculate_all_metrics(model_file, full_data)
            # import pdb;pdb.set_trace()
            # all_metrics = calculate_all_metrics(molecule_model)

            chain_by_res = full_data["token_chain_ids"]
            pae = full_data["pae"]
            pae_submatrix_1 = pae[chain_by_res == "A"][:, chain_by_res == "B"]
            pae_submatrix_2 = pae[chain_by_res == "B"][:, chain_by_res == "A"].T
            submatrix = np.mean([pae_submatrix_1, pae_submatrix_2], axis=0) ## Maybe we want the mean?

            ## Clustering
            low_coords, cluster_labels = InteractomeProcessor.cluster_pae(submatrix)

            ## here we do the plot of the pae clusters
            plot_pae_clusters(submatrix,low_coords, cluster_labels, save_name=model_file.replace(".cif", "_cluster.png")) 

            cluster_data = InteractomeProcessor.cluster_info(low_coords=low_coords, cluster_labels=cluster_labels)
            # Incluir en el df de los clusters el ppi_id
            if len(cluster_data)>0:
                cluster_data.loc[:, "PPI"] = ppi_id 
                cluster_data.loc[:, "model_num"] = model_number 
                cluster_data.loc[:, "path"] = model_file 
               

        # Return summary metrics and cluster details
        return {"PPI": ppi_id, "ORF_A": orf_a, "ORF_B": orf_b, "Folder": dir_name, 
                "Model_num": model_number, 
                "ipTM": full_data["iptm"], 
                "pTM": full_data["ptm"], 
                # "chain_length_A": np.sum(chain_by_res == "A"), "chain_length_B":np.sum(chain_by_res == "B"), 
                **all_metrics
                }, cluster_data
    
    def process_models(self, output_path: str = ".", model_type: str = "AF3", prefix: str = "", **kwargs):
        ##Load output.csv and filter out processed data
        interactome_df = pd.DataFrame()
        clusters_df = pd.DataFrame()
        all_ppi_models = self.model_list.copy()

        os.makedirs(output_path, exist_ok=True)

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
            worker = partial(self.process_ppi, model_type=model_type, prefix=prefix)
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
        interactome_df.round(2)
        interactome_df.to_csv(f'{output_path}/interactome_data.csv', index=False)
        ordered_columns = ['PPI', 'model_num', 'path', 'cluster_id', 'num_points', 'x_len', 'y_len', 'x_min', 'x_max', 'y_min',
                    'y_max', 'center_x', 'center_y', 'Cluster_ratio']
        clusters_df.round(2)
        clusters_df = clusters_df.loc[:, ordered_columns]
        clusters_df.to_csv(f'{output_path}/clusters_data.csv', index=False)

        ## TODO: move this to the analyzer class
        # by_protein_df = get_info_for_proteins(interactome_df)
        # by_protein_df.to_csv(f'{output_path}/interactome_data_by_protein.csv', index=False) 

        # ## Plotting boxplots
        # output_folder = f"{output_path}/plots"
        # plddt_array, labels_array1 = process_boxplot_data(by_protein_df, "pLDDT_mean_ORF")
        # pae_array, labels_array2 = process_boxplot_data(by_protein_df, "pae_mean_ORF")
        # plot_boxplots("plddt",plddt_array, labels_array1, output_path=output_folder)
        # plot_boxplots("pae",pae_array,labels_array2, output_path=output_folder)

        # ## Plotting scatterplots
        # plot_iptm_vs_ptm(interactome_df, output_path=output_folder)

class InteractomeAnalyzer:
    def __init__(self):
        self._interactome_path = None
        self._interactome_data = None
        self._cluster_path = None
        self._cluster_data = None
        self._candidate_clusters = None
        self._models_path = None
    
    ## Getters and setters
    @property
    def interactome_path(self):
        return self._interactome_path
    
    @property
    def interactome_data(self):
        return self._interactome_data 
    
    @interactome_path.setter
    def interactome_path(self, interactome_data_path: str):
        if not os.path.exists(interactome_data_path):
            raise ValueError(f"File {interactome_data_path} not found")
        
        self._interactome_path = interactome_data_path

        self._interactome_data = pd.read_csv(interactome_data_path)
        ## Here we expect a very specific data

        ## If the lenght is zero, also return errors

    @property
    def cluster_path(self):
        return self._cluster_path 
    
    @property
    def cluster_data(self):
        return self._cluster_data 
    
    @cluster_path.setter
    def cluster_path(self, cluster_data_path: str):
        if not os.path.exists(cluster_data_path):
            raise ValueError(f"File {cluster_data_path} not found")
        
        self._cluster_path = cluster_data_path

        self._cluster_data = pd.read_csv(cluster_data_path)

        self._models_path = os.path.commonpath(self.cluster_data.path.values.tolist())
        ## Here we expect a very specific data

        ## If the lenght is zero, also return errors

        ## If we find protein ids not found in interactome data

    @property
    def models_path(self):
        return self._models_path 

    @models_path.setter
    def models_path(self, new_model_path):
        old_models_path = self._models_path
        self._cluster_data.loc[: , "path"] = self._cluster_data.path.str.replace(self._models_path, new_model_path)
        self._interactome_data.loc[: , "Folder"] = self._interactome_data.Folder.str.replace(self._models_path, new_model_path)
        self._models_path = new_model_path

        print(f"INFO: Changing model_path in cluster_data.path and interactome_data.Folder")
        print(f"INFO: from {old_models_path} to {new_model_path}")

        return self._cluster_path 

    def __str__(self):
        # Resumen bonito para el usuario
        interactome_state = self._interactome_path if self._interactome_path is not None else "Interactome path: Empty"
        interactome_len = len(self._interactome_data) if self._interactome_data is not None else 0
        cluster_state = self._cluster_path if self._cluster_path is not None else "Interactome path: Empty"
        cluster_len = len(self._cluster_data) if self._cluster_data is not None else 0
        return f"""<InteractomeAnalyzer>
        Interactome path: {interactome_state}
        Interactions: {interactome_len}
        -------------
        Interactome path: {cluster_state}
        Interactions: {cluster_len}
        """
    
    def __len__(self):
        if self._interactome_data is not None:
            return len(self._interactome_data)
        return 0
    
    def run_full_pipeline(self):
        ## Plotting the iptm vs ptm

        ## Study protein-peptides
        self._analyze_peptide_proteins_pairs()
    
    def _get_candidate_clusters(self, cluster_ratio_threshold=5, 
                                min_peptide_len = 5):
        #Generate a copy of the df with candidate clusters
        df = self._cluster_data
        df = df[(df.x_len > 0) & (df.y_len > 0)]
        candidate_clusters = df[df.Cluster_ratio > cluster_ratio_threshold].copy()
        candidate_clusters = candidate_clusters.loc[
            (candidate_clusters.x_len >= min_peptide_len) & 
            (candidate_clusters.y_len >= min_peptide_len), :]
        
        peptide_start, peptide_end = [], []
        binder_start, binder_end = [],[]
        binder_name, peptide_name = [], []
        peptide_chain = []
        binder_chain = []

        for _, row in candidate_clusters.iterrows():
            orf_a, orf_b = row.PPI.split("__")
            if row.x_len > row.y_len:
                binder_chain.append("A")
                peptide_chain.append("B")
                peptide_start.append(int(row.y_min))
                peptide_end.append(int(row.y_max))
                binder_start.append(int(row.x_min))
                binder_end.append(int(row.x_max))
                binder_name.append(orf_a)
                peptide_name.append(orf_b)
            else:
                binder_chain.append("B")
                peptide_chain.append("A")
                peptide_start.append(int(row.x_min))
                peptide_end.append(int(row.x_max))
                binder_start.append(int(row.y_min))
                binder_end.append(int(row.y_max))
                binder_name.append(orf_b)
                peptide_name.append(orf_a)

        candidate_clusters[f"Binder_chain"] = binder_chain
        candidate_clusters["Binder_name"] = binder_name
        candidate_clusters["Peptide_chain"] = peptide_chain
        candidate_clusters["Peptide_name"] = peptide_name
        candidate_clusters["Peptide_start"] = peptide_start
        candidate_clusters["Peptide_end"] = peptide_end
        candidate_clusters[f"Binder_start"] = binder_start
        candidate_clusters[f"Binder_end"] = binder_end

        ## Filter by Binder quality
        ## for each binder get a df with their metrics 
        ## like mean_PAE, 

        return candidate_clusters.reset_index()

    def _curate_protein_peptide_models(self, ppi_data: pd.DataFrame):
        mol_list = [Molecule(i) for i in ppi_data.path]
        plddt = np.array([np.mean(mol.beta) for mol in mol_list])
        best_idx = np.argmax(plddt)

        ## Standarize molecule chains
        for idx, mol in enumerate(mol_list):
            if ppi_data.Peptide_chain.values[0] == "A":
                # break ##Already with the intended naming

                mol.set("chain", "C", "chain A")
                mol.set("chain", "A", "chain B")  ## chain A is binder
                mol.set("chain", "B", "chain C") ## chain B is peptide

        ## Get the "best" folded partner    
        reference_mol = mol_list[best_idx].copy()
        reference_mol.filter("chain A")
        reference_resids = reference_mol.resid[reference_mol.name == "CA"][reference_mol.beta[reference_mol.name == "CA"]>70]
        reference_resids = reference_resids.astype(str)
        reference_resid_str = ' '.join(reference_resids)

        reference_mol.filter(f"resid {reference_resid_str}")

        for idx, mol in enumerate(mol_list):
            ## Filter to get only the peptide of chain B
            mol.filter(f"(chain A and resid {reference_resid_str}) or (chain B and resid {ppi_data['Peptide_start'].values[idx] + 1} to {ppi_data['Peptide_end'].values[idx] + 1})")
            # mol.filter(f"(chain A) or (chain B and resid {ppi_data['Peptide_start'].values[idx] + 1} to {ppi_data['Peptide_end'].values[idx] + 1})")
            
            ## We align to the reference structure
            ## NOPE; we doe alignment later
            # mol.align(f"chain A and name CA",
            #           refmol=reference_mol,
            #           refsel=f"chain A and name CA")

        ## Saving the filtered molecules
        output_folder = f"{os.path.dirname(ppi_data.path.values[0])}/prot_peptide/"
        print(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        reference_mol.write(f"{output_folder}/reference.pdb")

        for idx, mol in enumerate(mol_list):
            mol.write(f"{output_folder}/{ppi_data.PPI.values[0]}_{ppi_data.model_num.values[idx]}.pdb")

    def _create_binder_alignments(self, binder_name, ppi_data):
        ## Get all models where is binder is the same protein
        all_structs = []
        for ppi in ppi_data.PPI.unique():
            tmp_structs = glob(f"{self.models_path}/*{ppi}/prot_peptide/*pdb")
            all_structs.extend(tmp_structs)
        ## We use the first one as default
        reference_mol_path = [p for p in all_structs if "reference" in p][0]
        # reference_mol_path = filter(lambda x: "reference" in x, all_structs)
        reference_mol = Molecule(reference_mol_path)
        
        ## Align all structure to that template
        for tmp_mol_path in all_structs:
            tmp_mol = Molecule(tmp_mol_path)
            tmp_mol.align(f"chain A",
                      refmol=reference_mol,
                      refsel=f"chain A",
                      mode="structure")
            tmp_mol.write(tmp_mol.topoloc.replace(".pdb", "_toref.pdb"))
    
    def _analyze_peptide_proteins_pairs(self):
        ## Find candidates clusters
        self._candidate_clusters = self._get_candidate_clusters()

        df = self._candidate_clusters.loc[:, ["PPI", "model_num", "x_len", "y_len", 
                                              "Binder_name", "Binder_chain", "Binder_start", "Binder_end",
                                              "Peptide_name", "Peptide_chain", 
                                             "Peptide_start", "Peptide_end",  "path"]]
        
        ## This will be removed... but we will have to handle it somehow
        # df.loc[: , "path"] = df.path.str.replace("/media/DATA/ppi_data/", "/home/daniel/ppi_data_remote/")
        df.loc[: , "PPI"] = df.PPI + "_" +  self._candidate_clusters.cluster_id.astype(str)
        ppis = df.PPI.unique()
        
        ## I want to iterate over ppi_clusters
        for ppi in ppis:
            clean_ppi = ppi[1:] ## Patch, we have a trailing _ at the beggining of the name
            ppi_data = df.loc[df.PPI == ppi,:]
            # self._curate_protein_peptide_models(ppi_data) ## Temporary comment

        ## Iterate over binder
        for binder in self._candidate_clusters.Binder_name.unique():
            ## Clustering coordinates
            ppi_data = self._candidate_clusters.loc[self._candidate_clusters.Binder_name == binder,:]
            ## 
            # self._create_binder_alignments(binder, ppi_data)


#     def calculate_network(self):
#         pass

#     def basic_plots(self):
#         pass

#     def protein_peptide_analysis(self):
#         pass

#     def cluster_analysis(self):
#         pass
