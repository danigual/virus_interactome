import os
import tqdm
import pandas as pd
import numpy as np
import yaml
import json
import concurrent.futures
import warnings
import csv
import logging


from glob import glob
from sklearn.cluster import DBSCAN
from functools import partial
from itertools import combinations, product
from typing import Dict, Iterable, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path
from moleculekit.molecule import Molecule

from .utils import load_json, load_boltz_input, check_sequence_validity, process_full_data_af3, process_full_data_boltz
from .proteome_manager import ProteomeManager
from .metrics import calculate_all_metrics
from .plotting import plot_boxplots, plot_iptm_vs_ptm, plot_pae_clusters, plot_paes, plot_plddt

## Configure logger just in case (best practice)
logger = logging.getLogger(__name__)

class InteractomeWriter:
    """
    Manages the configuration and writing of interactome data.

    This class initializes the environment to analyze interactions within a single
    proteome (intra) or between two different proteomes (inter). It normalizes the
    input by converting file paths (str) into ProteomeManager instances.

    Attributes:
        proteome_a (ProteomeManager): Instance of the first managed proteome.
        proteome_b (Optional[ProteomeManager]): Instance of the second proteome (if any).
        mode (str): The operation mode detected; can be "intra" or "inter".
    """
    def __init__(self, proteome_a: str | ProteomeManager, proteome_b: str | ProteomeManager | None = None):

        """
        Initializes a new instance of InteractomeWriter.

        Args:
            proteome_a (str | ProteomeManager): Path to the first proteome file or 
                an existing ProteomeManager instance. This is mandatory.
            proteome_b (str | ProteomeManager | None, optional): Path or instance of the 
                second proteome. If omitted or None, the mode is automatically set 
                to "intra". Defaults to None.

        Raises:
            ValueError: If `proteome_a` is neither a string path nor a ProteomeManager 
                instance.
        """

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
        
    
    def generate_intra_pairs(self) -> Iterable[Tuple[str, str]]:
        """
        Generates unique, unordered heteromeric pairs within Proteome A.

        This method creates combinations of sequences (A, B) where A and B belong 
        to Proteome A and A != B. This is used for calculating the intra-interactome.
        
        Note:
            - Homomers (A, A) are excluded. Use `generate_homo_pairs()` for those.
            - The order is canonical (based on the list order) to prevent duplicates 
              (i.e., (A, B) is generated, but (B, A) is not).

        Returns:
            Iterable[Tuple[str, str]]: An iterator of tuples, where each tuple 
            contains two distinct sequence IDs from Proteome A.

        Raises:
            ValueError: If the instance is not in valid usage (e.g., Proteome A is missing).
        """

        # if self.mode != "inter" or self.proteome_b is None:
        #     raise ValueError("generate_inter_pairs() requires 'inter' mode with a valid proteome_b.")
        if self.proteome_a is None:
             raise ValueError("generate_intra_pairs() requires a valid proteome_a.")

        ids = list(self.proteome_a.sequences.keys())

        return combinations(ids, 2)

    def generate_inter_pairs(self)-> Iterable[Tuple[str, str]]:
        """
        Generates the Cartesian product of sequences between Proteome A and Proteome B.

        This method produces all possible pairs (a, b) where 'a' belongs to Proteome A 
        and 'b' belongs to Proteome B. This is the standard approach for analyzing 
        inter-species or inter-system interactions.

        Returns:
            Iterable[Tuple[str, str]]: An iterator yielding tuples (id_a, id_b), 
            representing the interaction candidates.

        Raises:
            ValueError: If the instance is not in 'inter' mode or if `proteome_b` 
            has not been initialized.
        """

        # Ensure we are in the correct mode to prevent generating invalid data
        if self.mode != "inter" or self.proteome_b is None:
            raise ValueError("generate_inter_pairs() requires 'inter' mode with a valid proteome_b.")
        
       
        ids_a = list(self.proteome_a.sequences.keys())
        ids_b = list(self.proteome_b.sequences.keys())

        return product(ids_a, ids_b)

    def generate_homo_mers(self, nmin: int = 2, nmax: int = 6)-> Iterable[Tuple[str, int]]:
        """
        Generates homomeric configurations (oligomers) for sequences in Proteome A.

        This method iterates through all sequences in the proteome and defines
        oligomeric states ranging from 'nmin' to 'nmax' copies.
        
        Note:
            This is strictly for the 'intra' mode. Homomers generally do not make 
            sense in an 'inter' context (interactions between two different datasets) 
            within this pipeline.

        Args:
            nmin (int, optional): Minimum number of copies (oligomer size). 
                Must be >= 2. Defaults to 2.
            nmax (int, optional): Maximum number of copies. 
                Must be >= nmin. Defaults to 6.

        Yields:
            Iterable[Tuple[str, int]]: An iterator yielding tuples in the format 
            (protein_id, num_copies).

        Raises:
            ValueError: If called in 'inter' mode.
            ValueError: If 'nmin' < 2 (single copies should use `generate_single_run`).
            ValueError: If 'nmax' is less than `nmin` (invalid range).
        """

        if self.mode == "inter":
            raise ValueError("Homo mers can only be computed in 'intra' mode.")
        if nmin < 2:
            raise ValueError("nmin can not be lower than 2. If you want just 1 copy use the generate_single_run method")
        
        if nmax < nmin:
            raise ValueError(f"nmax ({nmax}) must be greater than or equal to nmin ({nmin}).")
        ids_a = list(self.proteome_a.sequences.keys())

        for tmp_protein_id in ids_a:
            for num_copies in range(nmin, nmax+1):
                yield(tmp_protein_id, num_copies)
     
    def generate_single_run(
        self,
        source: str = "a",
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None
    ) -> Iterable[Tuple[str, str, int]]:
        
        """
        Generates jobs for single protein folding (monomers).

        This method yields sequences for individual proteins (stoichiometry = 1).
        It allows selecting specific subsets of IDs via 'ids_a' or 'ids_b'.

        Args:
            source (str, optional): The source proteome(s) to process.
                - 'a': Only Proteome A.
                - 'b': Only Proteome B (requires 'inter' mode).
                - 'both': Proteome A followed by Proteome B.
                Defaults to "a".
            ids_a (Optional[List[str]], optional): A specific list of IDs from Proteome A 
                to process. If None, processes all sequences in A. Defaults to None.
            ids_b (Optional[List[str]], optional): A specific list of IDs from Proteome B 
                to process. If None, processes all sequences in B. Defaults to None.

        Yields:
            Iterable[Tuple[str, str, int]]: An iterator yielding tuples in the format 
            (protein_id, sequence_string, copy_count=1).

        Raises:
            ValueError: If 'source' is not one of 'a', 'b', or 'both'.
            ValueError: If 'source' includes 'b' but the instance is not in 'inter' mode.
        """

        valid_sources = {"a", "b", "both"}
        if source not in valid_sources:
            raise ValueError(f"source must be one of {valid_sources}")

        if source in {"b", "both"}:
            if self.mode != "inter" or self.proteome_b is None:
                raise ValueError("source='b' or 'both' requires 'inter' mode with a proteome_b.")
        
        if source in {"a", "both"}:
            target_ids = ids_a if ids_a is not None else self.proteome_a.ids

            for pid in target_ids:
                yield (pid, self.proteome_a.sequences[pid], 1)

        if source in {"b", "both"}:
            target_ids = ids_b if ids_b is not None else self.proteome_b.ids

            for pid in (ids_b or self.proteome_b.ids):
                yield (pid, self.proteome_b.sequences[pid], 1)

    def write_interactome_jobs(
        self,
        engine: str,
        output_dir: str,
        *,
        mode: str = "intra_pairs", 
        include_homo: bool = False,
        nmin: int = 2, 
        nmax: int = 6,
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
        Orchestrates the creation of input files (jobs) for protein folding engines.

        This method generates the necessary combinations (pairs, homomers, or monomers),
        validates sequence lengths against engine limits, writes the specific JSON/YAML 
        input files, and logs the metadata to a CSV index.

        Args:
            engine (str): The target folding engine. Options: 'af3' (AlphaFold 3) or 'boltz2'.
            output_dir (str): Directory where input files and the index CSV will be saved.
            mode (str, optional): Generation strategy. 
                Options: 'intra_pairs', 'inter_pairs', 'homomers', 'single'. 
                Defaults to "intra_pairs".
            include_homo (bool, optional): [Reserved] If True, may include homomers in pair runs. 
                Currently unused. Defaults to False.
            nmin (int, optional): Minimum stoichiometry for homomers. Defaults to 2.
            nmax (int, optional): Maximum stoichiometry for homomers. Defaults to 6.
            counts_map (Optional[Dict[str, int]], optional): Custom stoichiometry override per ID. 
                Defaults to None (1 copy).
            ids_a (Optional[List[str]], optional): Filter specific IDs for Proteome A.
            ids_b (Optional[List[str]], optional): Filter specific IDs for Proteome B.
            af3_threshold (int, optional): Max total residues allowed for AF3. Defaults to 5000.
            af3_batch_size (int, optional): [Reserved] Batch size for JSON grouping. Defaults to 30.
            boltz_threshold (int, optional): Max total residues allowed for Boltz. Defaults to 1600.
            skip_over_threshold (bool, optional): If True, files exceeding the residue 
                threshold are not written to disk. Defaults to False.
            filename_fmt (str, optional): F-string format for output filenames. 
                Defaults to "{engine}_{name}.{ext}".
            index_name (str, optional): Name of the metadata CSV file. Defaults to "index.csv".

        Returns:
            List[dict]: A list of metadata dictionaries representing every job processed 
            (including skipped ones).

        Raises:
            ValueError: If 'engine' is not 'af3' or 'boltz2'.
            ValueError: If 'mode' is unknown.
            RuntimeError: If the iterator yields an unexpected data structure.
        """

        # 1. Setup Environment
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        engine_lower = engine.lower()
        if engine_lower == "af3":
            ext = "json"
            residue_threshold = af3_threshold
        elif engine_lower == "boltz2":
            ext = "yaml"
            residue_threshold = boltz_threshold
        else:
            raise ValueError(f"Unsupported engine '{engine}'. Must be 'af3' or 'boltz2'.")

        # 2. Select Generator based on mode
        # Note: We pass ids_a/ids_b filters to the generators

        if mode == "intra_pairs":
            pairs = self.generate_intra_pairs() 
            if ids_a:
                pairs = (p for p in pairs if p[0] in ids_a and p[1] in ids_a)
            iterator = (("pair", p) for p in pairs)
        
        elif mode == "inter_pairs":
            pairs = self.generate_inter_pairs() 
            iterator = (("pair", p) for p in pairs)
        
        elif mode == "homomers":
            homes = self.generate_homo_mers(nmin=nmin, nmax=nmax)
            iterator = (("homo", h) for h in homes)
        
        elif mode == "single":
            singles = self.generate_single_run(source="both" if self.mode == "inter" else "a", ids_a=ids_a, ids_b=ids_b)  
            iterator = (("single", s) for s in singles)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        metas: List[dict] = []
        index_path = out_path/ index_name

        # 3. Process Jobs and Write CSV
        with open(index_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "engine", "mode", "name", "idA", "idB", 
                "countA", "countB", "total_residues", "warnings", "file_path"
            ]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()

            for kind, entry in iterator:

                # Build seq_list and job_name
                seq_list = []
                name = ""
                idA, idB = "", ""
                
                # --- Logic for Pairs ---
                if kind == "pair":
                    idA_raw, idB_raw = entry
                    # idA, idB = entry

                    # Determine source proteomes
                    prot_A = self.proteome_a
                    # For intra pairs, both are from A. For inter pairs, B is from B.
                    prot_B = self.proteome_b if mode == "inter_pairs" else self.proteome_a

                    # Sort IDs for canonical naming if in intra mode (to avoid A_B vs B_A dupes)
                    if mode == "intra_pairs" and idA_raw > idB_raw:
                        idA, idB = idB_raw, idA_raw
                        prot_A, prot_B = prot_B, prot_A # Swap proteomes (same obj in intra)
                    else:
                        idA, idB = idA_raw, idB_raw

                    # Fetch sequences
                    seqA = prot_A.sequences[idA]
                    seqB = prot_B.sequences[idB]
                    
                    # Determine stoichiometry
                    cntA = counts_map.get(idA, 1) if counts_map else 1
                    cntB = counts_map.get(idB, 1) if counts_map else 1
                    
                    seq_list = [(idA, seqA, cntA), (idB, seqB, cntB)]
                    name = f"{idA}__{idB}"

                # --- Logic for Homomers ---
                elif kind == "homo":
                    pid, copies = entry
                    seq_list = [(pid, self.proteome_a.sequences[pid], copies)]
                    name = f"{pid}__{copies}"
                    idA = pid
                    # idA, idB = pid, "" 

                # --- Logic for Monomers ---   
                elif kind == "single":
                    pid, seq, cnt = entry
                    seq_list = [(pid, seq, cnt)]
                    name = f"{pid}"
                    idA = pid
                    # idA, idB = pid, ""
                else:
                    raise RuntimeError(f"Unexpected iterator kind: {kind}")

                
                # 4. Calculate Size and Validate
                total_res = sum(len(s) * c for _, s, c in seq_list)
                # total_res = sum(len("".join(s.split())) * c for _, s, c in seq_list)
                is_over_limit = total_res > residue_threshold
                
                base_name = filename_fmt.format(engine=engine_lower, name=name, ext=ext)
                save_path_str = ""
                warns = []

                # Write file if within limits (or if we are not skipping)
                if not (skip_over_threshold and is_over_limit):
                    full_save_path = out_path / base_name
                    save_path_str = str(full_save_path)
                # save_path = str(Path(output_dir) / base_name) if not (skip_over_threshold and over) else ""

                # if save_path:
                    if engine_lower == "af3":
                        self.get_af3_input(seq_list, job_name=name,
                        save_path=save_path_str, 
                        residue_threshold=residue_threshold)
                    else:
                        self.get_boltz2_input(seq_list, save_path=save_path_str,
                        residue_threshold=residue_threshold)
                    
                if is_over_limit:
                    warns.append(f"Skipped: total residues {total_res} exceed {residue_threshold}")
                
                # 5. Record Metadata
                meta = {
                    "engine": engine_lower,
                    "mode": mode,
                    "name": name,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                    "warnings": "|".join(warns),
                    "file_path": save_path_str,
                }
                metas.append(meta)
                w.writerow(meta)

        return metas
    
    @staticmethod
    def check_input(seq_list: List[Tuple[str,str,int]], residue_threshold: int = 5000)-> Tuple[bool, Optional[str]]:
        """
        Validates a list of sequences and checks for residue limits.
        
        This utility method ensures that the sequence list is not empty, 
        counts are positive, and sequences are valid (using the global 
        'check_sequence_validity' helper). It also issues a warning if 
        the total number of residues exceeds the recommended threshold.

        Args:
            seq_list (List[Tuple[str, str, int]]): A list of tuples containing 
                (chain_id, sequence, count).
            residue_threshold (int, optional): The maximum recommended number 
                of residues before issuing a warning. Defaults to 5000.

        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None) if the input is valid (even if it exceeds the threshold, 
                  only a warning is issued).
                - (False, error_message) if a blocking error is found (empty list, 
                  invalid sequence, etc.).
        """

        total_res = 0
        # if len(seq_list) == 0:
        if not seq_list:
            return False, "Sequence list cannot be empty."
        
        for chain_id, seq, count in seq_list:
            if count < 1:
                return False, f"Count for {chain_id} needs to be at least 1."
            
            seq_clean = check_sequence_validity(seq)

            # _validate_seq(seq_clean, strict=strict)
            if not seq_clean:
                return False, f"{chain_id} is not a valid protein sequnce."
            # total_res += len(seq) * count
            total_res += len(seq_clean) * count

        if total_res > residue_threshold:
            msg = (
                f"Total residues {total_res} exceed recommended maximum ({residue_threshold})."
            )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
        
        return True, None

    @staticmethod
    def get_af3_input(
        seq_list: List[Tuple[str, str, int]],
        job_name: str = "AF3_job",
        residue_threshold: int = 5000,
        save_path: Optional[str] = None)-> Dict[str, Any]:

        """
        Generates the JSON input structure required for an AlphaFold 3 job.

        This method validates the sequence list, constructs the specific dictionary 
        format expected by the AF3 inference pipeline, and optionally writes it to 
        a JSON file.

        Args:
            seq_list (List[Tuple[str, str, int]]): A list of tuples defining the complex.
                Format: '[(chain_id, sequence_string, copy_count), ...]'.
                Example: '[("A", "MVE...", 1), ("B", "SEQ...", 2)]'.
            job_name (str, optional): The name of the job to be recorded in the JSON. 
                Defaults to "AF3_job".
            residue_threshold (int, optional): The maximum residue count before issuing 
                a warning. Defaults to 5000.
            save_path (Optional[str], optional): The file path where the JSON output 
                should be saved. If None, the file is not created. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the AF3 job payload. 
            Structure:
            {
                "name": str,
                "sequences": [
                    {
                        "proteinChain": {
                            "id": str,
                            "count": int,
                            "sequence": str
                        }
                    }, ...
                ],
                "modelSeeds": []
            }

        Raises:
            ValueError: If the input sequences fail the validation check (e.g., empty list, 
            invalid characters, or zero counts).
        """
        # Validate the input using the static method defined in the class
        is_valid, err_msg = InteractomeWriter.check_input(seq_list, residue_threshold=residue_threshold)
        
        if not is_valid:
            raise ValueError(f"Invalid input for AF3: {err_msg}")
        
        sequences = []

        for chain_id, seq, count in seq_list:
            sequences.append(
                {"proteinChain": {"id": chain_id, "count": count, "sequence": seq}}
            )

        data = {
            "name": job_name,
            "sequences": sequences,
            "modelSeeds": []
            }
        
        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as outfile:
                json.dump(data, outfile, indent=4)

        return data

    @staticmethod
    def get_boltz2_input(
        seq_list: List[Tuple[str, str, int]], 
        residue_threshold: int = 1600,
        save_path: Optional[str] = None
        )-> Dict[str, Any]:
        """
        Generates the YAML input structure required for Boltz 2.

        This method assigns unique chain IDs (A, B, C...) to the provided sequences,
        constructs the payload compliant with the Boltz YAML schema, and handles 
        file writing with proper formatting.

        Args:
            seq_list (List[Tuple[str, str, int]]): A list of tuples defining the complex.
                Format: '[(original_id, sequence, count), ...]'.
            residue_threshold (int, optional): Threshold for residue count warnings. 
                Defaults to 1600.
            save_path (Optional[str], optional): Destination path for the YAML file. 
                Defaults to None.

        Returns:
            Dict[str, Any]: The dictionary representation of the job payload.

        Raises:
            ValueError: If validation fails.
        """
        
        # Validate input
        is_valid, err_msg = InteractomeWriter.check_input(seq_list, residue_threshold=residue_threshold)
      
        if not is_valid:
            raise ValueError(f"Invalid input for Boltz: {err_msg}")
        

        # Generator for Chain IDs: A, B, ... Z, AA, AB, ...
        
        def chain_id_generator():
            from string import ascii_uppercase
            from itertools import product
            # Yield single letters first
            for char in ascii_uppercase:
                yield char
            # Yield double letters (AA, AB...) if needed
            for r in range(2, 4):
                for combo in product(ascii_uppercase, repeat=r):
                    yield "".join(combo)
      
        chain_gen = chain_id_generator()
        seqs2yaml = []


        for original_id, seq, counts in seq_list:

            # Generate the specific IDs for this entry (e.g., if count=2 -> ['A', 'B'])
            current_ids = [next(chain_gen) for _ in range(counts)]

            tmp_job = {
                "protein": {
                    "id": current_ids,
                    "sequence": seq, 
                }
            }
            seqs2yaml.append(tmp_job)

        data = {
            "version": 1,
            "sequences": seqs2yaml
        }

        if save_path:
            yaml_str = yaml.dump(data, default_flow_style=None, sort_keys=False)

            with open(save_path, 'w', encoding='utf-8') as outfile:
                outfile.write(yaml_str)

        return data

class InteractomeRunner:

    """
    Manages the execution lifecycle and monitoring of protein folding jobs.

    This class identifies input files for specific folding engines (AF3 or Boltz2),
    tracks existing outputs, and determines the execution status of the interactome.

    Attributes:
        mode (str): The active engine mode ('af3' or 'boltz2').
        input_dir (Path): The directory path containing input files.
        output_dir (Path): The directory path containing output results.
        inputs (List[Path]): A list of detected input files (.json or .yaml).
        outputs (List[Path]): A list of detected output directories.
        parse_job_dictionary (Dict[str, Callable]): A registry mapping modes to 
            their specific job parsing methods.
        status (Any): The current status of the run (determined by check_run).
    """


    def __init__(self, path_of_inputs: str, path_of_outputs: str, mode: str ="boltz2"):

        """
        Initializes the InteractomeRunner.

        Args:
            path_of_inputs (str): Path to the directory containing input files.
            path_of_outputs (str): Path to the directory where results are stored.
            mode (str, optional): The folding engine to use. 
                Must be 'af3' or 'boltz2'. Defaults to "boltz2".

        Raises:
            ValueError: If the provided mode is not supported.
            FileNotFoundError: If the input directory does not exist.
        """


        self._available_modes = ["af3", "boltz2"]

        if mode not in self._available_modes:
            raise ValueError(f"Mode '{mode}' is invalid. Supported modes: {', '.join(self._available_modes)}")
        
        self.mode = mode

        # Convert strings to Path objects for robust cross-platform handling
        self.input_dir = Path(path_of_inputs)
        self.output_dir = Path(path_of_outputs)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect Inputs based on Mode usage of pathlib.glob is safer than string concatenation glob()
    
        if self.mode == "af3":
            self.inputs = list(self.input_dir.glob("*.json"))
            
        elif self.mode == "boltz2":
            self.inputs = list(self.input_dir.glob("*.yaml"))
            
        else:
            self.inputs = []

        # Detect Outputs. Assuming outputs are directories inside the output path
        self.outputs = [p for p in self.output_dir.glob("*") if p.is_dir()]

        self.parse_job_dictionary: Dict[str, Callable] = {
            "af3": self._parse_af3_job,
            "boltz2": self._parse_boltz2_job,
        }
        
        # Determine initial status
        self.status = self.check_run()
        logger.info(f"Initialized Runner in '{mode}' mode. Found {len(self.inputs)} inputs.")

    def _parse_af3_job(self, input_json: Path)-> Dict[str, Any]:
        """
        Internal handler to parse AF3 input files with logging.
        Delegates to the robust load_json utility.
        """
        logger.debug(f"Parsing AF3 job: {input_json.name}")
        return load_json(str(input_json))


    def _parse_boltz2_job(self, input_yaml: Path)-> Dict[str, Any]:
        """
        Internal handler to parse Boltz2 input files with logging.
        Delegates to the hybrid adapter load_boltz_input utility.
        """

        logger.debug(f"Parsing Boltz job: {input_yaml.name}")
        return load_boltz_input(str(input_yaml))

    
    def check_run(self, expected_models : Optional[int] = None)-> pd.DataFrame:
        """
        Scans input and output directories to audit the execution status.

        This method determines the execution status based on the number of generated models.
        It automatically adjusts expectations based on the engine (AF3 vs Boltz) if 
        no specific threshold is provided.

        Args:
            expected_models (Optional[int], optional): The number of models required 
                to mark a job as COMPLETED. 
                If None, defaults to 10 for 'af3' and 5 for 'boltz2'. 
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame sorted by status and complexity.
        """
        # Dynamic Defaults Strategy
        if expected_models is None:
            if self.mode == "af3":
                expected_models = 10  # User specified standard for AF3
            elif self.mode == "boltz2":
                expected_models = 5   # Standard for Boltz
            else:
                expected_models = 5   # Fallback safe default
        
        logger.debug(f"Checking run status with threshold: {expected_models} models per job.")
        
        parse_job = self.parse_job_dictionary[self.mode]
        all_job_info = []

        # Parse Inputs to build the "Expected" list
        for model_input in self.inputs:

            # model_input is a Path object (from __init__)
            batch_jobs = parse_job(model_input)

            # AF3 might return multiple jobs in one JSON, Boltz usually one.

            ## For each job we get: job_id, #proteins, #aa
            for tmp_job in batch_jobs:
                
                job_name = tmp_job.get("name", "Unknown")
                sequences = tmp_job.get("sequences", [])
                
                n_chains = sum(
                        seq.get("proteinChain", {}).get("count", 1) 
                        for seq in sequences
                    )
                # Total Residues = Sequence Length * Copy Count
                # (Previous code ignored stoichiometry in residue calc)
                n_aa = sum(
                        len(seq.get("proteinChain", {}).get("sequence", "")) * seq.get("proteinChain", {}).get("count", 1)
                        for seq in sequences
                    )
                    
                all_job_info.append([job_name, n_chains, n_aa])


                # all_job_info.append(
                #     [tmp_job.get("name"), ## Job name
                #      sum([i.get("proteinChain").get("count") for i in tmp_job.get("sequences")]), ## Number of chains
                #      sum([len(i.get("proteinChain").get("sequence")) for i in tmp_job.get("sequences")]), ## Total number of residues
                #      ]
                # )

        df = pd.DataFrame(all_job_info, columns = ["PPI", "num_chain", "num_aa"])

        # Check Outputs 
        num_models_list = []
        # for tmp_job_name, _, _ in all_job_info:
        for job_name in df["PPI"]:
            # Pathlib approach: Safer than glob strings like f"{path}/{name}/*"
            job_output_dir = self.output_dir / job_name
            # tmp_num_models = len(glob(f"{self.path_of_outputs}/{tmp_job_name}/*model*cif"))
            if job_output_dir.exists():
                count = len(list(job_output_dir.glob("*model*cif")))
            else:
                count = 0
            num_models_list.append(count)
        
        df["num_models"] = num_models_list

        # Determine Status (Threshold Logic)
        # Default to PENDING
        df["status"] = "PENDING"

        #Replaced .mode() with explicit thresholds
        df.loc[df["num_models"] >= expected_models, "status"] = "COMPLETED"
        # RUNNING: > 0 but < expected
        mask_running = (df["num_models"] > 0) & (df["num_models"] < expected_models)
        df.loc[mask_running, "status"] = "RUNNING"
        # FAILED: 0 models (Not started or crashed)
        df.loc[df["num_models"] == 0, "status"] = "FAILED"
        # mode_num_models = int(df.num_models.mode().values)
        # df.loc[df.num_models == mode_num_models, "status"] = "COMPLETED"
        # df.loc[df.num_models != mode_num_models, "status"] = "PENDING"
        # df.loc[df.num_models == 0, "status"] = "FAILED"

        # Sort
        custom_order = ['FAILED', 'RUNNING', 'PENDING', 'COMPLETED']
        df['status'] = pd.Categorical(df['status'], categories=custom_order, ordered=True)
        df.sort_values(by=['status', 'num_aa'], ascending=[True, False], inplace=True)
        
        return df
    

    def write_status(self, file_name: Optional[Union[str, Path]] = None, update: bool = False) -> None:
        """
        Exports the current job status DataFrame to a CSV file.

        Args:
            file_name (Optional[Union[str, Path]], optional): The destination path for the CSV. 
                If None, saves as 'JOB_STATUS.csv' in the input directory. 
                Defaults to None.
            update (bool, optional): If True, re-runs the status check to ensure 
                the data is up-to-date before saving. Defaults to False.
        """
        # Option to refresh the status before writing
        if update:
            logger.info("Updating status before saving...")
            self.status = self.check_run()

        # Determine the file path
        if file_name is None:
            final_path = self.input_dir / "JOB_STATUS.csv"
        else:
            final_path = Path(file_name)
        logger.info(f"Saving execution status to: {final_path}")
        
        # Write to CSV
        self.status.to_csv(final_path, index=False)
    
    def write_missing_jobs(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Identifies non-completed jobs and copies their input files to a separate directory.

        This method is useful for isolating failed or pending jobs to re-run them 
        separately without re-processing the entire dataset. It supports both 
        AF3 (.json) and Boltz (.yaml) modes dynamically.

        Args:
            output_path (Optional[Union[str, Path]], optional): The destination directory 
                for the missing inputs. If None, creates a directory named 'input_missing' 
                sibling to the original input folder. Defaults to None.
        
        Raises:
            OSError: If there are permission issues creating directories or copying files.
        """
        import shutil

        # Identify missing jobs 
        if self.status is None or self.status.empty:
            logger.warning("Status dataframe is empty. Run check_run() first.")
            return

        missing_jobs = self.status.loc[self.status["status"] != "COMPLETED", "PPI"].values

        if len(missing_jobs) == 0:
            logger.info("No pending or failed jobs found. Nothing to copy.")
            return

        # Determine Paths 
        if output_path is None:
            output_path = self.input_dir.parent / "input_missing"
        else:
            output_path = Path(output_path)

        # Create directory 
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine Extension based on mode. In order to write missing jobs
        # from AF3 or from Boltz.
        file_ext = ".json" if self.mode == "af3" else ".yaml"

        logger.info(f"Backing up {len(missing_jobs)} missing jobs to: {output_path}")

        # Copy Loop
        copied_count = 0
        for ppi_id in missing_jobs:
            source_file = self.input_dir / f"{ppi_id}{file_ext}"
            dest_file = output_path / f"{ppi_id}{file_ext}"

            if source_file.exists():
                # copy2 preserves metadata (timestamps), which is better for traceability
                shutil.copy2(source_file, dest_file)
                copied_count += 1
            else:
                logger.error(f"Source file not found for job {ppi_id}: {source_file}")

        logger.info(f"Successfully copied {copied_count}/{len(missing_jobs)} input files.")

class InteractomeProcessor:
    """
    Analyzes the output models generated by folding engines (AF3 or Boltz).

    This class is responsible for parsing model files (CIF/JSON), extracting 
    quality metrics (pLDDT, PAE, ipTM...), and organizing the data into 
    DataFrames for heteromers and homomers.

    Attributes:
        model_paths (List[Path]): List of paths to the model directories or files to analyze.
        engine (str): The engine used to generate the models ('af3' or 'boltz').
        df_het (Optional[pd.DataFrame]): DataFrame containing metrics for heteromeric complexes.
        df_hom (Optional[pd.DataFrame]): DataFrame containing metrics for homomeric complexes.
        cluster_data (Optional[pd.DataFrame]): DataFrame containing clustering results (if applicable).
    """
    _SUPPORTED_ENGINES = {"af3", "boltz"}

    def __init__(self, model_list: List[Union[str, Path]], engine: str = "af3"):
        """
        Initializes the InteractomeProcessor.

        Args:
            model_list (List[Union[str, Path]]): A list of paths pointing to the 
                results to be processed. Can be strings or Path objects.
            engine (str, optional): The folding engine used. Case-insensitive. 
                Must be 'af3' or 'boltz'. Defaults to "af3".

        Raises:
            ValueError: If the provided engine is not supported.
        """
        # Normalize and Validate Engine
        self.engine = engine.lower()
        
        if self.engine not in self._SUPPORTED_ENGINES:
            valid_modes = ", ".join(self._SUPPORTED_ENGINES)
            logger.error(f"Invalid engine provided: '{engine}'. Expected: {valid_modes}")
            raise ValueError(f"Engine should be one of: {valid_modes}")

        # Sanitize Paths (Convert all to Path objects)
        self.model_paths = [Path(p) for p in model_list]
        
        if not self.model_paths:
            logger.warning("InteractomeProcessor initialized with an empty model list.")
        else:
            logger.info(f"Initialized Processor for {len(self.model_paths)} models using engine '{self.engine}'")

        # Initialize Data Containers. We explicitly type hint these as DataFrames or None
        self.df_het: Optional[pd.DataFrame] = None
        self.df_hom: Optional[pd.DataFrame] = None
        self.cluster_data: Optional[pd.DataFrame] = None
    
    @staticmethod
    def cluster_pae(pae_submatrix: np.ndarrray, threshold:float = 15.0, eps:float = 10.0, min_samples: int = 5)-> Tuple[np.ndarray, np.ndarray]:
        """
        Clusters low PAE regions in a PAE submatrix using DBSCAN algorithm.

        This utility method identifies coordinates where the Predicted Aligned Error (PAE)
        is below a specified threshold and groups them spatially using density-based clustering.

        Args:
            pae_submatrix (np.ndarray): A 2D array representing the PAE matrix subset.
            threshold (float, optional): The maximum PAE value to be considered a 'contact'. 
                Defaults to 15.0.
            eps (float, optional): The maximum distance between two samples for one to be 
                considered as in the neighborhood of the other (DBSCAN param). Defaults to 10.0.
            min_samples (int, optional): The number of samples in a neighborhood for a point 
                to be considered as a core point. Defaults to 5.

       Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - coords (np.ndarray): Array of shape (N, 2) with [row, col] coordinates.
                - labels (np.ndarray): Array of shape (N,) with cluster labels (-1 for noise).
                Returns empty arrays if no points satisfy the threshold.

        """
        # Find coordinates below threshold
        # np.argwhere returns indices of shape (N, 2) directly. Cleaner than column_stack/where.
        low_pae_coords = np.argwhere(pae_submatrix < threshold)
        # low_pae_coords = np.column_stack(np.where(pae_submatrix < threshold))
        
        # Early Exit (Fail Fast)
        if low_pae_coords.size == 0:
            # Return empty numpy arrays to maintain type consistency
            # Shape (0, 2) for coords, Shape (0,) for labels
            return np.empty((0, 2), dtype=int), np.array([], dtype=int)
        

        #Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(low_pae_coords)
        labels = clustering.labels_
        return  low_pae_coords, labels

    @staticmethod
    def cluster_info(low_coords: np.ndarray, cluster_labels: np.ndarray)-> pd.DataFrame:
        
        """
        Extracts geometric stats from clustered PAE coordinates (bounding box, center, density).

        Analyzes output from DBSCAN. For each cluster, calculates the bounding box 
        dimensions and centroids to characterize the interaction interface shape.
        
        Note on Coordinates: 
        Assumes Matrix notation (Row, Col). Maps Row -> Y, Col -> X.

        Args:
            low_coords (np.ndarray): Shape (N, 2). Coordinates [row, col].
            cluster_labels (np.ndarray): Shape (N,). Cluster ID for each point.

        Returns:
            pd.DataFrame: Contains columns ['cluster_id', 'num_points', 'x_min', 
            'x_max', 'y_min', 'y_max', 'x_len', 'y_len', 'center_x', 'center_y', 
            'aspect_ratio'].
        """

        unique_labels = np.unique(cluster_labels)
        cluster_info_list = []

        for label in unique_labels:
            # -1 represents noise in DBSCAN
            if label == -1:
                continue 

            # Boolean masking to select points belonging to this cluster
            cluster_points = low_coords[cluster_labels == label]
            
            # --- Coordinate Logic ---
            # Numpy: Axis 0 = Rows (Y), Axis 1 = Cols (X)
            rows = cluster_points[:, 0]
            cols = cluster_points[:, 1]

            # Calculate Bounding Box
            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)


            # Dimensions
            y_len = y_max - y_min
            x_len = x_max - x_min

            # Center of Mass 
            center_row = np.mean(rows)
            center_col = np.mean(cols)
            
            # Cluster Ratio (Longest side / Shortest side)
            min_side = min(x_len, y_len)
            max_side = max(x_len, y_len)
            cluster_ratio = max_side / min_side if min_side > 0 else 0.0
            
            cluster_info_list.append({
                "cluster_id": label,
                "num_points": len(cluster_points),
                "x_len": x_len,
                "y_len": y_len,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "center_x": round(center_col, 2), # Col is X
                "center_y": round(center_row, 2), # Col is Y
                "cluster_ratio": round(cluster_ratio, 2)
            })

        # Define columns explicitly to ensure DataFrame structure even if empty
        columns = [
            "cluster_id", "num_points", 
            "x_len", "y_len", "x_min", "x_max", 
            "y_min", "y_max", "center_x", "center_y", "cluster_ratio"
        ]

        return pd.DataFrame(cluster_info_list, columns=columns)
    

    @staticmethod
    def process_ppi(model_file: Union[str, Path], 
                    model_type: str = "AF3", 
                    mode: str = "heteromers", 
                    prefix: str = "")-> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Processes an AlphaFold3/Boltz CIF model file STRICTLY.

        Parses metadata, loads structural data, plots QA metrics, and analyzes interfaces.
        
        WARNING: This method has NO error handling. 
        - File names MUST follow conventions (e.g., "GeneA__GeneB").
        - Model names MUST contain "model_X".
        - Plotting libraries MUST work correctly.
        Any deviation will cause a crash.

        Args:
            model_file (Union[str, Path]): Path to the model file (.cif).
            model_type (str): Engine used ('AF3' or 'Boltz'). Defaults to "AF3".
            mode (str): Processing mode. Defaults to "heteromers".
            prefix (str): Optional prefix to strip from folder names.

        Returns:
            Tuple[Dict, pd.DataFrame]: Summary metrics and Cluster details.
        """
        
        path_obj = Path(model_file)
        logger.info(f"Processing model: {path_obj.name}")
        
        # Parse metadata using pathlib
        dir_name = path_obj.parent.name
        # dir_name = os.path.dirname(model_file)
        ppi_id = dir_name.replace(prefix, "")
        # ppi_id = dir_name.split("/")[-1].replace(prefix, "")
        
        # Assumes "ORF_A__ORF_B" format. 
        orf_a, orf_b = ppi_id.split("__")[:2]
        # orf_a, orf_b = ppi_id.split("__")

        # Assumes "...model_1..." format.
        base_name = path_obj.stem # removes .cif
        # base_name = os.path.basename(model_file).replace(".cif", "").replace(".pdb", "")
        
        model_number = int(base_name.split("model_")[-1].split("_")[0])
        # model_number = int(base_name.split("_")[-1].replace("model_",""))

        # Load Full Data
        if model_type.lower() == "af3":
            full_data = process_full_data_af3(str(path_obj))
            # full_data = process_full_data_af3(model_file)
            # molecule_model = MoleculeModel.from_af3(model_file)
        elif model_type.lower() == "boltz":
            full_data = process_full_data_boltz(str(path_obj))
            # full_data = process_full_data_boltz(model_file)
            # molecule_model = MoleculeModel.from_boltz(model_file)
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'AF3' or 'Boltz'.")

        ## Plotting pLDDT
        plddt_path = path_obj.with_name(f"{path_obj.stem}_plddt.png")
        # plddt_save_name = model_file.replace(".cif", "_plddt.png").replace(".pdb", "_plddt.png")

        plot_plddt(full_data["ca_plddts"],
                   full_data["chain_boundaries_by_res"],
                   full_data["token_chain_ids"],
                   str(plddt_path))
        
        # Plotting PAE
        pae_path = path_obj.with_name(f"{path_obj.stem}_pae.png")
        # pae_save_name = model_file.replace(".cif", "_pae.png").replace(".pdb", "_pae.png")
        
        plot_paes(full_data["pae"],
                  full_data["chain_boundaries_by_res"],
                  set(full_data["token_chain_ids"]),
                  f"{full_data.get('iptm', 'N/A')} ipTM - {full_data.get('ptm', 'N/A')} pTM",
                  str(pae_path))

        # Interface Analysis (Only for Pairs). Not for homomers
        all_metrics = {}
        cluster_data = pd.DataFrame()

        chain_ids = full_data.get("token_chain_ids")
        # chain_ids = full_data["token_chain_ids"]
        unique_chains = sorted(list(set(chain_ids))) if chain_ids is not None else []

        if len(unique_chains) == 2:
        # if len(set(full_data["token_chain_ids"])) == 2: ## We have two chains
            
            # We explicitly identify chains
            chain_a, chain_b = unique_chains[0], unique_chains[1]

            # Metrics
            all_metrics = calculate_all_metrics(str(path_obj), full_data)
            # all_metrics = calculate_all_metrics(model_file, full_data)

            # Symmetrized PAE Matrix Calculation
            pae = full_data["pae"]

            # Matrix 1: Effect of A on B
            pae_submatrix_1 = pae[chain_ids == chain_a][:, chain_ids == chain_b]
            # Matrix 2: Effect of B on A (Transposed)
            pae_submatrix_2 = pae[chain_ids == chain_b][:, chain_ids == chain_a].T
            # Mean error between directions
            submatrix = np.mean([pae_submatrix_1, pae_submatrix_2], axis=0)

            # Clustering
            low_coords, cluster_labels = InteractomeProcessor.cluster_pae(submatrix)

            # Plot clusters
            cluster_plot_path = path_obj.with_name(f"{path_obj.stem}_cluster.png")
            plot_pae_clusters(submatrix,low_coords, cluster_labels, save_name=str(cluster_plot_path))

            cluster_data = InteractomeProcessor.cluster_info(low_coords=low_coords, cluster_labels=cluster_labels)
            
            # Enrich DataFrame with Metadata
            if not cluster_data.empty:
                cluster_data["PPI"] = ppi_id 
                cluster_data["model_num"] = model_number 
                cluster_data["path"] = str (path_obj) 
               

        # Return summary metrics and cluster details
        summary_dict = {"PPI": ppi_id,
                "ORF_A": orf_a,
                "ORF_B": orf_b,
                "Folder": str(path_obj.parent), 
                "Model_num": model_number, 
                "ipTM": full_data["iptm"], 
                "pTM": full_data["ptm"],
                "pTM_chain_A": full_data["iptm_chain_pair"][0][0], 
                "pTM_chain_B": full_data["iptm_chain_pair"][1][1], 
                **all_metrics
                }
        
        return summary_dict, cluster_data
    
  
    def process_models(self,
                       output_path: Union[str, Path] = ".",
                       prefix: str = "", 
                       **kwargs)-> None:
        """
        Orchestrates the parallel processing of protein models and saves results to CSV.

        This method manages the full pipeline execution. It implements a 'resume' logic by
        checking for existing output files and skipping models that have already been processed.
        It utilizes a process pool to distribute the workload across multiple CPU cores and
        finally aggregates all metrics and cluster data into consolidated CSV files.

        Parameters
        ----------
        output_path : str or Path, optional
            The directory where output CSV files ('interactome_data.csv' and 'clusters_data.csv')
            will be saved. Defaults to current directory (".").
        prefix : str, optional
            A substring to be stripped from directory names when parsing PPI IDs (e.g., to clean
            up common prefixes in folder structures). Defaults to "".
        **kwargs
            Additional keyword arguments passed directly to `concurrent.futures.ProcessPoolExecutor`.
            Most commonly used to set `max_workers` (e.g., `max_workers=4`) to limit CPU usage.

        Returns
        -------
        None
            This method does not return values; it produces side effects (CSV files on disk).
        """
        # Setup Paths
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        interactome_csv = out_dir / "interactome_data.csv"
        clusters_csv = out_dir / "clusters_data.csv"

        # Initialize containers
        interactome_df = pd.DataFrame()
        clusters_df = pd.DataFrame()
        # Models to process (initially all)
        models_to_process = self.model_paths.copy()

        # Resume Logic (Check existing data)
        if interactome_csv.exists():
            logger.info(f"Found existing data at {interactome_csv}. Resuming run...")
            interactome_df = pd.read_csv(interactome_csv)
            if clusters_csv.exists():
                clusters_df = pd.read_csv(clusters_csv)
            # Filter out models that are already in the CSV
            # We assume the 'Folder' column contains the parent directory path
            if "Folder" in interactome_df.columns:
                processed_folders = set(interactome_df["Folder"].astype(str))
                    
                # Identify which paths from input are not in the processed set
                # We convert path.parent to string to match the CSV format
                models_to_process = [
                    p for p in models_to_process 
                    if str(p.parent) not in processed_folders
                ]
                logger.info(f"Skipping {len(self.model_paths) - len(models_to_process)} already processed models.")
        if not models_to_process:
            logger.info("All models are already processed. Exiting.")
            return
        
        logger.info(f"Starting parallel processing for {len(models_to_process)} models...")
    
        # Parallel Execution
        new_interactome_list = []
        new_clusters_list = []

        # We set default workers if not provided in kwargs
        # max_workers = kwargs.get('max_workers', None) # None lets ProcessPoolExecutor decide
        with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
            worker = partial(self.process_ppi,
                             model_type=self.engine,
                             prefix=prefix)
            
            # tqdm for progress bar
            # map maintains order, which matches models_to_process order
            results_iterator = tqdm.tqdm(
                executor.map(worker, models_to_process), 
                total=len(models_to_process),
                desc="Processing Models"
            )
            for metrics_dict, cluster_df in results_iterator:
                new_interactome_list.append(metrics_dict)
                if not cluster_df.empty:
                    new_clusters_list.append(cluster_df)
        
        # Aggregation
        logger.info("Aggregating and saving results...")  

        # Combine Lists into DataFrames
        new_interactome_df = pd.DataFrame(new_interactome_list)
        
        if new_clusters_list:
            new_clusters_df = pd.concat(new_clusters_list, ignore_index=True)
        else:
            new_clusters_df = pd.DataFrame()  

        # Concatenate with Existing Data (Resume)
        final_interactome_df = pd.concat([interactome_df, new_interactome_df], ignore_index=True)
        final_clusters_df = pd.concat([clusters_df, new_clusters_df], ignore_index=True)
       
        # Save dfs to .csv
        final_interactome_df = final_interactome_df.round(2)
        final_interactome_df.to_csv(interactome_csv, index=False)

        # Formatting Clusters DataFrame
        if not final_clusters_df.empty:
            desired_columns = [
                'PPI', 'model_num', 'path', 'cluster_id', 'num_points', 
                'x_len', 'y_len', 'x_min', 'x_max', 'y_min', 'y_max',
                'center_x', 'center_y', 'cluster_ratio'
            ]
        # Ensure columns exist (handle case-sensitivity if old CSV had 'Cluster_ratio')
            if 'Cluster_ratio' in final_clusters_df.columns and 'cluster_ratio' not in final_clusters_df.columns:
                final_clusters_df.rename(columns={'Cluster_ratio': 'cluster_ratio'}, inplace=True)
            
            # Select and order columns safely
            # Only select columns that actually exist to avoid KeyError
            existing_cols = [c for c in desired_columns if c in final_clusters_df.columns]
            final_clusters_df = final_clusters_df[existing_cols]

            final_clusters_df = final_clusters_df.round(2)
            final_clusters_df.to_csv(clusters_csv, index=False)
        
        logger.info(f"Done. Data saved to {out_dir}")

class InteractomeAnalyzer:
    """
    Manages the loading, validation, and path manipulation of interactome analysis datasets.

    This class serves as a data manager that:
    1. Loads interactome and cluster CSV files generated by the InteractomeProcessor.
    2. Validates data integrity (checking for empty files and missing columns).
    3. Ensures consistency between datasets (e.g., matching PPI IDs).
    4. Handles path relocation, allowing analysis to be ported between different machines/folders.

    Attributes
    ----------
    output_path : Path
        Base directory for output generation.
    REQUIRED_INTERACTOME_COLS : list
        List of column names required in the interactome CSV ("PPI", "Folder").
    REQUIRED_CLUSTER_COLS : list
        List of column names required in the cluster CSV ("PPI", "path", "cluster_id").
    """
    REQUIRED_INTERACTOME_COLS = ["PPI", "Folder"]
    REQUIRED_CLUSTER_COLS = ["PPI", "path", "cluster_id"]
    
    def __init__(self, output_path : Union[str, Path] = "."):
        """
        Initializes the Analyzer with a default output path.

        Parameters
        ----------
        output_path : str or Path, optional
            The directory where results will be saved. Defaults to current directory (".").
        """
        self.output_path = Path(output_path)
        
        self._interactome_path: Optional[Path] = None
        self._interactome_data: Optional[pd.DataFrame] = None
        self._cluster_path: Optional[Path] = None
        self._cluster_data: Optional[pd.DataFrame] = None
        self._models_path: Optional[str] = None
        self._binder_data: Optional[pd.DataFrame] = None
        self._candidate_clusters: Optional[pd.DataFrame] = None
    
    
    #Getters and setters

    # -------------------------------------------------------------------------
    # Binder Data
    # -------------------------------------------------------------------------
    
    @property
    def binder_data(self)-> Optional[pd.DataFrame]:
        return self._binder_data
    
    @binder_data.setter
    def binder_data(self, df: pd.DataFrame):
        self._binder_data = df

    # -------------------------------------------------------------------------
    # Interactome Data Management
    # -------------------------------------------------------------------------
    
    @property
    def interactome_path(self)-> Optional[Path]:
        return self._interactome_path
    
    @property
    def interactome_data(self)-> Optional[pd.DataFrame]:
        return self._interactome_data 
    
    @interactome_path.setter
    def interactome_path(self, interactome_data_path: str):
        """
        Sets the interactome file path and triggers strict validation and loading.
        """

        path = Path(interactome_data_path)
        
        # 1. Existence Check
        if not path.exists():
            raise FileNotFoundError(f"Interactome file not found at: {path}")
        
        logger.info(f"Loading interactome data from {path}...")
        df = pd.read_csv(path)

        # 2. Empty Check
        if df.empty:
            raise ValueError(f"The interactome file {path} is empty. Cannot proceed.")
        
        # 3. Column Validation
        missing_cols = [c for c in self.REQUIRED_INTERACTOME_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Interactome file invalid. Missing required columns: {missing_cols}")
        
        self._interactome_path = path

        self._interactome_data = df
        logger.info(f"Successfully loaded {len(df)} interactome records.")
    

    # -------------------------------------------------------------------------
    # Cluster Data Management
    # -------------------------------------------------------------------------
    
    @property
    def cluster_path(self)-> Optional[Path]:
        return self._cluster_path 
    
    @property
    def cluster_data(self)-> Optional[pd.DataFrame]:
        return self._cluster_data 
    
    @cluster_path.setter
    def cluster_path(self, cluster_data_path: Union[str, Path]):
        """
        Sets the cluster file path, loads data, validates it, and calculates the common root path.
        """
        path = Path(cluster_data_path)
        
        # 1. Existence Check
        if not path.exists():
            raise FileNotFoundError(f"Cluster file not found at: {path}")
        
        logger.info(f"Loading cluster data from {path}...")
        df = pd.read_csv(path)

        # 2. Empty Check
        if df.empty:
            raise ValueError(f"The cluster file {path} is empty.")

        # 3. Column Validation
        missing_cols = [c for c in self.REQUIRED_CLUSTER_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Cluster file invalid. Missing required columns: {missing_cols}")
        
        # 4. Consistency Check
        if self._interactome_data is not None:
            interactome_ppis = set(self._interactome_data["PPI"])
            cluster_ppis = set(df["PPI"])
            orphans = cluster_ppis - interactome_ppis
            
            if orphans:
                logger.warning(f"Consistency Warning: Found {len(orphans)} PPI IDs in clusters "
                               f"that are NOT present in the interactome data.")
        
        # 5. Determine Common Path (Logic for _models_path)
        # We try to find the common prefix of all paths to establish the 'models_path'
        if "path" in df.columns:
            # Drop NAs and convert to list
            paths = df["path"].dropna().tolist()
            if paths:
                try:
                    self._models_path = os.path.commonpath(paths)
                    logger.info(f"Detected common models root path: {self._models_path}")
                except ValueError as e:
                    logger.warning(f"Could not automatically determine common path: {e}")
                    self._models_path = ""        
        
        self._cluster_path = path

        self._cluster_data = df
       
    @property
    def models_path(self)-> Optional[str]:
        return self._models_path 

    @models_path.setter
    def models_path(self, new_model_path: str):
        """
        Updates the root directory for models across all loaded datasets.
        Useful when moving analysis to a new machine.
        """
        if self._cluster_data is None or self._interactome_data is None:
            raise RuntimeError("Cannot update models_path: Data not loaded. Set interactome_path and cluster_path first.")
        
        old_path = self._models_path if self._models_path else ""
        logger.info(f"Relocating models: replacing '{old_path}' with '{new_model_path}'")
        
        # Update Cluster Data
        # regex=False ensures special characters (like Windows backslashes) are treated literally
        if "path" in self._cluster_data.columns:
            self._cluster_data["path"] = self._cluster_data["path"].astype(str).str.replace(old_path, new_model_path, regex=False)
            # self._cluster_data.loc[: , "path"] = self._cluster_data.path.str.replace(self._models_path, new_model_path)
        
        # Update Interactome Data
        if "Folder" in self._interactome_data.columns:
            self._interactome_data["Folder"] = self._interactome_data["Folder"].astype(str).str.replace(old_path, new_model_path, regex=False)
            # self._interactome_data.loc[: , "Folder"] = self._interactome_data.Folder.str.replace(self._models_path, new_model_path)
        
        self._models_path = new_model_path
        logger.info(f"Path relocation complete. New root: {self._models_path}")
       
        return self._cluster_path 

    def __str__(self)-> str:
        """Returns a summary of the loaded data status."""
        interactome_state = str(self._interactome_path) if self._interactome_path else "Not Loaded"
        interactome_len = len(self._interactome_data) if self._interactome_data is not None else 0
        
        cluster_state = str(self._cluster_path) if self._cluster_path else "Not Loaded"
        cluster_len = len(self._cluster_data) if self._cluster_data is not None else 0
        
        return f"""<InteractomeAnalyzer>
        ---------------------------
        Interactome: {interactome_state}
        Records:     {interactome_len}
        ---------------------------
        Clusters:    {cluster_state}
        Records:     {cluster_len}
        ---------------------------
        """
    
    def __len__(self)-> int:
        if self._interactome_data is not None:
            return len(self._interactome_data)
        return 0
    
    def run_full_pipeline(self, **kwargs):
        """
        Executes the complete analysis pipeline.

        Currently triggers the peptide-protein pair analysis.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to downstream methods, such as:
            - cluster_ratio_threshold (float, default=5)
            - min_peptide_len (int, default=5)
        """
        if self._cluster_data is None:
            logger.warning("Cannot run pipeline: Cluster data is missing.")
            return
        logger.info("Starting peptide-protein analysis pipeline...")
        self.analyze_peptide_proteins_pairs(**kwargs)
    
    def _get_candidate_clusters(self, cluster_ratio_threshold: float = 5.0, 
                                min_peptide_len: int = 5)-> pd.DataFrame:
        """
        Identifies candidate peptide-protein interactions based on cluster geometry.

        Filters clusters that have a high aspect ratio (elongated shape), suggesting 
        a peptide binding to a larger protein surface. It determines which chain 
        corresponds to the 'Binder' (protein) and which to the 'Peptide' based on 
        the dimensions of the interaction interface.

        Parameters
        ----------
        cluster_ratio_threshold : float, optional
            The minimum aspect ratio (max_side / min_side) required to consider 
            a cluster as a peptide candidate. Defaults to 5.0.
        min_peptide_len : int, optional
            The minimum length (in residues) of the interface to be considered valid. 
            Defaults to 5.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing candidate clusters with additional columns:
            - Binder_name, Binder_chain, Binder_start, Binder_end
            - Peptide_name, Peptide_chain, Peptide_start, Peptide_end
        """
        # 1. Validate Data Availability
        if self._cluster_data is None or self._cluster_data.empty:
            logger.warning("Cluster data is empty. No candidates to process.")
            return pd.DataFrame()

        #Generate a copy of the df with candidate clusters
        df = self._cluster_data.copy()

        # 2. Handle Legacy Column Naming (Cluster_ratio vs cluster_ratio)
        ratio_col = "cluster_ratio" if "cluster_ratio" in df.columns else "Cluster_ratio"
        
        # 3. Apply Filters (Geometry & Thresholds)
        # Ensure dimensions are positive and ratio meets threshold
        mask = (
            (df["x_len"] > 0) & 
            (df["y_len"] > 0) & 
            (df[ratio_col] > cluster_ratio_threshold) & 
            (df["x_len"] >= min_peptide_len) & 
            (df["y_len"] >= min_peptide_len)
        )
        candidate_clusters = df[mask].copy()

        # df = df[(df.x_len > 0) & (df.y_len > 0)]
        # candidate_clusters = df[df.Cluster_ratio > cluster_ratio_threshold].copy()
        # candidate_clusters = candidate_clusters.loc[
        #     (candidate_clusters.x_len >= min_peptide_len) & 
        #     (candidate_clusters.y_len >= min_peptide_len), :]

        # 4. Process Candidates (Identify Binder vs Peptide)
        # Lists to store new column data
        new_cols = {
            "Binder_chain": [], "Binder_name": [], "Binder_start": [], "Binder_end": [],
            "Peptide_chain": [], "Peptide_name": [], "Peptide_start": [], "Peptide_end": []
        }
        
        peptide_start, peptide_end = [], []
        binder_start, binder_end = [],[]
        binder_name, peptide_name = [], []
        peptide_chain = []
        binder_chain = []

        for _, row in candidate_clusters.iterrows():
            # Robust split (take first two elements only) to handle names like "GenA__GenB__v2"
            parts = row["PPI"].split("__")
            if len(parts) >= 2:
                orf_a, orf_b = parts[:2]
            else:
                # Fallback for malformed PPI IDs
                orf_a, orf_b = row["PPI"], ""

            # Logic: Assign roles based on dimensions
            # If X dimension > Y dimension -> Chain A is Binder, Chain B is Peptide
            
            if row["x_len"] > row["y_len"]:
                # X is longer
                new_cols["Binder_chain"].append("A")
                new_cols["Peptide_chain"].append("B")
                # binder_chain.append("A")
                # peptide_chain.append("B")

                new_cols["Peptide_start"].append(int(row["y_min"]))
                new_cols["Peptide_end"].append(int(row["y_max"]))
                new_cols["Binder_start"].append(int(row["x_min"]))
                new_cols["Binder_end"].append(int(row["x_max"]))
                # peptide_start.append(int(row.y_min))
                # peptide_end.append(int(row.y_max))
                # binder_start.append(int(row.x_min))
                # binder_end.append(int(row.x_max))
                
                new_cols["Binder_name"].append(orf_a)
                new_cols["Peptide_name"].append(orf_b)
                # binder_name.append(orf_a)
                # peptide_name.append(orf_b)
            else:
                # Y is longer
                new_cols["Binder_chain"].append("B")
                new_cols["Peptide_chain"].append("A")
                # binder_chain.append("B")
                # peptide_chain.append("A")
                new_cols["Peptide_start"].append(int(row["x_min"]))
                new_cols["Peptide_end"].append(int(row["x_max"]))
                new_cols["Binder_start"].append(int(row["y_min"]))
                new_cols["Binder_end"].append(int(row["y_max"]))
                # peptide_start.append(int(row.x_min))
                # peptide_end.append(int(row.x_max))
                # binder_start.append(int(row.y_min))
                # binder_end.append(int(row.y_max))

                new_cols["Binder_name"].append(orf_b)
                new_cols["Peptide_name"].append(orf_a)
                # binder_name.append(orf_b)
                # peptide_name.append(orf_a)
        
        # 5. Assign new columns to DataFrame
        for col_name, data_list in new_cols.items():
            candidate_clusters[col_name] = data_list

        # candidate_clusters[f"Binder_chain"] = binder_chain
        # candidate_clusters["Binder_name"] = binder_name
        # candidate_clusters["Peptide_chain"] = peptide_chain
        # candidate_clusters["Peptide_name"] = peptide_name
        # candidate_clusters["Peptide_start"] = peptide_start
        # candidate_clusters["Peptide_end"] = peptide_end
        # candidate_clusters[f"Binder_start"] = binder_start
        # candidate_clusters[f"Binder_end"] = binder_end
        
        
        # Return with a clean index
        return candidate_clusters.reset_index(drop=True)

    def _curate_protein_peptide_models(self, data: pd.Series)-> Molecule:
        """
        Loads a PDB model and standardizes chain identifiers for analysis.

        This method ensures a consistent schema where:
        1. Chain A represents the 'Binder' (Protein).
        2. Chain B represents the 'Peptide'.
        3. Only the interface residues of the peptide are kept.

        Parameters
        ----------
        data : pd.Series
            A row from the candidate clusters DataFrame containing:
            - "path": Path to the PDB file.
            - "Peptide_chain": Original chain ID of the peptide ('A' or 'B').
            - "Peptide_start": Start residue index of the peptide interface.
            - "Peptide_end": End residue index of the peptide interface.

        Returns
        -------
        Molecule
            A MoleculeKit object with standardized chains and filtered atoms.
        """
        # Extract and cast types
        mol_path = str(data["path"])
        peptide_chain = str(data["Peptide_chain"])
        peptide_start = int(data["Peptide_start"])
        peptide_end = int(data["Peptide_end"])
        # mol_path, peptide_chain , peptide_start, peptide_end = data[["path", "Peptide_chain", "Peptide_start", "Peptide_end"]]

        mol = Molecule(mol_path)

        # Standardize: Binder -> A, Peptide -> B
        if peptide_chain == "A":
            # Swap logic using temporary chain 'C'
            mol.set("chain", "C", "chain A") # Peptide(A) -> C
            mol.set("chain", "A", "chain B") # Binder(B) -> A
            mol.set("chain", "B", "chain C") # Peptide(C) -> B
       
        # Filter: Keep Binder (A) or Peptide (B) within interface range
        mol.filter(f"(chain A) or (chain B and resid {peptide_start} to {peptide_end})")
        return mol
    
    
    ## Continuar aqui el refactoring del codigo           
    def _create_binder_alignments(self, model_to_align, reference_model):
   
        if isinstance(reference_model, str):
            reference_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            reference_mol = reference_model
        else:
            raise ValueError("reference_model should be a path (str) or a Molecule instance")
        residues_in_reference_chain = reference_mol.resid.astype(str)
        reference_resid_str = ' '.join(residues_in_reference_chain)

        tmp_mol = Molecule(model_to_align)
        tmp_mol.filter(f"(chain A and resid {reference_resid_str}) or (chain B)")
        tmp_mol.align(f"chain A",
                    refmol=reference_mol,
                    refsel=f"chain A",
                    mode="index"
                    )
        return tmp_mol
    
    def _get_reference_structure_for_binder(self, all_structs: list[str]) -> str:
        mol_list = [ Molecule(i) for i in all_structs]
        plddt_scores = []
        for mol in mol_list:
            ## Select the chain A which is the binder
            plddt_chain_A = mol.beta[mol.chain == "A"]
            # Calculate the median pLDDT just for the binder
            score = np.median(plddt_chain_A)
            plddt_scores.append(score)
        
        plddt_scores = np.array(plddt_scores)

        # Select the best index for the best binder 
        best_global_idx = np.argmax(plddt_scores)

        # Define the reference molecule, just one by binder
        reference_mol = mol_list[best_global_idx].copy()
        reference_mol.filter("chain A")
        reference_resids = reference_mol.resid[reference_mol.name == "CA"][reference_mol.beta[reference_mol.name == "CA"]>70]
        reference_resids = reference_resids.astype(str)
        reference_resid_str = ' '.join(reference_resids)

        reference_mol.filter(f"resid {reference_resid_str}")
        return reference_mol

    def analyze_peptide_proteins_pairs(self, **kwargs):
        output_path = f"{self.output_path}/prot_peptide"
        os.makedirs(output_path, exist_ok=True)
        ## Find candidates clusters
        self._candidate_clusters = self._get_candidate_clusters()

        for binder in self._candidate_clusters.Binder_name.unique():
            os.makedirs(f"{output_path}/{binder}/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/filtered/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/aligned/", exist_ok=True)

        df = self._candidate_clusters.loc[:, ["PPI", "model_num", "x_len", "y_len", 
                                              "Binder_name", "Binder_chain", "Binder_start", "Binder_end",
                                              "Peptide_name", "Peptide_chain", 
                                             "Peptide_start", "Peptide_end",  "path"]]
        
        df.loc[: , "PPI"] = df.PPI + "_" +  self._candidate_clusters.model_num.astype(str) + "_" + self._candidate_clusters.cluster_id.astype(str)
        
        ## I want to filter all structures
        filtered_names = []
        for idx, row in df.iterrows():
            output_name = f"{output_path}/{row.Binder_name}/filtered/{row.PPI}.pdb"
            filtered_names.append(output_name)
            if not os.path.exists(output_name):
                mol = self._curate_protein_peptide_models(row)
                mol.write(output_name)
            else:
                print(f"Skiping {output_name}, already_filtered...")
        df.loc[:, "filtered_path"] = filtered_names
        ## Iterate over each binder
        binder_df = pd.DataFrame()
        for binder in self._candidate_clusters.Binder_name.unique():

            ppi_data = df.loc[self._candidate_clusters.Binder_name == binder,:]
           
            all_structs = ppi_data.filtered_path.values
            
            ## Get reference structure
            reference_output_name = f"{output_path}/{binder}/reference_{binder}.pdb"
            if not os.path.exists(reference_output_name):
                reference_molecule = self._get_reference_structure_for_binder(all_structs)
                reference_molecule.write(reference_output_name)
            else:
                reference_molecule = Molecule(reference_output_name)
                print(f"Skiping {reference_output_name}, already_filtered...")

            aligned_models = []
            for tmp_mol_name in all_structs:
                tmp_mol_name_aligned = tmp_mol_name.replace("filtered", "aligned")
                if not os.path.exists(tmp_mol_name_aligned):
                    tmp_mol = self._create_binder_alignments(tmp_mol_name, reference_molecule)
                    tmp_mol.write(tmp_mol_name_aligned)
                else:
                    print(f"Skiping {tmp_mol_name_aligned}, already_filtered...")
                aligned_models.append(tmp_mol_name_aligned)

            ## Clustering coordinates
            tmp_df, cluster_info = self.cluster_protein_peptides(aligned_models, reference_output_name, **kwargs)
            tmp_df.insert(0, "Binder", binder)
            binder_df = pd.concat([binder_df, tmp_df], ignore_index=True)

            ## Write chimerax script
            ppi_data.loc[:, "Cluster_info"] = cluster_info.get("cluster_labels")
            peptide_centers = cluster_info.get("peptide_centers")
            ppi_data.loc[:, "Center_X"] = peptide_centers[:,0]
            ppi_data.loc[:, "Center_Y"] = peptide_centers[:,1]
            ppi_data.loc[:, "Center_Z"] = peptide_centers[:,2] 
            self._create_chimera_session(ppi_data, reference_output_name, tmp_df)
        binder_df.to_csv(f"{self.output_path}/peptide_binder_info.csv", index=False)

    def _create_chimera_session(self, ppi_data, ref_model, cluster_info):
        binder = ppi_data.Binder_name.values[0]
        script_path = f"{self.output_path}/prot_peptide/{binder}/{binder}_peptide_binding.cxc"
        session_path = f"{self.output_path}/prot_peptide/{binder}/{binder}_peptide_binding.cxs" 
        available_colors = ["cyan", "yellow", "magenta", "orange", "cornflower blue"]
        available_colors_ref = ["light coral", "medium slate blue", "orange", "green", "red", "yellow"]
        with open(script_path, "w") as f:
            ## Load reference_model and rename
            f.write("graphics silhouettes true\n") 
            f.write("lighting soft\n") 
            f.write("set bg white\n") 

            ## Load reference   
            f.write(f"\n# --- {binder} REFERENCE ---\n")
            f.write(f"open {ref_model}\n")
            f.write(f"rename #1 {binder}_ref\n")

            ## Color ref residues
            for idx, cluster_data in cluster_info.iterrows():
                cluster_id = cluster_data.Cluster_label
                if cluster_id == -1:
                    continue # We don't paint the orf resids which are nearby the noise
                # tmp_sel_str = paint_orf_res_chimerax(resids_list=resids_list)
                tmp_sel_str = ",".join(map(str, cluster_data.Residues))
                color_index = cluster_id % len(available_colors_ref)
                color_str = available_colors_ref[color_index]
                f.write(f"color #1:{tmp_sel_str} {color_str}\n") 

                ## Draw centroid sphere
                super_id = f"#5.{cluster_id + 1}"
                global_center_str = f"{cluster_data.Center_X},{cluster_data.Center_Y},{cluster_data.Center_Z}"
                f.write(f"shape sphere name Centroid_{cluster_id+1}_Mean radius 3 center {global_center_str} color {color_str} model {super_id} \n")
             
            ## Load peptide_proteins and rename
            aligned_models = ppi_data.filtered_path.str.replace("filtered", "aligned")
            ppi_data.loc[:, "aligned_path"] = aligned_models
            for tmp_cluster in ppi_data.Cluster_info.unique():
                current_sub_id = tmp_cluster + 2
                if tmp_cluster == -1:
                    color_str = "silver"
                    group_name = "Unclassified"
                else:
                    color_idx = tmp_cluster % len(available_colors) ## Save index in order to not have IndexError
                    color_str = available_colors[color_idx]
                    group_name = f"Cluster_{tmp_cluster + 1}"

                models_in_cluster = ppi_data.loc[ ppi_data.Cluster_info == tmp_cluster, :]
                for i, (idx, row) in enumerate(models_in_cluster.iterrows()):
            
                    ## Usamos i porque idx viene del df padre y no tiene por que ir en orden puedes tener 1,5,39....
                    pep_id = f"#3.{current_sub_id}.{i + 1}" # model #3 for peptides
                    cen_id = f"#4.{current_sub_id}.{i + 1}" # model #4 for centroids
                    # Open peptide 
                    f.write(f"open {row.aligned_path} id {pep_id}\n")
                    if tmp_cluster == -1:
                        model_name = f"{row.PPI}_unclassified"    
                    else:
                        # Rename peptide
                        model_name = f"{row.PPI}_c{current_sub_id}"
                    
                    f.write(f"rename {pep_id} {model_name}\n")
                    f.write(f"color {pep_id} {color_str}\n")
                    centroid_str = f"{row.Center_X},{row.Center_Y},{row.Center_Z}"
                    f.write(f"shape sphere name {model_name} radius 1 center {centroid_str} color {color_str} model {cen_id}\n")
                f.write(f"rename #3.{current_sub_id} {group_name}\n")
                f.write(f"rename #4.{current_sub_id} {group_name}_Center_of_mass\n")   

            
            ## Draw sphere in cluster centroid and rename
            f.write(f"lighting depthCue false\n")
            f.write(f"rename #3 Peptides\n")
            f.write(f"hide #3/A cartoon\n") # From the peptides we hide the chain A which is the chain of the ORF
            f.write(f"hide atoms\n")
            f.write(f"rename #4 Peptide_centers\n")
            # f.write(f"rename #5 Centroids\n")
            f.write(f"save {session_path}\n")
            f.write(f"exit\n")
        os.system(f"chimerax --nogui {script_path}") ##--nogui 
    
    def cluster_protein_peptides(self, aligned_models: list[str], reference_model: Molecule | str, **kwargs):
        ## Load molecules
        mols = [Molecule(i) for i in aligned_models]
        if isinstance(reference_model, str):
            ref_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            ref_mol = reference_model
        xyz = ref_mol.get("coords", sel="protein")
        
        ## get x, y, z means of coords
        
        mols_centroids = np.array([tmp_mol.get("coords", sel="chain B and name CA").mean(axis=0) for tmp_mol in mols])

        ## Cluster with DBSCAN
        clustering = DBSCAN(**kwargs).fit(mols_centroids)
        
        cluster_labels = []
        cluster_centers = []
        all_nearby_residues = []
        for cluster_label in np.unique(clustering.labels_):
            cluster_labels.append(cluster_label)
            cluster_center = mols_centroids[clustering.labels_ == cluster_label].mean(axis=0)
            cluster_centers.append(cluster_center)

            ## Calculate residues involved in the binding site
            tmp_centroid_distance = xyz - cluster_center
        
            # Euclidian distance from the ORF residues to the centroid of the currentcluster 
            tmp_euc = np.linalg.norm(tmp_centroid_distance, axis = 1)
            nearby_residues = np.unique(ref_mol.resid[tmp_euc < 8])
            all_nearby_residues.append(nearby_residues)

        cluster_centers = np.array(cluster_centers)
        return pd.DataFrame({
            "Cluster_label": cluster_labels,
            "Center_X": cluster_centers[:, 0],
            "Center_Y": cluster_centers[:, 1],
            "Center_Z": cluster_centers[:, 2],
            "Residues": all_nearby_residues
        }), {"cluster_labels": clustering.labels_, 
             "peptide_centers": mols_centroids}

#     def calculate_network(self):
#         pass

#     def basic_plots(self):
#         pass

#     def protein_peptide_analysis(self):
#         pass

#     def cluster_analysis(self):
#         pass
