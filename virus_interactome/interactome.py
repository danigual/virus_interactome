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
from typing import Dict, Iterable, List, Tuple, Optional, Any, Callable
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
        """Parses AF3 input with logging."""
        logger.debug(f"Parsing AF3 job: {input_json.name}")
        return load_json(str(input_json))


    def _parse_boltz2_job(self, input_yaml: Path)-> Dict[str, Any]:
        """Parses Boltz2 input with logging."""
        logger.debug(f"Parsing Boltz job: {input_yaml.name}")
        return load_boltz_input(str(input_yaml))

    
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
                "pTM_chain_A": full_data.get("iptm_chain_pair")[0][0], 
                "pTM_chain_B": full_data.get("iptm_chain_pair")[1][1], 
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
    def __init__(self, output_path = "."):
        self._interactome_path = None
        self._interactome_data = None
        self._cluster_path = None
        self._cluster_data = None
        self._candidate_clusters = None
        self._models_path = None
        self._binder_data = None
        self.output_path = output_path
    
    ## Getters and setters
    @property
    def binder_data(self):
        return self._binder_data
    
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
        # Summary for the user
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

        ## Study protein-peptides
        self.analyze_peptide_proteins_pairs()
    
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
        ## For each binder get a df with their metrics 
        ## pLDDT, PAE, minumun number of models per binder? Maybe it is interesting to use the median
        ## for pLDDT and PAE in order to not be affected by outliers?
        ## It may algo be interesting to filter within a PPI because we may have not so many models in a binder
        ## but all of them may correspond to the same PPI.  

        return candidate_clusters.reset_index()

    def _curate_protein_peptide_models(self, data):
        mol_path, peptide_chain , peptide_start, peptide_end = data[["path", "Peptide_chain", "Peptide_start", "Peptide_end"]]

        mol = Molecule(mol_path)

        if peptide_chain == "A":
            mol.set("chain", "C", "chain A")
            mol.set("chain", "A", "chain B")  ## chain A is binder
            mol.set("chain", "B", "chain C") ## chain B is peptide

        mol.filter(f"(chain A) or (chain B and resid {peptide_start} to {peptide_end})")
        return mol
                
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
        # best_global_idx = np.argmin(plddt_scores)

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
