import os
import time
import subprocess
import logging
import pandas as pd
import numpy as np
import yaml
import json
import concurrent.futures
import warnings
import csv
import logging

from sklearn.cluster import DBSCAN
from functools import partial
from itertools import combinations, product
from typing import Dict, Iterable, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path
from moleculekit.molecule import Molecule

from .utils import load_json, load_boltz_input, check_sequence_validity, process_full_data_af3, process_full_data_boltz, process_full_data_colabfold
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
            ValueError: If `proteome_b` is neither a string path nor a ProteomeManager 
                instance nor None.
            
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
        elif proteome_b is None:
            self.mode = "intra"
        else:
            raise ValueError("proteome_b must be a string, ProteomeManager, or None")

    
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

    @staticmethod
    def _build_colabfold_seq_str(seq_list: List[Tuple[str, str, int]]) -> str:
        """
        Builds the ColabFold multimer sequence string (SEQ_A:SEQ_B:...).

        Handles stoichiometry: count=2 for chain A repeats SEQ_A twice.

        Parameters
        ----------
        seq_list : List[Tuple[str, str, int]]
            List of (chain_id, sequence, count).

        Returns
        -------
        str
            Colon-separated sequence string ready for ColabFold.
        """
        parts = []
        for _, seq, count in seq_list:
            parts.extend([seq] * count)
        return ":".join(parts)
   
    def _build_seq_list_from_entry(self, kind: str, entry, mode: str,
                                counts_map: Optional[Dict[str, int]]) -> Tuple[list, str, str, str]:
        """
        Shared logic to build (seq_list, name, idA, idB) from an iterator entry.
        Extracted to avoid duplication between FASTA and CSV writers.
        """
        seq_list = []
        name = ""
        idA, idB = "", ""

        if kind == "pair":
            idA_raw, idB_raw = entry
            prot_A = self.proteome_a
            prot_B = self.proteome_b if mode == "inter_pairs" else self.proteome_a

            if mode == "intra_pairs" and idA_raw > idB_raw:
                idA, idB = idB_raw, idA_raw
                prot_A, prot_B = prot_B, prot_A
            else:
                idA, idB = idA_raw, idB_raw

            cntA = counts_map.get(idA, 1) if counts_map else 1
            cntB = counts_map.get(idB, 1) if counts_map else 1
            seq_list = [(idA, prot_A.sequences[idA], cntA),
                        (idB, prot_B.sequences[idB], cntB)]
            name = f"{idA}__{idB}"

        elif kind == "homo":
            pid, copies = entry
            seq_list = [(pid, self.proteome_a.sequences[pid], copies)]
            name = f"{pid}__{copies}"
            idA = pid

        elif kind == "single":
            pid, seq, cnt = entry
            seq_list = [(pid, seq, cnt)]
            name = pid
            idA = pid

        return seq_list, name, idA, idB 
    
    def _make_iterator(self, mode: str, nmin: int, nmax: int,
                    ids_a: Optional[list], ids_b: Optional[list]):
        """Builds the job iterator from mode. Shared between FASTA and CSV writers."""
        if mode == "intra_pairs":
            pairs = self.generate_intra_pairs()
            if ids_a:
                pairs = (p for p in pairs if p[0] in ids_a and p[1] in ids_a)
            return (("pair", p) for p in pairs)

        elif mode == "inter_pairs":
            return (("pair", p) for p in self.generate_inter_pairs())

        elif mode == "homomers":
            return (("homo", h) for h in self.generate_homo_mers(nmin=nmin, nmax=nmax))

        elif mode == "single":
            source = "both" if self.mode == "inter" else "a"
            return (("single", s) for s in self.generate_single_run(source=source, ids_a=ids_a, ids_b=ids_b))

        else:
            raise ValueError(f"Unknown mode: '{mode}'. Choose from: intra_pairs, inter_pairs, homomers, single.")


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
            
            if not check_sequence_validity(seq):
                return False, f"{chain_id} is not a valid protein sequence."

            total_res += len(seq) * count

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

    # =============================================================================
    # STRATEGY A — One FASTA per job (630 files for 36-protein interactome)
    # =============================================================================

    def write_colabfold_fastas(
        self,
        output_dir: str,
        *,
        mode: str = "intra_pairs",
        nmin: int = 2,
        nmax: int = 6,
        counts_map: Optional[Dict[str, int]] = None,
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None,
        index_name: str = "colabfold_index.csv",
    ) -> List[dict]:
        """
        Writes one FASTA file per job for colabfold_batch.

        Each pair generates a separate .fasta file:

            >ProtA__ProtB
            SEQUENCEA:SEQUENCEB

        This enables per-job control: easy resume, individual monitoring,
        and selective re-runs of failed jobs.

        Parameters
        ----------
        output_dir : str
            Directory where FASTA files and the index CSV will be saved.
        mode : str, optional
            Generation strategy: 'intra_pairs', 'inter_pairs', 'homomers', 'single'.
            Defaults to "intra_pairs".
        nmin : int, optional
            Minimum stoichiometry for homomers. Defaults to 2.
        nmax : int, optional
            Maximum stoichiometry for homomers. Defaults to 6.
        counts_map : dict, optional
            Custom stoichiometry override per protein ID. Defaults to None.
        ids_a : list, optional
            Subset of Proteome A IDs to process.
        ids_b : list, optional
            Subset of Proteome B IDs to process.
        index_name : str, optional
            Name for the metadata CSV. Defaults to "colabfold_index.csv".

        Returns
        -------
        List[dict]
            Metadata for every job written.

        Notes
        -----
        - Produces N separate colabfold_batch calls (one per FASTA).
        - Best for: fine-grained monitoring, partial re-runs, GPU parallelism across nodes.
        - For 36 proteins → 630 FASTA files and 630 colabfold_batch calls.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        iterator = self._make_iterator(mode, nmin, nmax, ids_a, ids_b)

        metas: List[dict] = []
        index_path = out_path / index_name

        with open(index_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["name", "mode", "idA", "idB", "countA", "countB",
                        "total_residues", "file_path"]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()

            for kind, entry in iterator:
                seq_list, name, idA, idB = self._build_seq_list_from_entry(
                    kind, entry, mode, counts_map
                )
                total_res = sum(len(s) * c for _, s, c in seq_list)
                seq_str = self._build_colabfold_seq_str(seq_list)

                save_path = out_path / f"{name}.fasta"
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(f">{name}\n{seq_str}\n")

                logger.debug(f"Written: {save_path.name} ({total_res} residues)")

                meta = {
                    "name": name,
                    "mode": mode,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                    "file_path": str(save_path),
                }
                metas.append(meta)
                w.writerow(meta)

        logger.info(f"[FASTA strategy] Written {len(metas)} FASTA files to: {out_path}")
        return metas


    # =============================================================================
    # STRATEGY B — Single CSV batch (one colabfold_batch call for all jobs)
    # =============================================================================

    def write_colabfold_csv(
        self,
        output_dir: str,
        *,
        mode: str = "intra_pairs",
        nmin: int = 2,
        nmax: int = 6,
        counts_map: Optional[Dict[str, int]] = None,
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None,
        csv_name: str = "colabfold_input.csv",
        index_name: str = "colabfold_index.csv",
    ) -> List[dict]:
        """
        Writes a single CSV file for colabfold_batch containing all jobs.

        ColabFold's CSV format:

            id,sequence
            ProtA__ProtB,SEQUENCEA:SEQUENCEB
            ProtC__ProtD,SEQUENCEC:SEQUENCED

        A single colabfold_batch call processes all rows:

            colabfold_batch input.csv outputs/

        ColabFold internally handles the MSA queue and can reuse MSA results
        across pairs sharing the same protein — significantly faster than
        630 individual calls for an interactome of 36 proteins.

        Parameters
        ----------
        output_dir : str
            Directory where the CSV and index will be saved.
        mode : str, optional
            Generation strategy: 'intra_pairs', 'inter_pairs', 'homomers', 'single'.
            Defaults to "intra_pairs".
        nmin : int, optional
            Minimum stoichiometry for homomers. Defaults to 2.
        nmax : int, optional
            Maximum stoichiometry for homomers. Defaults to 6.
        counts_map : dict, optional
            Custom stoichiometry override per protein ID. Defaults to None.
        ids_a : list, optional
            Subset of Proteome A IDs to process.
        ids_b : list, optional
            Subset of Proteome B IDs to process.
        csv_name : str, optional
            Filename for the ColabFold input CSV. Defaults to "colabfold_input.csv".
        index_name : str, optional
            Filename for the metadata index CSV. Defaults to "colabfold_index.csv".

        Returns
        -------
        List[dict]
            Metadata for every job in the CSV.

        Notes
        -----
        - Produces a single colabfold_batch call.
        - Best for: large interactomes, MSA reuse, single-node runs.
        - ColabFold writes each job's output to a subfolder named after the 'id' column.
        - For 36 proteins → 1 CSV with 630 rows and 1 colabfold_batch call.

        Warnings
        --------
        - If the run is interrupted, you lose progress on all unfinished jobs.
        Use `write_colabfold_fastas` if you need fine-grained resume control.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        iterator = self._make_iterator(mode, nmin, nmax, ids_a, ids_b)

        metas: List[dict] = []
        cf_csv_path = out_path / csv_name
        index_path = out_path / index_name

        # Write ColabFold input CSV (id, sequence)
        with open(cf_csv_path, "w", newline="", encoding="utf-8") as cf_fh, \
            open(index_path, "w", newline="", encoding="utf-8") as idx_fh:

            cf_writer = csv.writer(cf_fh)
            cf_writer.writerow(["id", "sequence"])

            idx_fieldnames = ["name", "mode", "idA", "idB", "countA", "countB",
                            "total_residues"]
            idx_writer = csv.DictWriter(idx_fh, fieldnames=idx_fieldnames)
            idx_writer.writeheader()

            for kind, entry in iterator:
                seq_list, name, idA, idB = self._build_seq_list_from_entry(
                    kind, entry, mode, counts_map
                )
                total_res = sum(len(s) * c for _, s, c in seq_list)
                seq_str = self._build_colabfold_seq_str(seq_list)

                cf_writer.writerow([name, seq_str])

                meta = {
                    "name": name,
                    "mode": mode,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                }
                metas.append(meta)
                idx_writer.writerow(meta)

        logger.info(
            f"[CSV strategy] Written {len(metas)} jobs to: {cf_csv_path}\n"
            f"  → Run with: colabfold_batch {cf_csv_path} <output_dir>"
        )
        return metas
    
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


        self._available_modes = ["af3", "boltz2","colabfold"]

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

        elif self.mode == "colabfold":
            self.inputs = list(self.input_dir.glob("*.fasta"))

        else:
            self.inputs = []

        # Detect Outputs. Assuming outputs are directories inside the output path
        self.outputs = [p for p in self.output_dir.glob("*") if p.is_dir()]

        self.parse_job_dictionary: Dict[str, Callable] = {
            "af3": self._parse_af3_job,
            "boltz2": self._parse_boltz2_job,
        }
        
        # Determine initial status
        if self.mode == "colabfold":
            self.status = self.check_colabfold_run()
        else:
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
    
    @staticmethod
    def _build_colabfold_command(
        colabfold_bin: str,
        num_recycle: int,
        num_models: int,
        model_order: str,
        amber: bool,
        templates: bool,
        use_gpu_relax: bool,
        random_seed: int,
        extra_flags: List[str],
    ) -> List[str]:
        """Builds the base colabfold_batch command (without input/output args)."""
        cmd = [
            colabfold_bin,
            "--num-recycle", str(num_recycle),
            "--num-models", str(num_models),
            "--model-order", model_order,
            "--random-seed", str(random_seed),
        ]
        if amber:
            cmd.append("--amber")
        if templates:
            cmd.append("--templates")
        if use_gpu_relax:
            cmd.append("--use-gpu-relax")
        cmd.extend(extra_flags)
        return cmd

    @staticmethod
    def _run_single_colabfold_job(
        name: str,
        fasta_path: Path,
        output_dir: Path,
        base_cmd: List[str],
        dry_run: bool = False,
    ) -> dict:
        """Runs colabfold_batch for a single FASTA and captures result."""
        output_dir.mkdir(parents=True, exist_ok=True)
        full_cmd = base_cmd + [str(fasta_path), str(output_dir)]

        if dry_run:
            logger.info(f"[DRY RUN] {name}: {' '.join(full_cmd)}")
            return {"name": name, "status": "DRY_RUN", "returncode": 0, "elapsed_s": 0.0}

        log_file = output_dir / "colabfold.log"
        t0 = time.time()

        try:
            with open(log_file, "w") as log_fh:
                proc = subprocess.run(
                    full_cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = round(time.time() - t0, 1)
            status = "COMPLETED" if proc.returncode == 0 else "FAILED"

            if proc.returncode == 0:
                logger.info(f"✓ {name} completed in {elapsed}s")
            else:
                logger.error(f"✗ {name} failed (rc={proc.returncode}). Log: {log_file}")

            return {"name": name, "status": status,
                    "returncode": proc.returncode, "elapsed_s": elapsed}

        except FileNotFoundError:
            logger.error(
                "colabfold_batch not found. Pass the full path via `colabfold_bin` "
                "or add localcolabfold to PATH."
            )
            raise


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

    def run_colabfold_fastas(
    self,
    *,
    colabfold_bin: str = "colabfold_batch",
    num_recycle: int = 3,
    num_models: int = 5,
    model_order: str = "1,2,3,4,5",
    amber: bool = True,
    templates: bool = True,
    use_gpu_relax: bool = True,
    random_seed: int = 0,
    extra_flags: Optional[List[str]] = None,
    max_workers: int = 1,
    dry_run: bool = False,
) -> List[dict]:
        """
        Launches one colabfold_batch call per pending FASTA in input_dir.

        Reads pending jobs from self.status (jobs not yet COMPLETED) and runs
        each one sequentially or in parallel via ThreadPoolExecutor.

        Parameters
        ----------
        colabfold_bin : str, optional
            Path or name of the colabfold_batch binary. Defaults to "colabfold_batch".
            If localcolabfold is not on PATH, pass the full absolute path.
        num_recycle : int, optional
            Number of recycling iterations. Defaults to 3.
        num_models : int, optional
            Number of models to generate per job. Defaults to 5.
        model_order : str, optional
            Comma-separated model indices. Defaults to "1,2,3,4,5".
        amber : bool, optional
            Apply AMBER relaxation. Defaults to True.
        templates : bool, optional
            Use structural templates. Defaults to True.
        use_gpu_relax : bool, optional
            Run relaxation on GPU. Defaults to True.
        random_seed : int, optional
            Reproducibility seed. Defaults to 0.
        extra_flags : List[str], optional
            Additional colabfold_batch flags. Defaults to None.
        max_workers : int, optional
            Parallel jobs. Keep at 1 for single-GPU setups. Defaults to 1.
        dry_run : bool, optional
            Print commands without executing. Defaults to False.

        Returns
        -------
        List[dict]
            One entry per job: {'name', 'status', 'returncode', 'elapsed_s'}.
        """
        self.status = self.check_colabfold_run()
        pending = self.status.loc[self.status["status"] != "COMPLETED", "PPI"].tolist()

        if not pending:
            logger.info("All jobs already COMPLETED. Nothing to run.")
            return []

        logger.info(f"[FASTA strategy] Launching {len(pending)} pending jobs...")

        jobs = []
        for ppi_name in pending:
            fasta_path = self.input_dir / f"{ppi_name}.fasta"
            if not fasta_path.exists():
                logger.warning(f"FASTA not found, skipping: {fasta_path}")
                continue
            jobs.append((ppi_name, fasta_path, self.output_dir / ppi_name))

        base_cmd = self._build_colabfold_command(
            colabfold_bin, num_recycle, num_models, model_order,
            amber, templates, use_gpu_relax, random_seed, extra_flags or []
        )

        results = []
        if max_workers == 1:
            for name, fasta, out_dir in jobs:
                results.append(self._run_single_colabfold_job(name, fasta, out_dir, base_cmd, dry_run))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_colabfold_job, name, fasta, out_dir, base_cmd, dry_run): name
                    for name, fasta, out_dir in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        n_ok = sum(1 for r in results if r["returncode"] == 0)
        logger.info(f"[FASTA strategy] Done: {n_ok}/{len(results)} jobs succeeded.")
        return results


    def run_colabfold_csv(
        self,
        csv_path: str,
        output_dir: str,
        *,
        colabfold_bin: str = "colabfold_batch",
        num_recycle: int = 3,
        num_models: int = 5,
        model_order: str = "1,2,3,4,5",
        amber: bool = True,
        templates: bool = True,
        use_gpu_relax: bool = True,
        random_seed: int = 0,
        extra_flags: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Launches a single colabfold_batch call using a pre-built CSV input file.

        This is the recommended approach for large interactomes. ColabFold
        processes all rows in the CSV sequentially, potentially reusing MSA
        results across pairs that share the same protein sequence.

        Parameters
        ----------
        csv_path : str
            Path to the ColabFold input CSV (generated by write_colabfold_csv).
        output_dir : str
            Directory where ColabFold will write results. Each job gets a
            subfolder named after its 'id' column value.
        colabfold_bin : str, optional
            Path or name of the colabfold_batch binary. Defaults to "colabfold_batch".
        num_recycle : int, optional
            Number of recycling iterations. Defaults to 3.
        num_models : int, optional
            Number of models to generate per job. Defaults to 5.
        model_order : str, optional
            Comma-separated model indices. Defaults to "1,2,3,4,5".
        amber : bool, optional
            Apply AMBER relaxation. Defaults to True.
        templates : bool, optional
            Use structural templates. Defaults to True.
        use_gpu_relax : bool, optional
            Run relaxation on GPU. Defaults to True.
        random_seed : int, optional
            Reproducibility seed. Defaults to 0.
        extra_flags : List[str], optional
            Additional colabfold_batch flags. Defaults to None.
        dry_run : bool, optional
            Print command without executing. Defaults to False.

        Returns
        -------
        dict
            Keys: 'status', 'returncode', 'elapsed_s', 'log_path'.

        Warnings
        --------
        - No per-job resume: if the run is interrupted, completed jobs within
        the CSV are not re-run (ColabFold skips existing outputs), but you
        cannot easily restart from the middle.
        - For fine-grained control, use run_colabfold_fastas instead.
        """
        csv_path = Path(csv_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"ColabFold input CSV not found: {csv_path}")

        base_cmd = self._build_colabfold_command(
            colabfold_bin, num_recycle, num_models, model_order,
            amber, templates, use_gpu_relax, random_seed, extra_flags or []
        )
        full_cmd = base_cmd + [str(csv_path), str(out_path)]

        if dry_run:
            logger.info(f"[DRY RUN] CSV batch: {' '.join(full_cmd)}")
            return {"status": "DRY_RUN", "returncode": 0, "elapsed_s": 0.0, "log_path": None}

        log_path = out_path / "colabfold_batch.log"
        logger.info(f"[CSV strategy] Launching batch job. Log: {log_path}")
        logger.debug(f"Command: {' '.join(full_cmd)}")

        t0 = time.time()
        try:
            with open(log_path, "w") as log_fh:
                proc = subprocess.run(
                    full_cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = round(time.time() - t0, 1)
            status = "COMPLETED" if proc.returncode == 0 else "FAILED"

            if proc.returncode == 0:
                logger.info(f"✓ Batch completed in {elapsed}s")
            else:
                logger.error(f"✗ Batch failed (rc={proc.returncode}). Log: {log_path}")

            return {"status": status, "returncode": proc.returncode,
                    "elapsed_s": elapsed, "log_path": str(log_path)}

        except FileNotFoundError:
            logger.error(
                "colabfold_batch not found. Pass the full path via `colabfold_bin`."
            )
            raise


    def check_colabfold_run(self, expected_models: int = 5) -> pd.DataFrame:
        """
        Checks execution status for ColabFold jobs (works for both strategies).

        Scans output_dir for subdirectories and counts ranked PDB files.
        Compatible with both FASTA-per-job and CSV batch outputs since
        ColabFold always writes to subfolder-per-job regardless of input format.

        Parameters
        ----------
        expected_models : int, optional
            Number of ranked PDB models expected per job. Defaults to 5.

        Returns
        -------
        pd.DataFrame
            Columns: PPI, num_models, status (COMPLETED / RUNNING / FAILED).
        """
        rows = []

        # Determine job names: from FASTA inputs if available, else from output dirs
        if self.inputs:
            job_names = [p.stem for p in self.inputs]
        else:
            job_names = [p.name for p in self.output_dir.glob("*") if p.is_dir()]

        if not job_names:
            logger.warning("No jobs found to check.")
            return pd.DataFrame(columns=["PPI", "num_models", "status"])

        for ppi_name in job_names:
            job_out = self.output_dir / ppi_name

            if job_out.exists():
                # Primary pattern: rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb
                pdb_count = len(list(job_out.glob("*rank_*_model_*.pdb")))
                # Fallback: relaxed models
                if pdb_count == 0:
                    pdb_count = len(list(job_out.glob("*relaxed*.pdb")))
            else:
                pdb_count = 0

            if pdb_count >= expected_models:
                status = "COMPLETED"
            elif pdb_count > 0:
                status = "RUNNING"
            else:
                status = "FAILED"

            rows.append({"PPI": ppi_name, "num_models": pdb_count, "status": status})

        df = pd.DataFrame(rows)
        custom_order = ["FAILED", "RUNNING", "PENDING", "COMPLETED"]
        df["status"] = pd.Categorical(df["status"], categories=custom_order, ordered=True)
        df.sort_values("status", inplace=True)

        logger.info(
            f"ColabFold status — "
            f"COMPLETED: {(df.status == 'COMPLETED').sum()} | "
            f"RUNNING: {(df.status == 'RUNNING').sum()} | "
            f"FAILED: {(df.status == 'FAILED').sum()}"
        )
        return df

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
    _SUPPORTED_ENGINES = {"af3", "boltz", "colabfold"}

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
    def cluster_pae(pae_submatrix: np.ndarray, threshold:float = 15.0, eps:float = 10.0, min_samples: int = 5)-> Tuple[np.ndarray, np.ndarray]:
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
        ppi_id = dir_name.replace(prefix, "")
        
        # Handle naming: idA__idB (pairs), idA__copies (homomers), or idA (monomers)
        parts = ppi_id.split("__")
        orf_a = parts[0]
        orf_b = parts[1] if len(parts) > 1 else ""

        # Assumes "...model_1..." format.
        base_name = path_obj.stem # removes .cif or .pdb
        
        model_number = int(base_name.split("model_")[-1].split("_")[0])
        # model_number = int(base_name.split("_")[-1].replace("model_",""))

        # Load Full Data
        if model_type.lower() == "af3":
            full_data = process_full_data_af3(str(path_obj))
            # full_data = process_full_data_af3(model_file)
            # molecule_model = MoleculeModel.from_af3(model_file)
        elif model_type.lower() == "boltz":
            full_data = process_full_data_boltz(str(path_obj))
        elif model_type.lower() == "colabfold":
            full_data = process_full_data_colabfold(str(path_obj))
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'AF3', 'Boltz' or 'ColabFold'.")

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
                "msa_depth": full_data.get("msa_depth", np.nan),
                "msa_coverage": full_data.get("msa_coverage", np.nan),
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
        import tqdm
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
    
    def get_confidence_tiers(
        self, 
        ipsae_threshold: float = 0.5, 
        pdockq2_threshold: float = 0.23, 
        msa_threshold: int = 20
    ) -> pd.DataFrame:
        """
        Categorizes interactome results into confidence Tiers based on 
        structural, physical, and evolutionary metrics.

        Parameters
        ----------
        ipsae_threshold : float, default=0.5
            Minimum ipSAE for confidence.
        pdockq2_threshold : float, default=0.23
            Minimum pDockQ2 for physical plausibility.
        msa_threshold : int, default=20
            Minimum MSA depth for evolutionary support.

        Returns
        -------
        pd.DataFrame
            The interactome data with an added 'Tier' column.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        # Handle variations in column names
        ipsae_col = "ipSAE_AB" if "ipSAE_AB" in df.columns else "ipSAE"
        pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
        msa_col = "msa_depth"

        def classify(row):
            if ipsae_col not in row or pdockq2_col not in row:
                return "Unknown"
            
            msa_val = row.get(msa_col, 0)
            ipsae_val = row[ipsae_col]
            pdockq2_val = row[pdockq2_col]

            # Tier 1: High structure, High physics, High MSA
            if ipsae_val > ipsae_threshold and pdockq2_val > pdockq2_threshold and msa_val > msa_threshold:
                return "Tier 1 (High Confidence)"
            
            # Tier 2: High structure, High physics, Low MSA (Potential Novel/Specific)
            if ipsae_val > ipsae_threshold and pdockq2_val > pdockq2_threshold and msa_val <= msa_threshold:
                return "Tier 2 (Specific/Novel)"

            # Tier 3: High structure, Low physics
            if ipsae_val > ipsae_threshold and pdockq2_val <= pdockq2_threshold:
                return "Tier 3 (Weak/Dynamic)"

            return "Low Confidence"

        df["Tier"] = df.apply(classify, axis=1)
        logger.info(f"Tier Classification: \n{df['Tier'].value_counts()}")
        return df

    def plot_confidence_landscape(self, output_path: Optional[Union[str, Path]] = None):
        """
        Generates a scatter plot of the interactome confidence landscape.
        X-axis: pDockQ2, Y-axis: ipSAE_d0_dom, Size: msa_depth, Color: pLDDT_B (AlphaFold colors).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded.")

        df = self._interactome_data.copy()
        
        # Priority for the y-axis: ipSAE_d0_dom_AB -> ipSAE_d0dom_AB -> ipSAE_AB
        y_col = None
        for col in ["ipSAE_d0_dom_AB", "ipSAE_d0dom_AB", "ipSAE_AB", "ipSAE"]:
            if col in df.columns:
                y_col = col
                break
        
        pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
        plddt_col = "pLDDT_mean" if "pLDDT_mean" in df.columns else None
        msa_col = "msa_depth" if "msa_depth" in df.columns else None

        if not y_col or pdockq2_col not in df.columns:
            logger.error(f"Required columns for plotting not found. Using {y_col} and {pdockq2_col}")
            return

        # --- AlphaFold Color Scheme ---
        # >90: #0053D6, 70-90: #65CBF3, 50-70: #FFDB13, <50: #FF7D45
        af_colors = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]
        cmap = ListedColormap(af_colors)
        norm = BoundaryNorm([0, 50, 70, 90, 100], cmap.N)

        plt.figure(figsize=(10, 8))
        
        # Scaling bubbles: Use square root to normalize size differences and scale down
        if msa_col and msa_col in df.columns:
            # Scale factor 8 and base 15 makes them visible but small
            sizes = np.sqrt(df[msa_col].fillna(0)) * 8 + 15
        else:
            sizes = 40

        # Jitter: Add a tiny bit of noise to prevent overlap of identical results
        x_values = df[pdockq2_col] + np.random.normal(0, 0.003, size=len(df))
        y_values = df[y_col] + np.random.normal(0, 0.003, size=len(df))

        scatter = plt.scatter(
            x_values, 
            y_values, 
            s=sizes, 
            c=df[plddt_col] if plddt_col else "gray", 
            cmap=cmap,
            norm=norm,
            alpha=0.75, 
            edgecolors="black",
            linewidths=0.5
        )
        
        # Reference lines (updated thresholds)
        plt.axhline(0.4, color="gray", linestyle="--", alpha=0.4, label="ipSAE_dom 0.4")
        plt.axvline(0.23, color="gray", linestyle="--", alpha=0.4, label="pDockQ2 0.23")
        
        plt.xlabel("Physical Plausibility (pDockQ2)")
        plt.ylabel(f"Interface Confidence ({y_col})")
        plt.title("Interactome Confidence Landscape (Adenovirus HAdV-5)")
        
        if plddt_col:
            cbar = plt.colorbar(scatter, ticks=[25, 60, 80, 95])
            cbar.set_ticklabels(["<50 (Very Low)", "50-70 (Low)", "70-90 (High)", ">90 (Very High)"])
            cbar.set_label("Mean pLDDT (Global Model Confidence)")
        
        plt.legend(loc="upper left", fontsize=9, frameon=True)
        plt.grid(True, linestyle=":", alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confidence landscape plot saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def plot_interactive_landscape(self, output_path: Optional[Union[str, Path]] = None):
        """
        Generates an interactive HTML scatter plot of the confidence landscape using Plotly.
        Includes hover info for PPI names and metrics.
        """
        try:
            import plotly.express as px
        except ImportError:
            logger.error("Plotly is required for interactive plots. Install it with 'pip install plotly'.")
            return

        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded.")

        df = self._interactome_data.copy()
        
        # Identify metrics
        y_col = None
        for col in ["ipSAE_d0_dom_AB", "ipSAE_d0dom_AB", "ipSAE_AB", "ipSAE"]:
            if col in df.columns:
                y_col = col
                break
        
        pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
        plddt_col = "pLDDT_mean" if "pLDDT_mean" in df.columns else None
        msa_col = "msa_depth" if "msa_depth" in df.columns else None

        if not y_col or pdockq2_col not in df.columns:
            logger.error("Required columns for interactive plotting not found.")
            return

        # Categorize pLDDT for AlphaFold coloring in Plotly
        if plddt_col:
            df["Confidence_Level"] = pd.cut(
                df[plddt_col], 
                bins=[0, 50, 70, 90, 100], 
                labels=["Very Low (<50)", "Low (50-70)", "High (70-90)", "Very High (>90)"]
            )
        
        # Bubble sizing (normalized for Plotly)
        size_col = "Size"
        if msa_col:
            df[size_col] = np.sqrt(df[msa_col].fillna(0)) + 5
        else:
            df[size_col] = 10

        # Create Plotly figure
        fig = px.scatter(
            df,
            x=pdockq2_col,
            y=y_col,
            size=size_col,
            color="Confidence_Level" if plddt_col else None,
            hover_name="PPI",
            hover_data={
                "ORF_A": True,
                "ORF_B": True,
                msa_col: True,
                y_col: ":.3f",
                pdockq2_col: ":.3f",
                size_col: False,
                "Confidence_Level": False
            },
            color_discrete_map={
                "Very High (>90)": "#0053D6",
                "High (70-90)": "#65CBF3",
                "Low (50-70)": "#FFDB13",
                "Very Low (<50)": "#FF7D45"
            },
            title=f"Interactive Confidence Landscape (Adenovirus HAdV-5)<br><sup>Bubble size = sqrt(MSA depth)</sup>",
            labels={y_col: "Interface Confidence (ipSAE_dom)", pdockq2_col: "Physical Plausibility (pDockQ2)"},
            template="plotly_white"
        )

        # Add reference lines
        fig.add_hline(y=0.4, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.23, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            legend_title_text="Global pLDDT",
            hoverlabel=dict(bgcolor="white", font_size=12)
        )

        if output_path:
            out_file = str(Path(output_path).with_suffix(".html"))
            fig.write_html(out_file)
            logger.info(f"Interactive landscape saved to: {out_file}")
        else:
            fig.show()

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
    
    def run_full_pipeline(self, ipsae_filter: Optional[float] = None, **kwargs):
        """
        Executes the complete analysis pipeline.

        Parameters
        ----------
        ipsae_filter : float, optional
            If provided, logs how many PPIs are above this confidence threshold.
            Structural analysis will proceed for ALL candidates regardless.
        **kwargs
            Arguments passed to downstream methods:
            - cluster_ratio_threshold (float, default=7.0)
            - min_peptide_len (int, default=5)
        """
        if self._cluster_data is None:
            logger.warning("Cannot run pipeline: Cluster data is missing.")
            return
        
        logger.info("Starting peptide-protein analysis pipeline...")
        
        # Default cluster ratio for peptides set to 7.0 as requested
        kwargs.setdefault('cluster_ratio_threshold', 7.0)

        # Log filtering info if requested, but DO NOT filter the structural pipeline
        if ipsae_filter is not None and self._interactome_data is not None:
            df = self._interactome_data
            ipsae_col = None
            for col in ["ipSAE_d0dom_AB", "ipSAE_d0_dom_AB", "ipSAE_AB", "ipSAE"]:
                if col in df.columns:
                    ipsae_col = col
                    break

            if ipsae_col:
                high_conf_ppis = df[df[ipsae_col] > ipsae_filter]["PPI"].unique()
                logger.info(f"Report: {len(high_conf_ppis)} PPIs are above {ipsae_col} > {ipsae_filter}.")
        
        # Run structural analysis for ALL candidates
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
    
    
             
    def _create_binder_alignments(self, 
                                  model_to_align: str, 
                                  reference_model: Union[str, Molecule])-> Molecule:
        """
        Aligns a target protein model to a reference structure based on the Binder (Chain A).

        This method performs a structural alignment using the following steps:
        1. Loads the target model.
        2. Filters the target's Chain A to keep only residues present in the reference 
           (ensuring a valid index-based alignment).
        3. Preserves Chain B (Peptide) entirely.
        4. Aligns the filtered target to the reference.

        Parameters
        ----------
        model_to_align : str
            Path to the PDB file of the model to be aligned.
        reference_model : str or Molecule
            The reference structure used as the fixed point. Can be a file path 
            (str) or a loaded Molecule object.

        Returns
        -------
        Molecule
            The aligned Molecule object (containing filtered Chain A + Chain B).
        """
        # 1. Resolve Reference Model (Path vs Object)
        if isinstance(reference_model, str):
            reference_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            reference_mol = reference_model
        else:
            raise ValueError("reference_model should be a path (str) or a Molecule instance")
        
        # 2. Get Reference Residues
        residues_in_reference_chain = reference_mol.resid.astype(str)
        reference_resid_str = ' '.join(residues_in_reference_chain)
        
        # 3. Load Target Model
        tmp_mol = Molecule(model_to_align)
        
        # 4. Filter Target
        # Keep Chain A (only matching reference residues) OR Chain B (Peptide)
        tmp_mol.filter(f"(chain A and resid {reference_resid_str}) or (chain B)")
        
        # 5. Align
        # mode="index" requires strict atom-to-atom correspondence.
        tmp_mol.align(f"chain A",
                    refmol=reference_mol,
                    refsel=f"chain A",
                    mode="index"
                    )
        return tmp_mol
    
    def _get_reference_structure_for_binder(self, all_structs: List[str]) -> Molecule:
        """
        Selects the best structural model to serve as a reference for alignment.

        The selection relies on the pLDDT score.
        It selects the model with the highest median pLDDT for Chain A (Binder)
        and trims it to keep only the high-confidence residues (pLDDT > 70).

        Parameters
        ----------
        all_structs : List[str]
            A list of file paths to the PDB models.

        Returns
        -------
        Molecule
            A MoleculeKit object of the reference binder, containing only 
            Chain A residues with high structural confidence.
        """
        # Load all molecules
        mol_list = [ Molecule(i) for i in all_structs]
        
        plddt_scores = []
        for mol in mol_list:
            # Select Chain A (Binder)
            # numpy boolean masking on the 'chain' attribute
            mask_a = mol.chain == "A"
            if np.any(mask_a):
                plddt_chain_A = mol.beta[mask_a]
                score = np.median(plddt_chain_A)
            else:
                score = 0.0 # Fallback if chain A is missing
            plddt_scores.append(score)
        
        plddt_scores = np.array(plddt_scores)

        # Select the best index (Highest median pLDDT) 
        best_global_idx = np.argmax(plddt_scores)

        # Define the reference molecule, just one by binder
        reference_mol = mol_list[best_global_idx].copy()
        
        # 1. Filter: Keep only Chain A
        reference_mol.filter("chain A")

        # 2. Filter: Keep only high confidence residues (pLDDT > 70)
        # Get Residue IDs of CA atoms where Beta > 70
        # We use CA atoms as representative for the whole residue to avoid duplicates
        mask_ca = reference_mol.name == "CA"
        
        # Two-step masking: 
        # 1. Get betas for CAs. 
        # 2. Check which are > 70.
        # 3. Apply that boolean mask to the resids of CAs.
        high_conf_mask = reference_mol.beta[mask_ca] > 70
        reference_resids = reference_mol.resid[mask_ca][high_conf_mask]
        # reference_resids = reference_mol.resid[reference_mol.name == "CA"][reference_mol.beta[reference_mol.name == "CA"]>70]
        
        # Create selection string
        reference_resids = reference_resids.astype(str)
        reference_resid_str = ' '.join(reference_resids)

        # Apply final filter
        if reference_resid_str:
            reference_mol.filter(f"resid {reference_resid_str}")
        else:
            logger.warning("No high-confidence residues (pLDDT > 70) found in best model. Returning full Chain A.")
        
        return reference_mol

    def analyze_peptide_proteins_pairs(self, **kwargs):
        """
        Executes the structural analysis pipeline for peptide-protein interactions.
        ...
        """
        # 1. Setup Directories
        output_path = f"{self.output_path}/prot_peptide"
        os.makedirs(output_path, exist_ok=True)

        # Extract filtering arguments intended for _get_candidate_clusters
        filter_args = {
            'cluster_ratio_threshold': kwargs.pop('cluster_ratio_threshold', 5.0),
            'min_peptide_len': kwargs.pop('min_peptide_len', 5)
        }

        # 2. Identify Candidates
        self._candidate_clusters = self._get_candidate_clusters(**filter_args)

        if self._candidate_clusters is None or self._candidate_clusters.empty:
            logger.warning("No candidate peptide-protein clusters found. Skipping analysis.")
            return
        
        # Create folder structure for each unique binder
        for binder in self._candidate_clusters.Binder_name.unique():
            os.makedirs(f"{output_path}/{binder}/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/filtered/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/aligned/", exist_ok=True)
        
        # 3. Prepare Working DataFrame
        cols = ["PPI", "model_num", "x_len", "y_len", 
                "Binder_name", "Binder_chain", "Binder_start", "Binder_end",
                "Peptide_name", "Peptide_chain", "Peptide_start", "Peptide_end", "path"]
        
        df = self._candidate_clusters.loc[:,cols].copy()
        
        # Generate Unique IDs (PPI + Model + ClusterID)
        df.loc[: , "PPI"] = df.PPI + "_" +  self._candidate_clusters.model_num.astype(str) + "_" + self._candidate_clusters.cluster_id.astype(str)
        
        # 4. Filtering Loop (Curate Structures)
        filtered_names = []
        for idx, row in df.iterrows():
            output_name = f"{output_path}/{row.Binder_name}/filtered/{row.PPI}.pdb"
            filtered_names.append(output_name)
            
            if not os.path.exists(output_name):
                mol = self._curate_protein_peptide_models(row)
                mol.write(output_name)
            else:
                logger.info(f"Skipping {output_name}, already filtered...")
        df.loc[:, "filtered_path"] = filtered_names
        
        # 5. Analysis Loop per Binder
        binder_df = pd.DataFrame()
        
        for binder in self._candidate_clusters.Binder_name.unique():

            ppi_data = df.loc[self._candidate_clusters.Binder_name == binder,:].copy()
            all_structs = ppi_data.filtered_path.values
            
            if len(all_structs) == 0:
                logger.warning(f"No valid models for binder {binder} after filtering. Skipping...")
                continue

            # A. Get/Create Reference Structure
            reference_output_name = f"{output_path}/{binder}/reference_{binder}.pdb"
            if not os.path.exists(reference_output_name):
                reference_molecule = self._get_reference_structure_for_binder(all_structs)
                reference_molecule.write(reference_output_name)
            else:
                reference_molecule = Molecule(reference_output_name)
                logger.info(f"Skipping reference generation for {binder}, exists...")

            # B. Alignment Loop
            aligned_models = []
            for tmp_mol_name in all_structs:

                # Simple string replacement for path (assumes standard folder structure)
                tmp_mol_name_aligned = tmp_mol_name.replace("filtered", "aligned")
                
                if not os.path.exists(tmp_mol_name_aligned):
                    tmp_mol = self._create_binder_alignments(tmp_mol_name, reference_molecule)
                    tmp_mol.write(tmp_mol_name_aligned)
                else:
                    logger.debug(f"Skipping alignment for {tmp_mol_name_aligned}...")
                
                aligned_models.append(tmp_mol_name_aligned)

            # C. Clustering (DBSCAN)
            tmp_df, cluster_info = self.cluster_protein_peptides(aligned_models, reference_output_name, **kwargs)
            
            if tmp_df.empty or len(cluster_info.get("cluster_labels", [])) == 0:
                logger.warning(f"No spatial clusters found for {binder}. Skipping ChimeraX generation.")
                continue

            tmp_df.insert(0, "Binder", binder)
            binder_df = pd.concat([binder_df, tmp_df], ignore_index=True)

            # D. Visualization Preparation (ChimeraX)
            # Ensure lengths match before assignment
            labels = cluster_info.get("cluster_labels")
            centers = cluster_info.get("peptide_centers")

            if len(labels) == len(ppi_data):
                ppi_data.loc[:, "Cluster_info"] = labels
                ppi_data.loc[:, "Center_X"] = centers[:,0]
                ppi_data.loc[:, "Center_Y"] = centers[:,1]
                ppi_data.loc[:, "Center_Z"] = centers[:,2] 
                
                self._create_chimera_session(ppi_data, reference_output_name, tmp_df)
            else:
                logger.error(f"Shape mismatch for {binder}: labels({len(labels)}) != data({len(ppi_data)}). Skipping.")
        
        # 6. Save Final Summary
        binder_df.to_csv(f"{self.output_path}/peptide_binder_info.csv", index=False)

    def _create_chimera_session(self,
                                ppi_data: pd.DataFrame, 
                                ref_model: str,
                                cluster_info: pd.DataFrame):
        """
        Generates and executes a ChimeraX script (.cxc) to visualize the analysis.

        This method writes a set of ChimeraX commands to:
        1. Load the reference binder structure.
        2. Color the binder's interface residues according to the cluster they interact with.
        3. Load and align all peptide structures, colored by their cluster ID.
        4. Represent cluster centroids as spheres.
        5. Save the session as a .cxs file for easy reopening.

        Parameters
        ----------
        ppi_data : pd.DataFrame
            DataFrame containing details of the peptide models (paths, cluster IDs, etc.).
        ref_model : str
            Path to the reference PDB file of the binder.
        cluster_info : pd.DataFrame
            Summary DataFrame containing cluster labels, centroids, and interacting residues.
        """
        import subprocess

        binder = ppi_data.Binder_name.values[0]

        # Use Pathlib for robust path handling
        base_dir = self.output_path / "prot_peptide" / binder
        script_path = base_dir / f"{binder}_peptide_binding.cxc"
        session_path = base_dir / f"{binder}_peptide_binding.cxs" 
        
        available_colors = ["cyan", "yellow", "magenta", "orange", "cornflower blue"]
        available_colors_ref = ["light coral", "medium slate blue", "orange", "green", "red", "yellow"]
        
        with open(script_path, "w") as f:
            ## Global settings
            f.write("graphics silhouettes true\n") 
            f.write("lighting soft\n") 
            f.write("set bg white\n") 

            ## Load reference   
            f.write(f"\n# --- {binder} REFERENCE ---\n")
            f.write(f"open \"{ref_model}\"\n")
            f.write(f"rename #1 {binder}_ref\n")

            ## Color ref residues based on cluster interactions
            for idx, cluster_data in cluster_info.iterrows():
                cluster_id = int(cluster_data["Cluster_label"])
                if cluster_id == -1:
                    continue # Skip noise
                
                # Join residues for selection
                tmp_sel_str = ",".join(map(str, cluster_data["Residues"]))
                
                color_index = cluster_id % len(available_colors_ref)
                color_str = available_colors_ref[color_index]
                
                f.write(f"color #1:{tmp_sel_str} {color_str}\n") 

                ## Draw centroid sphere for the cluster
                super_id = f"#5.{cluster_id + 1}"
                global_center_str = f"{cluster_data['Center_X']},{cluster_data['Center_Y']},{cluster_data['Center_Z']}"
                f.write(f"shape sphere name Centroid_{cluster_id+1}_Mean radius 3 center {global_center_str} color {color_str} model {super_id} \n")
             
            ## Load peptide_proteins and rename
            ppi_data["aligned_path"] = ppi_data["filtered_path"].str.replace("filtered", "aligned")

            for tmp_cluster in ppi_data["Cluster_info"].unique():
                tmp_cluster = int(tmp_cluster)
                current_sub_id = tmp_cluster + 2
                
                if tmp_cluster == -1:
                    color_str = "silver"
                    group_name = "Unclassified"
                
                else:
                    color_idx = tmp_cluster % len(available_colors) 
                    color_str = available_colors[color_idx]
                    group_name = f"Cluster_{tmp_cluster + 1}"

                models_in_cluster = ppi_data.loc[ppi_data["Cluster_info"] == tmp_cluster, :]
                
                for i, (idx, row) in enumerate(models_in_cluster.iterrows()):
                    # Model ID hierarchy: #3.ClusterID.ModelIndex
                    pep_id = f"#3.{current_sub_id}.{i + 1}" 
                    cen_id = f"#4.{current_sub_id}.{i + 1}" 
                    
                    # Open peptide 
                    f.write(f"open \"{row['aligned_path']}\" id {pep_id}\n")
                    
                    if tmp_cluster == -1:
                        model_name = f"{row['PPI']}_unclassified"    
                    else:
                        model_name = f"{row['PPI']}_c{current_sub_id}"
                    
                    f.write(f"rename {pep_id} {model_name}\n")
                    f.write(f"color {pep_id} {color_str}\n")

                    centroid_str = f"{row['Center_X']},{row['Center_Y']},{row['Center_Z']}"
                    f.write(f"shape sphere name {model_name} radius 1 center {centroid_str} color {color_str} model {cen_id}\n")
                f.write(f"rename #3.{current_sub_id} {group_name}\n")
                f.write(f"rename #4.{current_sub_id} {group_name}_Center_of_mass\n")   

            
            ## Final cleanup and save
            f.write(f"lighting depthCue false\n")
            f.write(f"rename #3 Peptides\n")
            f.write(f"hide #3/A cartoon\n") # Hide the Binder chain in the aligned peptide models
            f.write(f"hide atoms\n")
            f.write(f"rename #4 Peptide_centers\n")
            # f.write(f"rename #5 Centroids\n")
            f.write(f"save {session_path}\n")
            f.write(f"exit\n")

        # Execute ChimeraX using subprocess (safer than os.system)
        # os.system(f"chimerax --nogui {script_path}") ##--nogui 
        logger.info(f"Executing ChimeraX script: {script_path}")
        try:
            subprocess.run(["chimerax", "--nogui", str(script_path)], check=True)
        except FileNotFoundError:
            logger.error("ChimeraX executable not found in PATH. Script generated but not executed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ChimeraX execution failed: {e}")

    def cluster_protein_peptides(self, aligned_models: List[str], 
                                 reference_model: Union[str, Molecule], 
                                 **kwargs)-> tuple[pd.DataFrame, dict]:
        """
        Clusters peptide structures based on their spatial position relative to the binder.

        This method uses the DBSCAN algorithm to group peptides that bind to similar 
        regions on the reference protein surface. It calculates the centroid (Center of Mass) 
        of the peptide backbones (Chain B, CA atoms) and clusters these points.

        Parameters
        ----------
        aligned_models : List[str]
            List of file paths to the aligned PDB models.
        reference_model : str or Molecule
            The reference binder structure used to map the binding sites.
        **kwargs
            Arguments passed to the DBSCAN constructor (e.g., `eps=5`, `min_samples=3`).

        Returns
        -------
        pd.DataFrame
            Summary table with columns:
            - 'Cluster_label': The ID assigned by DBSCAN (-1 indicates noise).
            - 'Center_X/Y/Z': Geometric center of the cluster.
            - 'Residues': List of binder residues within 8Å of the cluster center.
        dict
            Dictionary containing raw 'cluster_labels' and 'peptide_centers' arrays.
        """
        
        # 1. Load Molecules
        mols = [Molecule(str(i)) for i in aligned_models]

        # 2. Resolve Reference
        if isinstance(reference_model, str):
            ref_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            ref_mol = reference_model
        else:
            raise ValueError("reference_model must be a path (str) or Molecule object")
        
        # 3. Get Reference Geometry
        xyz = ref_mol.get("coords", sel="protein")
        ref_resids = ref_mol.get("resid", sel="protein")
        
        # 4. Calculate Peptide Centroids (Chain B, Alpha Carbons)
        # We assume all models have a Chain B with CA atoms (enforced by previous steps)
        # mols_centroids = np.array([tmp_mol.get("coords", sel="chain B and name CA").mean(axis=0) for tmp_mol in mols])
        
        valid_centroids = []
        valid_indices = []
        for i, tmp_mol in enumerate(mols):
            coords = tmp_mol.get("coords", sel="chain B and name CA")
            if coords.size > 0:
                valid_centroids.append(coords.mean(axis=0))
                valid_indices.append(i)
            else:
                logger.warning(f"Model {aligned_models[i]} has no Chain B CA atoms. Skipping from clustering.")

        if not valid_centroids:
            return pd.DataFrame(), {"cluster_labels": np.array([]), "peptide_centers": np.array([])}

        mols_centroids = np.array(valid_centroids)

        # 5. Perform Clustering (DBSCAN)
        # kwargs allows passing eps, min_samples, etc.
        clustering = DBSCAN(**kwargs).fit(mols_centroids)
        
        # Adjust cluster labels for original input list
        full_labels = np.full(len(mols), -1, dtype=int)
        for idx, label in zip(valid_indices, clustering.labels_):
            full_labels[idx] = label

        # 6. Analyze Clusters
        cluster_labels = []
        cluster_centers = []
        all_nearby_residues = []

        unique_labels = np.unique(clustering.labels_)

        for cluster_label in unique_labels:
            cluster_labels.append(cluster_label)

            # Calculate geometric center of the cluster (mean of peptide centroids)
            mask = clustering.labels_ == cluster_label
            cluster_center = mols_centroids[mask].mean(axis=0)
            cluster_centers.append(cluster_center)

            # 7. Identify Binding Site Residues
            # Calculate distance from every protein atom to the cluster center
            tmp_centroid_distance = xyz - cluster_center
        
            # Euclidean norm (distance)
            tmp_euc = np.linalg.norm(tmp_centroid_distance, axis = 1)
            
            nearby_residues = np.unique(ref_resids[tmp_euc < 8])
            all_nearby_residues.append(nearby_residues)
        
        # 8. Format Output
        cluster_centers = np.array(cluster_centers)
        
        results_df = pd.DataFrame({
            "Cluster_label": cluster_labels,
            "Center_X": cluster_centers[:, 0],
            "Center_Y": cluster_centers[:, 1],
            "Center_Z": cluster_centers[:, 2],
            "Residues": all_nearby_residues
        })

        extra_info = {
            "cluster_labels": full_labels, 
            "peptide_centers": np.array([valid_centroids[valid_indices.index(i)] if i in valid_indices else [0,0,0] for i in range(len(mols))])
        }

        return results_df, extra_info

#     def calculate_network(self):
#         pass

#     def basic_plots(self):
#         pass

#     def protein_peptide_analysis(self):
#         pass

#     def cluster_analysis(self):
#         pass
