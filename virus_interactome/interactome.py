from enum import Enum
import os
import time
import subprocess
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import csv

from sklearn.cluster import DBSCAN
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path
from moleculekit.molecule import Molecule

from .utils import load_json, load_boltz_input, parse_msa_metrics, reorganize_colabfold_outputs
from .model import Engine, Model
from .metrics import (
    calculate_all_metrics,
    calculate_ipsae,
    calculate_LIS_family,
    calculate_pdockq,
    calculate_pdockq2,
)
from .plotting import plot_boxplots, plot_iptm_vs_ptm, plot_pae_clusters, plot_paes, plot_plddt
from .model import Engine

logger = logging.getLogger(__name__)


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
                For monomer runs (``mode='single'``), pass the value explicitly:
                ``expected_models=5`` (AF3) or ``expected_models=1`` (Boltz2).
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
            # Handle single-job dict (get_af3_input saves a dict, not a list)
            if isinstance(batch_jobs, dict):
                batch_jobs = [batch_jobs]

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

        df = pd.DataFrame(all_job_info, columns = ["PPI", "num_chain", "num_aa"])

        # Check Outputs
        num_models_list = []
        for job_name in df["PPI"]:
            job_output_dir = self.output_dir / job_name
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

            logger.info("Reorganizing ColabFold outputs...")
            reorganize_colabfold_outputs(out_path)

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


class InteractomeMode(Enum):
    MONOMER = "single"
    INTRA_PAIRS = "intra_pairs"
    INTER_PAIRS = "inter_pairs"
    HOMOMERS = "homomers"

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
    # _SUPPORTED_ENGINES = {"af3", "boltz", "colabfold"}

    def __init__(self, model_list: List[Union[str, Path]], engine: Engine | str = Engine.COLABFOLD, mode: InteractomeMode | str = InteractomeMode.INTRA_PAIRS):
        """
        Initializes the InteractomeProcessor.

        Args:
            model_list (List[Union[str, Path]]): A list of paths pointing to the 
                results to be processed. Can be strings or Path objects.
            engine (str, optional): The folding engine used. Case-insensitive. 
                Must be 'af3' or 'boltz'. Defaults to "af3".
            mode (str, optional): The mode of interaction analysis. Case-insensitive.
                Must be one of 'monomer', 'intra_pairs', 'inter_pairs', or 'homomers'.
                Defaults to 'inter_pairs'.

        Raises:
            ValueError: If the provided engine is not supported.
        """
        try:
            self._engine = Engine(engine.lower()) if isinstance(engine, str) else engine
        except ValueError:
            valid = ", ".join(e.value for e in Engine)
            raise ValueError(f"Engine should be one of: {valid}")
        self._mode = InteractomeMode(mode.lower()) if isinstance(mode, str) else mode

        # Sanitize Paths (Convert all to Path objects)
        self.model_paths = [Path(p) for p in model_list]
        
        if not self.model_paths:
            logger.warning("InteractomeProcessor initialized with an empty model list.")
        else:
            logger.info(f"Initialized Processor for {len(self.model_paths)} models using engine '{self._engine}'")

        # Initialize Data Containers. We explicitly type hint these as DataFrames or None
        self.df_het: Optional[pd.DataFrame] = None
        self.df_hom: Optional[pd.DataFrame] = None
        self.cluster_data: Optional[pd.DataFrame] = None

    @property
    def engine(self) -> str:
        return self._engine.value

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
        # np.argwhere returns indices of shape (N, 2) directly.
        low_pae_coords = np.argwhere(pae_submatrix < threshold)
        
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
    
    def process_ppi(self, model_file: Union[str, Path],
                    prefix: str = "") -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Processes a single CIF model file using the engine set at construction.

        Args:
            model_file: Path to the model .cif file.
            prefix: Substring to strip from the parent directory name when parsing PPI IDs.

        Returns:
            Tuple of (summary_dict, cluster_data_df).
        """
        path_obj = Path(model_file)

        dir_name = path_obj.parent.name
        ppi_id = dir_name.replace(prefix, "")
        parts = ppi_id.split("__")
        orf_a = parts[0]
        orf_b = parts[1] if len(parts) > 1 else ""
        model_number = int(path_obj.stem.split("model_")[-1].split("_")[0])

        model = Model(path_obj, self._engine)
        m = model.metrics
        md = model.model_data
        chain_ids = md.token_chain_ids

        plddt_path = path_obj.with_name(f"{path_obj.stem}_plddt.png")
        plot_plddt(m.ca_plddts, md.chain_boundaries_by_res, chain_ids, str(plddt_path))

        pae_path = path_obj.with_name(f"{path_obj.stem}_pae.png")
        plot_paes(m.pae, md.chain_boundaries_by_res, set(chain_ids),
                  f"{m.iptm} ipTM - {m.ptm} pTM", str(pae_path))

        if self._mode in [InteractomeMode.HOMOMERS, InteractomeMode.MONOMER]:
            return {
                "PPI": ppi_id,
                "ORF": orf_a,
                "Num_copies": orf_b,
                "Model_num": model_number,
                "mean_plddt": float(m.ca_plddts.mean()),
                "mean_pae": float(m.pae.mean()),
                "ipTM": m.iptm,
                "pTM": m.ptm,
                "Path": str(path_obj),
            }, pd.DataFrame()

        # INTER_PAIRS / INTRA_PAIRS — interface analysis
        unique_chains = sorted(set(chain_ids))
        chain_a, chain_b = unique_chains[0], unique_chains[1]

        metrics_input = {
            "pae": m.pae,
            "cb_plddts": m.cb_plddts,
            "token_chain_ids": chain_ids,
        }
        all_metrics = calculate_all_metrics(str(path_obj), metrics_input)

        pae_submatrix_1 = m.pae[chain_ids == chain_a][:, chain_ids == chain_b]
        pae_submatrix_2 = m.pae[chain_ids == chain_b][:, chain_ids == chain_a].T
        submatrix = np.mean([pae_submatrix_1, pae_submatrix_2], axis=0)

        low_coords, cluster_labels = InteractomeProcessor.cluster_pae(submatrix)

        cluster_plot_path = path_obj.with_name(f"{path_obj.stem}_cluster.png")
        plot_pae_clusters(submatrix, low_coords, cluster_labels, save_name=str(cluster_plot_path))

        cluster_data = InteractomeProcessor.cluster_info(low_coords=low_coords, cluster_labels=cluster_labels)
        if not cluster_data.empty:
            cluster_data["PPI"] = ppi_id
            cluster_data["model_num"] = model_number
            cluster_data["path"] = str(path_obj)

        summary_dict = {
            "PPI": ppi_id,
            "ORF_A": orf_a,
            "ORF_B": orf_b,
            "Folder": str(path_obj.parent),
            "Path": str(path_obj),
            "Model_num": model_number,
            "ipTM": m.iptm,
            "pTM": m.ptm,
            "pTM_chain_A": float(m.iptm_chain_pair[0][0]),
            "pTM_chain_B": float(m.iptm_chain_pair[1][1]),
            **all_metrics,
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
            # We assume the 'Path' column contains the path to the model
            if "Path" in interactome_df.columns:
                processed_paths = set(interactome_df["Path"].astype(str))
                    
                # Identify which paths from input are not in the processed set
                # We convert path.parent to string to match the CSV format
                models_to_process = [
                    p for p in models_to_process 
                    if str(p) not in processed_paths
                ]
                logger.info(f"Skipping {len(self.model_paths) - len(models_to_process)} already processed models.")
        if not models_to_process:
            logger.info("All models are already processed. Exiting.")
            return
        
        logger.info(f"Starting parallel processing for {len(models_to_process)} models...")
    
        # Parallel Execution
        new_interactome_list = []
        new_clusters_list = []

        with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
            worker = partial(self.process_ppi,
                            #  model_type=self._engine,
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
        # if self._mode in [InteractomeMode.HOMOMERS]:
        #     new_interactome_df.rename(columns={"ORF_A": "ORF", "ORF_B": "N_copies"}, inplace=True)
        
        # if self._mode in [InteractomeMode.INTER_PAIRS, InteractomeMode.INTRA_PAIRS]:
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
        ## NOTE: I do not know what is going on below... Maybe should be removed
        if self._mode in [InteractomeMode.INTER_PAIRS, InteractomeMode.INTRA_PAIRS] and not final_clusters_df.empty:
            desired_columns = [
                'PPI', 'model_num', 'path', 'cluster_id', 'num_points', 
                'x_len', 'y_len', 'x_min', 'x_max', 'y_min', 'y_max',
                'center_x', 'center_y', 'cluster_ratio'
            ]
        # Ensure columns exist (handle case-sensitivity if old CSV had 'Cluster_ratio')
            # if 'Cluster_ratio' in final_clusters_df.columns and 'cluster_ratio' not in final_clusters_df.columns:
            #     final_clusters_df.rename(columns={'Cluster_ratio': 'cluster_ratio'}, inplace=True)
            
            # Select and order columns safely
            # Only select columns that actually exist to avoid KeyError
            existing_cols = [c for c in desired_columns if c in final_clusters_df.columns]
            final_clusters_df = final_clusters_df[existing_cols]

            final_clusters_df = final_clusters_df.round(2)
            final_clusters_df.to_csv(clusters_csv, index=False)
        
        logger.info(f"Done. Data saved to {out_dir}")

    # ── Pooled ColabFold processing ───────────────────────────────────────────

    _IPTM_SIZE_CORRECTION_INTERCEPT: float = -0.036255571
    _IPTM_SIZE_CORRECTION_SLOPE: float = 0.004470512

    @staticmethod
    def _size_correct_iptm(iptm: float, len_a: int, len_b: int) -> float:
        """Apply size correction to a pairwise ipTM score (Todor et al. 2026).

        Corrects for the systematic positive bias of ipTM with larger protein
        pairs.  Coefficients derived for ~200–500 aa bacterial proteins; the
        direction of the correction is valid for viral proteins as well.

        Returns ``iptm - expected_iptm`` where
        ``expected_iptm = intercept + slope * sqrt(len_a + len_b)``.
        """
        expected = (
            InteractomeProcessor._IPTM_SIZE_CORRECTION_INTERCEPT
            + InteractomeProcessor._IPTM_SIZE_CORRECTION_SLOPE * np.sqrt(len_a + len_b)
        )
        return float(iptm - expected)

    @staticmethod
    def _process_pool_model(
        cif_path: Path,
        chain_to_protein: Dict[str, str],
        engine: Engine,
        ppi_separator: str = "__",
    ) -> List[Dict[str, Any]]:
        """Extract pairwise metrics for every protein pair in one pool model.

        Computes structural metrics (ipSAE, LIS family, pDockQ2, pLDDT, PAE)
        for all unique unordered chain pairs in a pooled multimer CIF.
        Metric DataFrames are computed once per file and sliced per pair.

        Parameters
        ----------
        cif_path : Path
            Path to the ColabFold multimer CIF file.
        chain_to_protein : Dict[str, str]
            Mapping of chain letter → protein ID in pool order
            (e.g. ``{'A': 'E1A', 'B': 'pVII', 'C': 'fiber'}``).
        engine : Engine
            Folding engine (must be :attr:`Engine.COLABFOLD`).
        ppi_separator : str
            Separator used to build PPI identifiers. Defaults to ``'__'``.

        Returns
        -------
        List[dict]
            One dict per unique protein pair with all metric columns plus
            ``PPI``, ``ORF_A``, ``ORF_B``, ``Path``, ``ipTM``,
            ``size_corrected_ipTM``.
        """
        model = Model(cif_path, engine)
        m = model.metrics
        md = model.model_data
        chain_ids = np.array(md.token_chain_ids)
        model_chains = np.unique(chain_ids)
        chain_to_idx: Dict[str, int] = {c: i for i, c in enumerate(model_chains)}

        cif_str = str(cif_path)
        ipsae_df  = calculate_ipsae(cif_str, m.pae)
        lis_df    = calculate_LIS_family(cif_str, m.pae)
        pdockq2_df = calculate_pdockq2(cif_str, plddt_by_res=m.cb_plddts, pae_matrix=m.pae)

        def _ipsae_val(c1: str, c2: str, col: str) -> float:
            row = ipsae_df.loc[(ipsae_df.chain1 == c1) & (ipsae_df.chain2 == c2), col]
            return float(row.values[0]) if len(row) > 0 else 0.0

        def _lis_val(c1: str, c2: str, col: str):
            row = lis_df.loc[(lis_df.chain1 == c1) & (lis_df.chain2 == c2), col]
            return row.values[0] if len(row) > 0 else (0.0 if col not in ("LIR", "cLIR") else "")

        def _pdq2_val(c1: str, c2: str) -> float:
            row = pdockq2_df.loc[(pdockq2_df.chain1 == c1) & (pdockq2_df.chain2 == c2), "pDockQ2"]
            return float(row.values[0]) if len(row) > 0 else 0.0

        pool_chains = sorted(chain_to_protein.keys())
        results: List[Dict[str, Any]] = []

        for i, chain_a in enumerate(pool_chains):
            for chain_b in pool_chains[i + 1:]:
                protein_a = chain_to_protein[chain_a]
                protein_b = chain_to_protein[chain_b]
                ppi_id = f"{protein_a}{ppi_separator}{protein_b}"

                mask_a = chain_ids == chain_a
                mask_b = chain_ids == chain_b

                idx_a = chain_to_idx[chain_a]
                idx_b = chain_to_idx[chain_b]
                iptm_pair = float(m.iptm_chain_pair[idx_a, idx_b])

                len_a = int(mask_a.sum())
                len_b = int(mask_b.sum())
                sc_iptm = InteractomeProcessor._size_correct_iptm(iptm_pair, len_a, len_b)

                pae_ab = m.pae[np.ix_(mask_a, mask_b)]
                pae_ba = m.pae[np.ix_(mask_b, mask_a)]

                lis_ab   = float(_lis_val(chain_a, chain_b, "LIS"))
                lis_ba   = float(_lis_val(chain_b, chain_a, "LIS"))
                lia_ab   = float(_lis_val(chain_a, chain_b, "LIA"))
                lia_ba   = float(_lis_val(chain_b, chain_a, "LIA"))
                clis_ab  = float(_lis_val(chain_a, chain_b, "cLIS"))
                clis_ba  = float(_lis_val(chain_b, chain_a, "cLIS"))
                clia_ab  = float(_lis_val(chain_a, chain_b, "cLIA"))
                clia_ba  = float(_lis_val(chain_b, chain_a, "cLIA"))
                ilis_ab  = float(_lis_val(chain_a, chain_b, "iLIS"))
                ilis_ba  = float(_lis_val(chain_b, chain_a, "iLIS"))
                ilia_ab  = float(_lis_val(chain_a, chain_b, "iLIA"))
                ilia_ba  = float(_lis_val(chain_b, chain_a, "iLIA"))
                ipsae_ab = _ipsae_val(chain_a, chain_b, "ipSAE")
                ipsae_ba = _ipsae_val(chain_b, chain_a, "ipSAE")

                results.append({
                    "PPI":        ppi_id,
                    "ORF_A":      protein_a,
                    "ORF_B":      protein_b,
                    "Path":       cif_str,
                    "ipTM":       iptm_pair,
                    "size_corrected_ipTM": sc_iptm,
                    "pTM_chain_A": float(m.iptm_chain_pair[idx_a][idx_a]),
                    "pTM_chain_B": float(m.iptm_chain_pair[idx_b][idx_b]),
                    "pLDDT_mean":     float(np.mean(m.cb_plddts)),
                    "pLDDT_mean_A":   float(np.mean(m.cb_plddts[mask_a])),
                    "pLDDT_mean_B":   float(np.mean(m.cb_plddts[mask_b])),
                    "pLDDT_median_A": float(np.median(m.cb_plddts[mask_a])),
                    "pLDDT_median_B": float(np.median(m.cb_plddts[mask_b])),
                    "pae_mean":    float(np.mean(m.pae)),
                    "pae_mean_A":  float(np.mean(m.pae[np.ix_(mask_a, mask_a)])),
                    "pae_mean_B":  float(np.mean(m.pae[np.ix_(mask_b, mask_b)])),
                    "pae_mean_AB": float(np.mean([np.mean(pae_ab), np.mean(pae_ba)])),
                    "pDockQ2_AB":  _pdq2_val(chain_a, chain_b),
                    "pDockQ2_BA":  _pdq2_val(chain_b, chain_a),
                    "LIS_AB":  lis_ab,  "LIS_BA":  lis_ba,
                    "LIA_AB":  lia_ab,  "LIA_BA":  lia_ba,
                    "cLIS_AB": clis_ab, "cLIS_BA": clis_ba,
                    "cLIA_AB": clia_ab, "cLIA_BA": clia_ba,
                    "iLIS_AB": ilis_ab, "iLIS_BA": ilis_ba,
                    "iLIA_AB": ilia_ab, "iLIA_BA": ilia_ba,
                    "Best_LIS":  float(max(lis_ab,  lis_ba)),
                    "Best_LIA":  float(max(lia_ab,  lia_ba)),
                    "Best_iLIS": float(max(ilis_ab, ilis_ba)),
                    "Best_iLIA": float(max(ilia_ab, ilia_ba)),
                    "LIR_AB":   _lis_val(chain_a, chain_b, "LIR"),
                    "cLIR_AB":  _lis_val(chain_a, chain_b, "cLIR"),
                    "ipSAE_AB":      ipsae_ab,
                    "ipSAE_BA":      ipsae_ba,
                    "max_ipSAE":     float(max(ipsae_ab, ipsae_ba)),
                    "ipSAE_d0chn_AB": _ipsae_val(chain_a, chain_b, "ipSAE_d0chn"),
                    "ipSAE_d0chn_BA": _ipsae_val(chain_b, chain_a, "ipSAE_d0chn"),
                    "ipSAE_d0dom_AB": _ipsae_val(chain_a, chain_b, "ipSAE_d0dom"),
                    "ipSAE_d0dom_BA": _ipsae_val(chain_b, chain_a, "ipSAE_d0dom"),
                })

        return results

    @classmethod
    def process_pooled(
        cls,
        pool_manifest_path: Union[str, Path],
        colabfold_output_dir: Union[str, Path],
        output_path: Union[str, Path],
        *,
        ppi_separator: str = "__",
    ) -> pd.DataFrame:
        """Process pooled ColabFold outputs into a single aggregated CSV.

        Reads ``pool_manifest.csv`` (written by
        :meth:`InteractomeWriter.write_pooled_jobs`), finds CIF files for
        each pool under ``colabfold_output_dir/{pool_id}/``, extracts
        pairwise metrics for every protein pair in each pool, and aggregates:

        - **Within a pool**: numeric columns are averaged over all CIF ranks.
        - **Across pools**: pairs appearing in multiple pools are averaged again;
          a ``n_pools`` column records how many pools contributed.

        The output schema matches ``interactome_data.csv`` produced by
        :meth:`process_models`, so :class:`InteractomeAnalyzer` can load it
        without changes.

        Parameters
        ----------
        pool_manifest_path : str or Path
            Path to ``pool_manifest.csv`` from :meth:`~InteractomeWriter.write_pooled_jobs`.
        colabfold_output_dir : str or Path
            Root directory of ColabFold outputs.  Expected structure after
            :func:`~virus_interactome.utils.reorganize_colabfold_outputs`::

                colabfold_output_dir/
                    pool_0000/
                        pool_0000_unrelaxed_rank_001_*.cif
                        ...
                    pool_0001/
                        ...

        output_path : str or Path
            Directory where ``interactome_data.csv`` is written.
        ppi_separator : str
            Separator used to build PPI identifiers. Defaults to ``'__'``.

        Returns
        -------
        pd.DataFrame
            Aggregated metrics, one row per unique protein pair.
        """
        manifest = pd.read_csv(pool_manifest_path)
        cf_dir = Path(colabfold_output_dir)
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_records: List[Dict[str, Any]] = []

        for _, row in manifest.iterrows():
            pool_id: str = str(row["pool_id"])
            proteins: List[str] = str(row["proteins"]).split(",")
            chain_letters = [chr(ord("A") + i) for i in range(len(proteins))]
            chain_to_protein: Dict[str, str] = dict(zip(chain_letters, proteins))

            pool_dir = cf_dir / pool_id
            if not pool_dir.exists():
                logger.warning(f"Pool directory not found, skipping: {pool_dir}")
                continue

            cif_files = sorted(pool_dir.glob("*.cif"))
            if not cif_files:
                logger.warning(f"No CIF files found in {pool_dir}, skipping.")
                continue

            # Collect per-model records for this pool
            pool_records: List[Dict[str, Any]] = []
            for cif_path in cif_files:
                try:
                    records = cls._process_pool_model(
                        cif_path, chain_to_protein, Engine.COLABFOLD, ppi_separator
                    )
                    for rec in records:
                        rec["pool_id"] = pool_id
                    pool_records.extend(records)
                except Exception as exc:
                    logger.error(f"Failed to process {cif_path}: {exc}")

            if not pool_records:
                continue

            # Average numeric columns over CIF ranks within this pool
            pool_df = pd.DataFrame(pool_records)
            str_cols = {"PPI", "ORF_A", "ORF_B", "Path", "pool_id", "LIR_AB", "cLIR_AB"}
            num_cols = [c for c in pool_df.columns if c not in str_cols]

            agg = pool_df.groupby("PPI")[num_cols].mean().reset_index()
            meta = pool_df.groupby("PPI")[["ORF_A", "ORF_B", "pool_id"]].first().reset_index()
            pool_avg = meta.merge(agg, on="PPI")
            all_records.append(pool_avg)

        if not all_records:
            logger.warning("process_pooled: no records produced.")
            return pd.DataFrame()

        combined = pd.concat(all_records, ignore_index=True)

        str_cols = {"PPI", "ORF_A", "ORF_B", "pool_id", "LIR_AB", "cLIR_AB"}
        num_cols = [c for c in combined.columns if c not in str_cols]

        agg_final = combined.groupby("PPI")[num_cols].mean().reset_index()
        n_pools = combined.groupby("PPI")["pool_id"].nunique().reset_index(name="n_pools")
        meta_final = combined.groupby("PPI")[["ORF_A", "ORF_B"]].first().reset_index()

        result = meta_final.merge(n_pools, on="PPI").merge(agg_final, on="PPI")
        result = result.round(4)

        out_csv = out_dir / "interactome_data.csv"
        result.to_csv(out_csv, index=False)
        logger.info(f"process_pooled: {len(result)} PPIs → {out_csv}")

        return result

    @staticmethod
    def _extract_monomer_plddt(
        cif_path: Path,
        engine: str,
    ) -> Dict[str, Any]:
        """
        Extracts pLDDT statistics from a single-chain (monomer) CIF file.

        Parameters
        ----------
        cif_path : Path
            Path to the monomer ``.cif`` model file.
        engine : str
            Engine that produced the file. One of ``'af3'``, ``'boltz'``, ``'boltz2'``,
            ``'colabfold'``.

        Returns
        -------
        dict
            Keys: ``plddt_mean``, ``plddt_median``, ``n_residues``.
            Values are ``np.nan`` if parsing fails.
        """
        nan_result: Dict[str, Any] = {
            "plddt_mean": np.nan,
            "plddt_median": np.nan,
            "n_residues": np.nan,
        }
        try:
            engine_norm = "boltz" if str(engine).lower() == "boltz2" else engine
            model  = Model(cif_path, engine=engine_norm)
            plddts = model._metrics.ca_plddts
            return {
                "plddt_mean":   float(np.mean(plddts)),
                "plddt_median": float(np.median(plddts)),
                "n_residues":   int(len(plddts)),
            }
        except Exception as exc:
            logger.warning(f"Could not extract pLDDT from {cif_path}: {exc}")
            return nan_result

    def process_monomers(
        self,
        output_path: Union[str, Path] = ".",
        prefix: str = "",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Processes a list of monomer CIF files in parallel and writes a summary CSV.

        Extracts per-protein pLDDT statistics (mean, median) without computing
        interface metrics (ipSAE, LIS, etc.), which are undefined for monomers.
        Implements the same resume-logic as :meth:`process_models`: already-processed
        proteins are skipped on re-runs.

        Parameters
        ----------
        output_path : str or Path, optional
            Directory where ``monomer_data.csv`` is saved. Defaults to ``"."``.
        prefix : str, optional
            Substring stripped from folder names when parsing the protein ID.
            Defaults to ``""``.
        **kwargs
            Forwarded to :class:`concurrent.futures.ProcessPoolExecutor`
            (e.g. ``max_workers=4``).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ``protein_id``, ``cif_path``, ``n_residues``,
            ``plddt_mean``, ``plddt_median``.
            Also written to ``{output_path}/monomer_data.csv``.

        Notes
        -----
        For AF3 monomers the default is 5 models per protein; for Boltz2, 1 model.
        Pass the appropriate ``model_list`` to :class:`InteractomeProcessor`
        accordingly.
        """
        import tqdm

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        monomer_csv = out_dir / "monomer_data.csv"

        existing_df = pd.DataFrame()
        models_to_process = self.model_paths.copy()

        # Resume logic: skip already-processed CIF paths
        if monomer_csv.exists():
            logger.info(f"Found existing data at {monomer_csv}. Resuming run...")
            existing_df = pd.read_csv(monomer_csv)
            if "cif_path" in existing_df.columns:
                processed = set(existing_df["cif_path"].astype(str))
                models_to_process = [
                    p for p in models_to_process if str(p) not in processed
                ]
                logger.info(
                    f"Skipping {len(self.model_paths) - len(models_to_process)} "
                    "already processed models."
                )

        if not models_to_process:
            logger.info("All monomer models already processed. Exiting.")
            return existing_df

        logger.info(f"Processing {len(models_to_process)} monomer models...")

        def _worker(cif_path: Path) -> Dict[str, Any]:
            dir_name = cif_path.parent.name
            protein_id = dir_name.replace(prefix, "")
            stats = InteractomeProcessor._extract_monomer_plddt(cif_path, self._engine)
            return {"protein_id": protein_id, "cif_path": str(cif_path), **stats}

        new_rows: list = []
        with concurrent.futures.ProcessPoolExecutor(**kwargs) as executor:
            for row in tqdm.tqdm(
                executor.map(_worker, models_to_process),
                total=len(models_to_process),
                desc="Processing Monomers",
            ):
                new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.round(2)
        final_df.to_csv(monomer_csv, index=False)
        logger.info(f"Monomer data saved to {monomer_csv}")
        return final_df


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
        msa_threshold: int = 20,
        lis_threshold: float = 0.203,
        lia_threshold: float = 3432.0,
        ilis_threshold: float = 0.223,
    ) -> pd.DataFrame:
        """
        Categorizes interactome results into confidence tiers.

        Adds three independent tier columns:
          - ``Tier``      — ipSAE-based (original; Dunbrack 2025 + pDockQ2 + MSA depth).
          - ``LIS_Tier``  — LIS-based (Kim et al. 2024): Best LIS + Best LIA dual threshold.
          - ``iLIS_Tier`` — iLIS-based (Kim et al. 2025): single Best iLIS threshold.

        Parameters
        ----------
        ipsae_threshold : float, default=0.5
        pdockq2_threshold : float, default=0.23
        msa_threshold : int, default=20
        lis_threshold : float, default=0.203
            Best LIS threshold for LIS_Tier "High Confidence".
        lia_threshold : float, default=3432
            Best LIA threshold for LIS_Tier "High Confidence".
        ilis_threshold : float, default=0.223
            Best iLIS threshold for iLIS_Tier "High Confidence".

        Returns
        -------
        pd.DataFrame
            Interactome data with added ``Tier``, ``LIS_Tier``, and ``iLIS_Tier`` columns.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        # ── ipSAE-based tier (original) ──────────────────────────────────────
        ipsae_col   = "ipSAE_AB" if "ipSAE_AB" in df.columns else "ipSAE"
        pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
        msa_col     = "msa_depth"

        def _ipsae_classify(row) -> str:
            if ipsae_col not in row or pdockq2_col not in row:
                return "Unknown"
            msa_val     = row.get(msa_col, np.nan)
            ipsae_val   = row[ipsae_col]
            pdockq2_val = row[pdockq2_col]
            good_struct = ipsae_val > ipsae_threshold and pdockq2_val > pdockq2_threshold
            msa_ok      = pd.isna(msa_val) or msa_val > msa_threshold
            if good_struct and msa_ok:
                return "Tier 1 (High Confidence)"
            if good_struct and not msa_ok:
                return "Tier 2 (Specific/Novel)"
            if ipsae_val > ipsae_threshold and pdockq2_val <= pdockq2_threshold:
                return "Tier 3 (Weak/Dynamic)"
            return "Low Confidence"

        df["Tier"] = df.apply(_ipsae_classify, axis=1)

        # ── LIS-based tier (Kim et al. 2024) ─────────────────────────────────
        if "Best_LIS" in df.columns and "Best_LIA" in df.columns:
            def _lis_classify(row) -> str:
                if row["Best_LIS"] >= lis_threshold and row["Best_LIA"] >= lia_threshold:
                    return "High Confidence"
                if row["Best_LIS"] >= lis_threshold:
                    return "Low LIA"
                return "Low Confidence"
            df["LIS_Tier"] = df.apply(_lis_classify, axis=1)
        else:
            logger.warning(
                "Columns 'Best_LIS'/'Best_LIA' not found — LIS_Tier set to 'N/A'. "
                "Re-process models with the current metrics.py to generate them."
            )
            df["LIS_Tier"] = "N/A"

        # ── iLIS-based tier (Kim et al. 2025) ────────────────────────────────
        if "Best_iLIS" in df.columns:
            df["iLIS_Tier"] = df["Best_iLIS"].apply(
                lambda v: "High Confidence" if v >= ilis_threshold else "Low Confidence"
            )
        else:
            logger.warning(
                "Column 'Best_iLIS' not found — iLIS_Tier set to 'N/A'. "
                "Re-process models with the current metrics.py to generate them."
            )
            df["iLIS_Tier"] = "N/A"

        logger.info(
            f"Tier (ipSAE):\n{df['Tier'].value_counts()}\n"
            f"LIS_Tier:\n{df['LIS_Tier'].value_counts()}\n"
            f"iLIS_Tier:\n{df['iLIS_Tier'].value_counts()}"
        )
        return df

    def plot_confidence_landscape(self, output_path: Optional[Union[str, Path]] = None,
                                  title: str = "Interactome Confidence Landscape"):
        """
        Generates a scatter plot of the interactome confidence landscape.
        X-axis: pDockQ2, Y-axis: ipSAE_d0_dom, Size: msa_depth, Color: pLDDT (AlphaFold colors).
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
        plt.title(title)
        
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

    def plot_interactive_landscape(self, output_path: Optional[Union[str, Path]] = None,
                                   title: str = "Interactome Confidence Landscape"):
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
            title=f"{title}<br><sup>Bubble size = sqrt(MSA depth)</sup>",
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
        """Peptide-protein candidate DataFrame set during pipeline execution."""
        return self._binder_data

    @binder_data.setter
    def binder_data(self, df: pd.DataFrame):
        """Set the binder candidate DataFrame directly."""
        self._binder_data = df

    # -------------------------------------------------------------------------
    # Interactome Data Management
    # -------------------------------------------------------------------------
    
    @property
    def interactome_path(self)-> Optional[Path]:
        """Path to the loaded interactome CSV file."""
        return self._interactome_path

    @property
    def interactome_data(self)-> Optional[pd.DataFrame]:
        """Loaded interactome DataFrame (None until ``interactome_path`` is set)."""
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
        """Path to the loaded cluster CSV file."""
        return self._cluster_path

    @property
    def cluster_data(self)-> Optional[pd.DataFrame]:
        """Loaded cluster DataFrame (None until ``cluster_path`` is set)."""
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
        """Common root directory for model ``.cif`` files."""
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
        
        # Update Cluster Data (regex=False: treat path separators as literals)
        if "path" in self._cluster_data.columns:
            self._cluster_data["path"] = self._cluster_data["path"].astype(str).str.replace(old_path, new_model_path, regex=False)

        # Update Interactome Data
        if "Folder" in self._interactome_data.columns:
            self._interactome_data["Folder"] = self._interactome_data["Folder"].astype(str).str.replace(old_path, new_model_path, regex=False)
        
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
        """Return the number of PPI rows in the loaded interactome (0 if not loaded)."""
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

        # 4. Process Candidates (Identify Binder vs Peptide)
        new_cols = {
            "Binder_chain": [], "Binder_name": [], "Binder_start": [], "Binder_end": [],
            "Peptide_chain": [], "Peptide_name": [], "Peptide_start": [], "Peptide_end": []
        }

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
                new_cols["Peptide_start"].append(int(row["y_min"]))
                new_cols["Peptide_end"].append(int(row["y_max"]))
                new_cols["Binder_start"].append(int(row["x_min"]))
                new_cols["Binder_end"].append(int(row["x_max"]))
                new_cols["Binder_name"].append(orf_a)
                new_cols["Peptide_name"].append(orf_b)
            else:
                # Y is longer
                new_cols["Binder_chain"].append("B")
                new_cols["Peptide_chain"].append("A")
                new_cols["Peptide_start"].append(int(row["x_min"]))
                new_cols["Peptide_end"].append(int(row["x_max"]))
                new_cols["Binder_start"].append(int(row["y_min"]))
                new_cols["Binder_end"].append(int(row["y_max"]))
                new_cols["Binder_name"].append(orf_b)
                new_cols["Peptide_name"].append(orf_a)
        
        # 5. Assign new columns to DataFrame
        for col_name, data_list in new_cols.items():
            candidate_clusters[col_name] = data_list
        
        
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
            f.write(f"save {session_path}\n")
            f.write(f"exit\n")

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

    # =========================================================================
    # --- Generic Analysis Methods ---
    # =========================================================================

    def _resolve_metric_col(self, preferred: str, fallback: str) -> Optional[str]:
        """Returns preferred column name if present, fallback if not, else None."""
        if self._interactome_data is None:
            return None
        if preferred in self._interactome_data.columns:
            return preferred
        if fallback in self._interactome_data.columns:
            return fallback
        return None

    def filter_by_metrics(self, criteria: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Filters the interactome by multiple metric ranges simultaneously.

        Parameters
        ----------
        criteria : Dict[str, Tuple[float, float]]
            Keys are column names, values are (min, max) inclusive ranges.
            Example: {"ipSAE_AB": (0.5, 1.0), "msa_depth": (20, 9999)}

        Returns
        -------
        pd.DataFrame
            Filtered subset of the interactome data.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()
        missing = [col for col in criteria if col not in df.columns]
        if missing:
            logger.warning(f"filter_by_metrics: columns not found and will be skipped: {missing}")

        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in criteria.items():
            if col in df.columns:
                mask &= df[col].between(lo, hi)

        result = df[mask].reset_index(drop=True)
        logger.info(f"filter_by_metrics: {len(result)}/{len(df)} rows passed the filter.")
        return result

    def get_top_interactions(
        self,
        metric: str = "ipSAE_AB",
        top_n: int = 10,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the top N interactions ranked by a given metric.

        Falls back to 'ipSAE' if the preferred column is absent.

        Parameters
        ----------
        metric : str
            Column name to rank by. Defaults to "ipSAE_AB".
        top_n : int
            Number of interactions to return.
        ascending : bool
            If True, returns the N lowest values instead.

        Returns
        -------
        pd.DataFrame
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data
        fallback = metric.replace("_AB", "") if "_AB" in metric else metric
        col = metric if metric in df.columns else (fallback if fallback in df.columns else None)

        if col is None:
            raise ValueError(f"Column '{metric}' (and fallback '{fallback}') not found in interactome data.")

        return (
            df.sort_values(col, ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )

    def summarize_by_protein(self, ppi_separator: str = "__") -> pd.DataFrame:
        """
        Generates a per-protein summary across all interactions.

        Parses the 'PPI' column to extract individual protein IDs, then aggregates:
        - degree: number of interaction partners
        - mean_ipSAE, mean_pDockQ2: average confidence metrics
        - best_partner: partner with the highest ipSAE value

        Parameters
        ----------
        ppi_separator : str
            Delimiter used in the PPI column to separate the two protein IDs.
            Defaults to "__".

        Returns
        -------
        pd.DataFrame
            One row per protein with aggregated statistics.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()
        ipsae_col = self._resolve_metric_col("ipSAE_AB", "ipSAE")
        pdockq2_col = self._resolve_metric_col("pDockQ2_AB", "pDockQ2")

        try:
            split = df["PPI"].str.split(ppi_separator, n=1, expand=True)
            df["_prot_a"] = split[0]
            df["_prot_b"] = split[1] if 1 in split.columns else split[0]
        except Exception as e:
            raise ValueError(f"Failed to parse PPI column with separator '{ppi_separator}': {e}")

        records: Dict[str, Any] = {}

        for _, row in df.iterrows():
            for prot, partner in [(row["_prot_a"], row["_prot_b"]), (row["_prot_b"], row["_prot_a"])]:
                if prot not in records:
                    records[prot] = {"ipsae_vals": [], "pdockq2_vals": [], "partners": {}}
                ipsae_val = row[ipsae_col] if ipsae_col else np.nan
                records[prot]["ipsae_vals"].append(ipsae_val)
                if pdockq2_col:
                    records[prot]["pdockq2_vals"].append(row[pdockq2_col])
                # Track best partner by ipSAE
                current_best = records[prot]["partners"].get(partner, -np.inf)
                records[prot]["partners"][partner] = max(current_best, ipsae_val if not np.isnan(ipsae_val) else -np.inf)

        summary_rows = []
        for prot, data in records.items():
            ipsae_arr = [v for v in data["ipsae_vals"] if not np.isnan(v)]
            pdockq2_arr = [v for v in data["pdockq2_vals"] if not np.isnan(v)]
            best_partner = max(data["partners"], key=data["partners"].get) if data["partners"] else None
            summary_rows.append({
                "protein": prot,
                "degree": len(data["partners"]),
                "mean_ipSAE": np.mean(ipsae_arr) if ipsae_arr else np.nan,
                "max_ipSAE": np.max(ipsae_arr) if ipsae_arr else np.nan,
                "mean_pDockQ2": np.mean(pdockq2_arr) if pdockq2_arr else np.nan,
                "best_partner": best_partner,
            })

        result = pd.DataFrame(summary_rows).sort_values("degree", ascending=False).reset_index(drop=True)
        logger.info(f"summarize_by_protein: {len(result)} unique proteins found.")
        return result

    def export_to_network(
        self,
        output_format: str = "cytoscape",
        ppi_separator: str = "__",
        extra_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Exports the interactome as an edge list for network visualization tools.

        Produces a DataFrame compatible with Cytoscape (edge list format) or Gephi
        (with 'Source'/'Target' headers). Edge attributes include ipSAE, pDockQ2,
        pLDDT_mean, msa_depth, and Tier if present.

        Parameters
        ----------
        output_format : str
            "cytoscape" or "gephi". Controls header naming convention.
        ppi_separator : str
            Delimiter to split the PPI column into source/target nodes.
        extra_cols : list of str, optional
            Additional interactome columns to include as edge attributes.

        Returns
        -------
        pd.DataFrame
            Edge list with node and attribute columns.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        try:
            split = df["PPI"].str.split(ppi_separator, n=1, expand=True)
            df["_source"] = split[0]
            df["_target"] = split[1] if 1 in split.columns else split[0]
        except Exception as e:
            raise ValueError(f"Failed to parse PPI column with separator '{ppi_separator}': {e}")

        src_col, tgt_col = ("Source", "Target") if output_format == "gephi" else ("source", "target")

        default_attrs = ["ipSAE_AB", "ipSAE", "pDockQ2_AB", "pDockQ2",
                         "pLDDT_mean", "msa_depth", "Tier"]
        attr_cols = [c for c in default_attrs if c in df.columns]
        if extra_cols:
            attr_cols += [c for c in extra_cols if c in df.columns and c not in attr_cols]

        edge_df = df[["_source", "_target"] + attr_cols].rename(
            columns={"_source": src_col, "_target": tgt_col}
        ).reset_index(drop=True)

        logger.info(f"export_to_network: exported {len(edge_df)} edges with {len(attr_cols)} attributes.")
        return edge_df

    def compare_engines(
        self,
        other_df: pd.DataFrame,
        suffix_self: str = "_a",
        suffix_other: str = "_b",
        on: str = "PPI",
    ) -> pd.DataFrame:
        """
        Compares metrics between two interactome runs (e.g., AF3 vs Boltz2).

        Merges the loaded interactome with a second DataFrame on the PPI key,
        computing the delta for each shared numeric metric.

        Parameters
        ----------
        other_df : pd.DataFrame
            Second interactome result DataFrame. Must contain a 'PPI' column.
        suffix_self : str
            Suffix appended to columns from the loaded (self) interactome.
        suffix_other : str
            Suffix appended to columns from `other_df`.
        on : str
            Join key column name. Defaults to "PPI".

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with per-metric delta columns (prefix 'delta_').
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")
        if on not in other_df.columns:
            raise ValueError(f"Column '{on}' not found in other_df.")

        merged = pd.merge(
            self._interactome_data,
            other_df,
            on=on,
            suffixes=(suffix_self, suffix_other),
            how="inner",
        )

        # Compute deltas for numeric columns that appear in both
        base_cols = set(self._interactome_data.columns) - {on}
        other_cols = set(other_df.columns) - {on}
        shared_numeric = [
            c for c in base_cols & other_cols
            if pd.api.types.is_numeric_dtype(self._interactome_data[c])
        ]

        for col in shared_numeric:
            try:
                merged[f"delta_{col}"] = merged[f"{col}{suffix_self}"] - merged[f"{col}{suffix_other}"]
            except KeyError:
                pass  # column may have been renamed or absent after merge

        logger.info(
            f"compare_engines: {len(merged)} PPIs in common, "
            f"{len(shared_numeric)} numeric metrics compared."
        )
        return merged

    def cluster_interactome_by_metrics(
        self,
        n_clusters: int = 4,
        metric_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Applies K-Means clustering to group interactions by their metric profile.

        Discovers non-obvious interaction patterns beyond single-threshold filtering.
        Uses StandardScaler normalization before clustering to handle metric heterogeneity.

        Parameters
        ----------
        n_clusters : int
            Number of K-Means clusters. Defaults to 4.
        metric_cols : list of str, optional
            Columns to use as feature vector. If None, auto-selects all available
            numeric columns among: ipSAE_AB, pDockQ2_AB, pLDDT_mean, msa_depth,
            ipTM, pTM, and their non-suffixed variants.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Interactome data with an added 'km_cluster' column.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for cluster_interactome_by_metrics.")

        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        if metric_cols is None:
            candidates = [
                "ipSAE_AB", "ipSAE", "pDockQ2_AB", "pDockQ2",
                "pLDDT_mean", "msa_depth", "ipTM", "pTM",
                "ipSAE_d0_dom_AB", "ipSAE_d0dom_AB",
            ]
            metric_cols = [c for c in candidates if c in df.columns]
        else:
            metric_cols = [c for c in metric_cols if c in df.columns]

        if not metric_cols:
            raise ValueError("No valid metric columns found for clustering.")

        feature_matrix = df[metric_cols].copy()
        # Drop rows with all-NaN features; impute remaining NaNs with column median
        feature_matrix = feature_matrix.dropna(how="all")
        feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True))

        if len(feature_matrix) < n_clusters:
            raise ValueError(
                f"Not enough valid rows ({len(feature_matrix)}) for {n_clusters} clusters."
            )

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(X)

        result = df.loc[feature_matrix.index].copy()
        result["km_cluster"] = labels

        logger.info(
            f"cluster_interactome_by_metrics: {n_clusters} clusters on {len(feature_matrix)} rows "
            f"using features {metric_cols}."
        )
        logger.info(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")
        return result.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Network topology analysis
    # -------------------------------------------------------------------------

    def _build_ppi_graph(
        self,
        weight_col: str,
        min_weight: float,
        ppi_separator: str,
        model_agg: str,
    ) -> Any:
        """Build a weighted undirected NetworkX graph from _interactome_data.

        Aggregates multiple model-rank rows per PPI, then adds one edge per pair
        whose aggregated weight exceeds min_weight.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        df = self._interactome_data.copy()

        if weight_col not in df.columns:
            raise ValueError(f"Column '{weight_col}' not found in interactome data.")

        _META = {"PPI", "ORF_A", "ORF_B", "Folder", "Path",
                 "LIR_AB", "cLIR_AB", "pool_id", "Tier", "LIS_Tier", "iLIS_Tier"}
        num_cols = [
            c for c in df.columns
            if c not in _META and pd.api.types.is_numeric_dtype(df[c])
        ]

        if model_agg == "mean":
            meta = df.groupby("PPI")[["ORF_A", "ORF_B"]].first().reset_index() \
                if {"ORF_A", "ORF_B"}.issubset(df.columns) \
                else df.groupby("PPI")[[]].first().reset_index()
            agg = df.groupby("PPI")[num_cols].mean().reset_index()
            ppi_df = meta.merge(agg, on="PPI")
        elif model_agg == "max":
            meta = df.groupby("PPI")[["ORF_A", "ORF_B"]].first().reset_index() \
                if {"ORF_A", "ORF_B"}.issubset(df.columns) \
                else df.groupby("PPI")[[]].first().reset_index()
            agg = df.groupby("PPI")[num_cols].max().reset_index()
            ppi_df = meta.merge(agg, on="PPI")
        elif model_agg == "best":
            ppi_df = df.loc[df.groupby("PPI")[weight_col].idxmax()].reset_index(drop=True)
        else:
            raise ValueError(f"model_agg must be 'mean', 'max', or 'best'. Got: '{model_agg}'")

        G = nx.Graph()
        for _, row in ppi_df.iterrows():
            parts = str(row["PPI"]).split(ppi_separator, 1)
            if len(parts) != 2:
                logger.warning(f"Cannot parse PPI '{row['PPI']}' with separator '{ppi_separator}'. Skipping.")
                continue
            source, target = parts
            w = float(row[weight_col]) if not pd.isna(row[weight_col]) else 0.0
            if w > min_weight:
                G.add_edge(source, target, weight=w)

        return G

    def compute_network_properties(
        self,
        weight_col: str = "Best_iLIS",
        min_weight: float = 0.0,
        ppi_separator: str = "__",
        model_agg: str = "mean",
    ) -> pd.DataFrame:
        """Compute graph-theory metrics for every protein in the interactome.

        Aggregates multiple model-rank rows per PPI using ``model_agg``, builds a
        weighted undirected NetworkX graph, and returns per-protein centrality metrics.
        Hubs and bottlenecks are identified using adaptive thresholds (mean + 1σ of
        degree and betweenness distributions respectively), making the classification
        valid for any proteome size.

        Parameters
        ----------
        weight_col : str
            Numeric column to use as edge weight. Defaults to ``"Best_iLIS"``.
        min_weight : float
            Edges with weight ≤ min_weight are excluded. Defaults to ``0.0``
            (keeps all pairs with any detectable interaction signal).
        ppi_separator : str
            Separator used in the PPI column. Defaults to ``"__"``.
        model_agg : str
            Strategy to collapse multiple model-rank rows per PPI:
            ``"mean"`` (average across ranks), ``"max"`` (take maximum),
            or ``"best"`` (keep the rank with the highest weight_col value).
            Defaults to ``"mean"``.

        Returns
        -------
        pd.DataFrame
            One row per protein with columns: ``protein``, ``degree``,
            ``weighted_degree``, ``betweenness_centrality``,
            ``closeness_centrality``, ``eigenvector_centrality``,
            ``clustering_coefficient``, ``is_hub``, ``is_bottleneck``.
            Sorted descending by ``degree``.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        ValueError
            If ``weight_col`` is absent or ``model_agg`` is invalid.
        ImportError
            If ``networkx`` is not installed.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        G = self._build_ppi_graph(weight_col, min_weight, ppi_separator, model_agg)

        if G.number_of_nodes() == 0:
            logger.warning("compute_network_properties: graph has no nodes after filtering.")
            return pd.DataFrame(columns=[
                "protein", "degree", "weighted_degree",
                "betweenness_centrality", "closeness_centrality",
                "eigenvector_centrality", "clustering_coefficient",
                "is_hub", "is_bottleneck",
            ])

        import networkx as nx

        degrees = dict(G.degree())
        weighted_degrees = dict(G.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(G)

        try:
            eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge; values set to NaN.")
            eigenvector = {n: np.nan for n in G.nodes()}

        clustering = nx.clustering(G, weight="weight")

        deg_vals = np.array(list(degrees.values()), dtype=float)
        bet_vals = np.array(list(betweenness.values()), dtype=float)
        hub_threshold = float(deg_vals.mean() + deg_vals.std())
        bottleneck_threshold = float(bet_vals.mean() + bet_vals.std())

        rows = [
            {
                "protein": node,
                "degree": degrees[node],
                "weighted_degree": round(weighted_degrees[node], 4),
                "betweenness_centrality": round(betweenness[node], 4),
                "closeness_centrality": round(closeness[node], 4),
                "eigenvector_centrality": round(eigenvector[node], 4)
                    if not np.isnan(eigenvector[node]) else np.nan,
                "clustering_coefficient": round(clustering[node], 4),
                "is_hub": bool(degrees[node] > hub_threshold),
                "is_bottleneck": bool(betweenness[node] > bottleneck_threshold),
            }
            for node in G.nodes()
        ]

        result = (
            pd.DataFrame(rows)
            .sort_values("degree", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            f"compute_network_properties: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges. "
            f"Hubs: {result['is_hub'].sum()}, Bottlenecks: {result['is_bottleneck'].sum()}."
        )
        return result

    def plot_network(
        self,
        network_df: Optional[pd.DataFrame] = None,
        color_by: str = "betweenness_centrality",
        size_by: str = "degree",
        weight_col: str = "Best_iLIS",
        min_weight: float = 0.0,
        ppi_separator: str = "__",
        model_agg: str = "mean",
        label_top_n: int = 5,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Interactome Network",
    ) -> None:
        """Visualise the PPI network using a force-directed spring layout.

        Node size encodes ``size_by``, node colour encodes ``color_by`` (viridis
        colormap). Edge width scales with interaction weight. Only the top
        ``label_top_n`` nodes by ``size_by`` are labelled to avoid clutter.
        Hub and bottleneck nodes are outlined in red and blue respectively.

        Parameters
        ----------
        network_df : pd.DataFrame, optional
            Pre-computed output of :meth:`compute_network_properties`. If ``None``,
            it is computed using ``weight_col``, ``min_weight``, ``ppi_separator``,
            and ``model_agg``.
        color_by : str
            Node attribute column for colour mapping. Defaults to
            ``"betweenness_centrality"``.
        size_by : str
            Node attribute column for size scaling. Defaults to ``"degree"``.
        weight_col : str
            Edge weight column (used when rebuilding the graph). Defaults to
            ``"Best_iLIS"``.
        min_weight : float
            Edge weight threshold (used when rebuilding the graph).
        ppi_separator : str
            PPI column separator.
        model_agg : str
            Model-rank aggregation strategy.
        label_top_n : int
            Number of highest-``size_by`` nodes to label. Defaults to ``5``.
        output_path : str or Path, optional
            If provided, saves the figure to this path (300 dpi). Otherwise
            calls ``plt.show()``.
        title : str
            Figure title.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        ImportError
            If ``networkx`` or ``matplotlib`` is not installed.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        if network_df is None:
            network_df = self.compute_network_properties(
                weight_col=weight_col,
                min_weight=min_weight,
                ppi_separator=ppi_separator,
                model_agg=model_agg,
            )

        if network_df.empty:
            logger.warning("plot_network: no nodes to plot.")
            return

        G = self._build_ppi_graph(weight_col, min_weight, ppi_separator, model_agg)
        import networkx as nx
        pos = nx.spring_layout(G, weight="weight", seed=42)

        node_lookup = network_df.set_index("protein")

        # Node sizes — min-max scale to [300, 2500]
        size_vals = network_df.set_index("protein")[size_by].reindex(G.nodes()).fillna(0)
        s_min, s_max = size_vals.min(), size_vals.max()
        if s_max > s_min:
            node_sizes = 300 + 2200 * (size_vals - s_min) / (s_max - s_min)
        else:
            node_sizes = pd.Series(1000, index=size_vals.index)

        # Node colours — viridis on color_by
        color_vals = network_df.set_index("protein")[color_by].reindex(G.nodes()).fillna(0)
        norm = mcolors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
        cmap = cm.viridis
        node_colors = [cmap(norm(color_vals[n])) for n in G.nodes()]

        # Edge widths — scale to [0.5, 3.0]
        edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        if edge_weights.max() > edge_weights.min():
            edge_widths = 0.5 + 2.5 * (edge_weights - edge_weights.min()) / (
                edge_weights.max() - edge_weights.min()
            )
        else:
            edge_widths = np.full(len(edge_weights), 1.5)

        # Node edge colours: red for hubs, blue for bottlenecks, grey otherwise
        def _node_edge_color(node: str) -> str:
            if node not in node_lookup.index:
                return "grey"
            if node_lookup.at[node, "is_hub"]:
                return "red"
            if node_lookup.at[node, "is_bottleneck"]:
                return "blue"
            return "grey"

        node_edge_colors = [_node_edge_color(n) for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(12, 9))

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths, alpha=0.4, edge_color="grey",
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=[node_sizes[n] for n in G.nodes()],
            node_color=node_colors,
            edgecolors=node_edge_colors,
            linewidths=2.0,
            alpha=0.9,
        )

        # Labels for top-N nodes
        top_nodes = set(
            network_df.nlargest(label_top_n, size_by)["protein"].tolist()
        )
        labels = {n: n for n in G.nodes() if n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_by.replace("_", " ").title(), shrink=0.7)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markeredgecolor="red", markersize=10, label="Hub"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markeredgecolor="blue", markersize=10, label="Bottleneck"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
        ax.set_title(title)
        ax.axis("off")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Network plot saved to {output_path}")
        else:
            plt.show()
        plt.close()

    # -------------------------------------------------------------------------
    # Foldseek structural homology search
    # -------------------------------------------------------------------------

    # _FOLDSEEK_API = "https://search.foldseek.com/api"

    # def _submit_foldseek_job(
    #     self,
    #     cif_content: str,
    #     databases: List[str],
    #     mode: str = "3diaa",
    # ) -> str:
    #     """
    #     Submits a structure search to the Foldseek web API.

    #     Parameters
    #     ----------
    #     cif_content : str
    #         Raw text content of the CIF file.
    #     databases : list of str
    #         Foldseek database identifiers (e.g. ``["afdb-swissprot", "pdb100"]``).
    #     mode : str, optional
    #         Search mode. ``"3diaa"`` (default) or ``"tmalign"``.

    #     Returns
    #     -------
    #     str
    #         Ticket ID assigned by the Foldseek server.

    #     Raises
    #     ------
    #     ImportError
    #         If ``requests`` is not installed.
    #     RuntimeError
    #         If the server returns a non-200 status code.
    #     """
    #     try:
    #         import requests
    #     except ImportError:
    #         raise ImportError("requests is required for Foldseek searches. Install it with: pip install requests")

    #     payload: Dict[str, Any] = {"q": cif_content, "mode": mode}
    #     for db in databases:
    #         payload.setdefault("database[]", [])
    #         payload["database[]"].append(db)  # type: ignore[union-attr]

    #     response = requests.post(f"{self._FOLDSEEK_API}/ticket", data=payload, timeout=30)
    #     if response.status_code != 200:
    #         raise RuntimeError(
    #             f"Foldseek submission failed (HTTP {response.status_code}): {response.text}"
    #         )
    #     ticket_id: str = response.json()["id"]
    #     return ticket_id

    # def _poll_foldseek_job(
    #     self,
    #     ticket_id: str,
    #     poll_interval: int = 10,
    #     timeout: int = 600,
    # ) -> None:
    #     """
    #     Polls the Foldseek API until the job completes or times out.

    #     Parameters
    #     ----------
    #     ticket_id : str
    #         Ticket ID returned by :meth:`_submit_foldseek_job`.
    #     poll_interval : int, optional
    #         Seconds between status checks. Defaults to 10.
    #     timeout : int, optional
    #         Maximum total wait time in seconds. Defaults to 600.

    #     Raises
    #     ------
    #     TimeoutError
    #         If the job does not complete within ``timeout`` seconds.
    #     RuntimeError
    #         If the server reports an ``ERROR`` status.
    #     """
    #     try:
    #         import requests
    #     except ImportError:
    #         raise ImportError("requests is required for Foldseek searches. Install it with: pip install requests")

    #     elapsed = 0
    #     while elapsed < timeout:
    #         resp = requests.get(f"{self._FOLDSEEK_API}/ticket/{ticket_id}", timeout=30)
    #         resp.raise_for_status()
    #         status = resp.json().get("status", "")
    #         if status == "COMPLETE":
    #             return
    #         if status == "ERROR":
    #             raise RuntimeError(f"Foldseek job {ticket_id} reported ERROR status.")
    #         time.sleep(poll_interval)
    #         elapsed += poll_interval

    #     raise TimeoutError(
    #         f"Foldseek job {ticket_id} did not complete within {timeout}s."
    #     )

    # def _download_foldseek_results(
    #     self,
    #     ticket_id: str,
    #     out_dir: Path,
    #     protein_id: str,
    # ) -> Path:
    #     """
    #     Downloads and decompresses Foldseek results for one protein.

    #     Parameters
    #     ----------
    #     ticket_id : str
    #         Completed ticket ID.
    #     out_dir : Path
    #         Directory where the raw TSV is written.
    #     protein_id : str
    #         Used as the output filename stem: ``{protein_id}.tsv``.

    #     Returns
    #     -------
    #     Path
    #         Path to the written TSV file.

    #     Raises
    #     ------
    #     RuntimeError
    #         If the download request fails.
    #     """
    #     try:
    #         import requests
    #     except ImportError:
    #         raise ImportError("requests is required for Foldseek searches. Install it with: pip install requests")

    #     import io
    #     import tarfile

    #     resp = requests.get(
    #         f"{self._FOLDSEEK_API}/result/download/{ticket_id}",
    #         timeout=120,
    #         stream=True,
    #     )
    #     if resp.status_code != 200:
    #         raise RuntimeError(
    #             f"Failed to download Foldseek results for {protein_id} "
    #             f"(HTTP {resp.status_code}): {resp.text}"
    #         )

    #     tsv_path = out_dir / f"{protein_id}.tsv"
    #     raw_bytes = resp.content

    #     # The API returns a tar.gz archive containing one or more TSV files
    #     try:
    #         with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tar:
    #             tsv_lines: List[str] = []
    #             for member in tar.getmembers():
    #                 if member.name.endswith(".tsv") or member.name.endswith(".m8"):
    #                     f = tar.extractfile(member)
    #                     if f is not None:
    #                         tsv_lines.append(f.read().decode("utf-8"))
    #             content = "\n".join(tsv_lines)
    #     except tarfile.TarError:
    #         # Fallback: server returned plain TSV
    #         content = raw_bytes.decode("utf-8")

    #     tsv_path.write_text(content, encoding="utf-8")
    #     return tsv_path

    # def run_foldseek_search(
    #     self,
    #     protein_ids: List[str],
    #     monomer_cif_dir: Union[str, Path],
    #     databases: List[str] = ("afdb-swissprot", "pdb100"),  # type: ignore[assignment]
    #     top_n: int = 10,
    #     evalue_cutoff: float = 1e-3,
    #     output_dir: Optional[Union[str, Path]] = None,
    #     best_plddt: bool = False,
    #     poll_interval: int = 10,
    #     timeout: int = 600,
    #     mode: str = "3diaa",
    # ) -> pd.DataFrame:
    #     """
    #     Searches each protein's monomer structure against Foldseek databases.

    #     Operates standalone — does not require :attr:`interactome_data` to be loaded.
    #     For each protein in ``protein_ids`` the method locates the corresponding
    #     CIF file in ``monomer_cif_dir``, submits it to the Foldseek web API,
    #     polls until completion, downloads results, and aggregates a summary.

    #     The folder layout expected inside ``monomer_cif_dir`` mirrors the output of
    #     AF3/Boltz2/ColabFold in single-protein mode::

    #         monomer_cif_dir/
    #         ├── proteinA/
    #         │   ├── proteinA_model_0.cif
    #         │   └── proteinA_model_1.cif
    #         └── proteinB/
    #             └── proteinB_model_0.cif

    #     Parameters
    #     ----------
    #     protein_ids : list of str
    #         Protein identifiers to search. Each must have a matching subdirectory
    #         inside ``monomer_cif_dir``.
    #     monomer_cif_dir : str or Path
    #         Root directory containing one subdirectory per protein, each with
    #         ``*model*.cif`` files.
    #     databases : list of str, optional
    #         Foldseek databases to search. Defaults to
    #         ``["afdb-swissprot", "pdb100"]``.
    #         Other available options: ``"afdb50"``, ``"afdb-proteome"``,
    #         ``"mgnify_esm30"``, ``"gmgcl_id"``.
    #     top_n : int, optional
    #         Maximum number of hits to retain per protein in the summary.
    #         Defaults to 10.
    #     evalue_cutoff : float, optional
    #         Hits with e-value above this threshold are discarded. Defaults to 1e-3.
    #     output_dir : str or Path, optional
    #         Root output directory. Defaults to :attr:`output_path`.
    #         Raw TSVs are written to ``{output_dir}/foldseek_results/``.
    #         The summary CSV is written to ``{output_dir}/foldseek_summary.csv``.
    #     best_plddt : bool, optional
    #         If ``False`` (default), uses the model with the lowest index
    #         (``model_0`` first, by sorted filename). If ``True``, parses all
    #         available models and selects the one with the highest mean pLDDT —
    #         useful when multiple models per protein are present.
    #     poll_interval : int, optional
    #         Seconds between Foldseek status checks. Defaults to 10.
    #     timeout : int, optional
    #         Maximum seconds to wait for a single job. Defaults to 600.
    #     mode : str, optional
    #         Foldseek search mode. ``"3diaa"`` (default) or ``"tmalign"``.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Summary DataFrame with columns:
    #         ``protein_id``, ``rank``, ``target``, ``fident``, ``alnlen``,
    #         ``evalue``, ``bits``, ``qstart``, ``qend``, ``tstart``, ``tend``.
    #         Also written to ``{output_dir}/foldseek_summary.csv``.

    #     Raises
    #     ------
    #     FileNotFoundError
    #         If no CIF file is found for a given protein ID.
    #     RuntimeError
    #         If the Foldseek API returns an unexpected error.
    #     TimeoutError
    #         If a job exceeds ``timeout`` seconds.
    #     """
    #     databases = list(databases)
    #     base_dir = Path(output_dir) if output_dir is not None else self.output_path
    #     raw_dir = base_dir / "foldseek_results"
    #     raw_dir.mkdir(parents=True, exist_ok=True)
    #     summary_csv = base_dir / "foldseek_summary.csv"

    #     cif_root = Path(monomer_cif_dir)
    #     summary_rows: List[Dict[str, Any]] = []

    #     _RAW_COLS = [
    #         "query", "target", "fident", "alnlen", "mismatch",
    #         "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bits",
    #     ]

    #     for protein_id in protein_ids:
    #         protein_dir = cif_root / protein_id
    #         if not protein_dir.exists():
    #             logger.warning(f"No directory found for '{protein_id}' in {cif_root}. Skipping.")
    #             continue

    #         cif_candidates = sorted(protein_dir.glob("*model*.cif"))
    #         if not cif_candidates:
    #             logger.warning(f"No CIF files found for '{protein_id}'. Skipping.")
    #             continue

    #         # Select which model to submit
    #         if best_plddt and len(cif_candidates) > 1:
    #             best_cif: Optional[Path] = None
    #             best_score = -1.0
    #             for candidate in cif_candidates:
    #                 try:
    #                     # engine unknown here — try all parsers
    #                     for eng in ("af3", "boltz", "colabfold"):
    #                         try:
    #                             stats = InteractomeProcessor._extract_monomer_plddt(candidate, eng)
    #                             score = stats.get("plddt_mean", -1.0)
    #                             if not np.isnan(score) and score > best_score:
    #                                 best_score = score
    #                                 best_cif = candidate
    #                             break
    #                         except Exception:
    #                             continue
    #                 except Exception:
    #                     continue
    #             cif_path = best_cif if best_cif is not None else cif_candidates[0]
    #         else:
    #             cif_path = cif_candidates[0]

    #         logger.info(f"Submitting {protein_id} ({cif_path.name}) to Foldseek...")

    #         try:
    #             cif_content = cif_path.read_text(encoding="utf-8")
    #             ticket_id = self._submit_foldseek_job(cif_content, databases, mode=mode)
    #             self._poll_foldseek_job(ticket_id, poll_interval=poll_interval, timeout=timeout)
    #             tsv_path = self._download_foldseek_results(ticket_id, raw_dir, protein_id)
    #         except (RuntimeError, TimeoutError, OSError) as exc:
    #             logger.error(f"Foldseek search failed for '{protein_id}': {exc}")
    #             continue

    #         # Parse raw TSV
    #         try:
    #             df_raw = pd.read_csv(
    #                 tsv_path,
    #                 sep="\t",
    #                 header=None,
    #                 names=_RAW_COLS,
    #                 comment="#",
    #             )
    #         except Exception as exc:
    #             logger.warning(f"Could not parse TSV for '{protein_id}': {exc}")
    #             continue

    #         df_filtered = df_raw[df_raw["evalue"] <= evalue_cutoff].copy()
    #         df_top = df_filtered.sort_values("evalue").head(top_n).reset_index(drop=True)
    #         df_top.insert(0, "protein_id", protein_id)
    #         df_top.insert(1, "rank", range(1, len(df_top) + 1))

    #         keep_cols = [
    #             "protein_id", "rank", "target", "fident", "alnlen",
    #             "evalue", "bits", "qstart", "qend", "tstart", "tend",
    #         ]
    #         summary_rows.append(df_top[[c for c in keep_cols if c in df_top.columns]])
    #         logger.info(
    #             f"  {protein_id}: {len(df_top)} hits retained "
    #             f"(e-value ≤ {evalue_cutoff}, top {top_n})."
    #         )

    #     if summary_rows:
    #         summary_df = pd.concat(summary_rows, ignore_index=True)
    #     else:
    #         summary_df = pd.DataFrame(columns=[
    #             "protein_id", "rank", "target", "fident", "alnlen",
    #             "evalue", "bits", "qstart", "qend", "tstart", "tend",
    #         ])

    #     summary_df.to_csv(summary_csv, index=False)
    #     logger.info(f"Foldseek summary saved to {summary_csv}")
    #     return summary_df
