import time
import subprocess
import logging
import concurrent.futures
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

from .utils import load_json, load_boltz_input, reorganize_colabfold_outputs

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


