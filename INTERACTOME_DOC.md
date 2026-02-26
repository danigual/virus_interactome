# Interactome Module Documentation (`interactome.py`)

The `interactome.py` file is the core of the `virus_interactome` package, containing four main classes that manage the entire pipeline from job creation to complex structural analysis.

---

## 1. `InteractomeWriter`
**Purpose:** Manages the configuration and generation of input files for protein folding engines.

### Key Features:
- **Modes:** Supports `intra` (within one proteome) and `inter` (between two proteomes).
- **Pair Generation:**
  - `generate_intra_pairs()`: Unique heteromeric pairs within one proteome.
  - `generate_inter_pairs()`: Cartesian product between two proteomes.
  - `generate_homo_mers(nmin, nmax)`: Oligomeric states (2 to 6 by default).
  - `generate_single_run()`: Monomers (stoichiometry = 1).
- **Engine Support:**
  - **AF3:** Generates `.json` inputs.
  - **Boltz2:** Generates `.yaml` inputs.
  - **ColabFold:** Generates `.fasta` or `.csv` batch inputs.
- **Validation:** `check_input()` validates sequences and residue counts against engine-specific thresholds (e.g., 5000 for AF3, 1600 for Boltz2).
- **Orchestration:** `write_interactome_jobs()` handles the full process of writing inputs and an `index.csv`.

---

## 2. `InteractomeRunner`
**Purpose:** Orchestrates execution and monitors the status of folding jobs.

### Key Features:
- **Status Tracking:** `check_run()` and `check_colabfold_run()` scan output directories for `.cif`/`.pdb` files and determines if jobs are `PENDING`, `RUNNING`, `COMPLETED`, or `FAILED`.
- **Error Handling:** `write_missing_jobs()` identifies non-completed jobs and copies their inputs to a separate folder for re-running.
- **Execution:**
  - `run_colabfold_fastas()`: Runs `colabfold_batch` on individual fasta files with parallel worker support.
  - `run_colabfold_csv()`: Launches a single batch job from a CSV.
- **Reporting:** `write_status()` exports a `JOB_STATUS.csv`.

---

## 3. `InteractomeProcessor`
**Purpose:** Analyzes output models, extracts metrics, and performs interface clustering.

### Key Features:
- **Engine Support:** Full support for **AF3**, **Boltz2**, and **ColabFold** outputs.
- **Metric Extraction:** `process_ppi()` extracts `ipTM`, `pTM`, chain-specific pTM, and interface metrics:
  - **pDockQ / pDockQ2**: Quality of the predicted complex.
  - **ipSAE**: Interface-specific Predicted Aligned Error.
  - **LIS**: Local Interaction Strength.
- **PAE Analysis:**
  - `cluster_pae()`: Uses **DBSCAN** to identify low-error contact regions in the PAE matrix.
  - `cluster_info()`: Calculates bounding box, centroids, and aspect ratios for interaction interfaces.
- **Visualization:** Generates plots for pLDDT, PAE matrices, and identified clusters (saved alongside models).
- **High-Throughput:** `process_models()` uses `ProcessPoolExecutor` for parallel analysis and supports "resume" logic via existing CSV data.

---

## 4. `InteractomeAnalyzer`
**Purpose:** Advanced post-processing for data management, path relocation, and structural ensemble analysis.

### Key Features:
- **Data Management:** Loads and validates `interactome_data.csv` and `clusters_data.csv`.
- **Path Relocation:** Allows updating model roots when moving analysis between machines.
- **Peptide-Protein Pipeline:** Specialized workflow to superimpose and cluster peptide binding sites on a reference binder structure.
- **ChimeraX Integration:** Generates `.cxc` and `.cxs` files for 3D visualization of interface clusters.

---

## 🛠 Project Status & Progress (Session Feb 26, 2026)

### Recent Achievements:
1.  **Adeno Interactome Analysis Completed**: Successfully processed 630 ColabFold heterodimer models on the remote machine.
    - Results located at: `/home/daniel/ppi_data_remote/adeno/3_analysis/2026_02_26_AF2.3/`
    - Output files: `interactome_data.csv` and `clusters_data.csv` are generated and verified.
2.  **ColabFold Integration**: Completed the pipeline for ColabFold. Updated `InteractomeProcessor` to support `.pdb` outputs and `_scores.json` files.
3.  **Codebase Stability**: Verified that `InteractomeProcessor` handles both CIF (AF3/Boltz) and PDB (ColabFold) formats seamlessly.

### Current Goals & Proposed Enhancements:
1.  **Refine ColabFold Functionality**:
    - **MSA Metrics**: Extract sequence coverage and `N_effective` from `.a3m` files to quantify evolutionary info.
    - **Per-Chain ipTM Estimation**: Implement a method to estimate per-chain pair `ipTM` from ColabFold's PAE matrix (since it's not provided in the raw output).
    - **Model Selection**: Add an option to `process_models()` to only keep the "top-ranked" model per PPI based on a metric (e.g., `pDockQ2`).
    - **Log Analysis**: Parse ColabFold logs for recycling convergence and template usage.
2.  **Robustness Update**: Update `process_ppi` to handle PPI IDs without the standard `__` separator or for monomers.

---

## 🔄 Session Continuity (Working Instructions)

If a session is interrupted, follow these steps to resume:
1.  **Data Location**: All Adeno models are in `/home/daniel/ppi_data_remote/adeno/5_AF2_multimer/output/`.
2.  **Analysis Location**: Existing results are in `/home/daniel/ppi_data_remote/adeno/3_analysis/2026_02_26_AF2.3/`.
3.  **To Resume Processing**:
    ```python
    from virus_interactome import InteractomeProcessor, InteractomeRunner
    # Detect all output folders
    runner = InteractomeRunner(path_of_inputs="input_dir/", path_of_outputs="output_dir/", mode="colabfold")
    # Process with resume logic
    processor = InteractomeProcessor(model_list=runner.outputs, engine="colabfold")
    processor.process_models(output_path="analysis_dir/")
    ```
4.  **Key dependencies**: Always verify `moleculekit` and `sklearn.cluster.DBSCAN` are available.

---

## Key Dependencies
- `moleculekit`: Structural handling and alignment.
- `sklearn.cluster.DBSCAN`: Interface clustering.
- `pandas`, `numpy`: Data manipulation.
- `concurrent.futures`: Multi-threaded processing.
