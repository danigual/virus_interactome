# Interactome Module Documentation (`interactome.py`)

The `interactome.py` file is the core of the `virus_interactome` package, containing four main classes that manage the entire pipeline from job creation to complex structural analysis.

---

## 🛠 Project Status & Progress (Session Feb 26, 2026 - FINAL UPDATE)

### Recent Achievements:
1.  **Adeno Interactome Excellence Analysis**: 
    - Processed 630 heterodimers generated with ColabFold.
    - Successfully recalculated **manual pTM/ipTM** for all models, eliminating NaNs in the final results.
    - Integrated **MSA Metrics** (depth and coverage) into the CSV output, providing evolutionary context for each pair.
2.  **InteractomeAnalyzer Upgrades**:
    - **Confidence Tiering**: Categorizes PPIs into Tiers based on `ipSAE`, `pDockQ2`, and `msa_depth`.
    - **Interactive Landscape**: Generated a Plotly HTML plot (`adeno_interactive_exploration.html`) with AlphaFold coloring and hover tooltips for the 630 models.
    - **Structural Pipeline Robustness**: Fixed bugs related to empty chain B selections and pandas index mismatches during structural superpositions.
3.  **Filtrado de Red de Excelencia**: 
    - Implemented a multi-criteria filtering script to identify Top-Hits based on 10 structural, physical, and evolutionary metrics (ipSAE, ipSAE_d0_dom, ipTM, Cluster Ratio, etc.).

---

## 🔄 Session Continuity (Working Instructions)

If a session is interrupted, follow these steps to resume:

1.  **Data Location**: 
    - All Adeno models are in `/home/daniel/ppi_data_remote/adeno/5_AF2_multimer/output/`.
    - Current analysis results: `/media/DATA/ppi_data/adeno/3_analysis/2026_02_26_AF2.3/`.
2.  **To Resume / Rerun Analysis**:
    - Run the `analyze_colabfold_data.py` script on the remote machine to generate Tiers and Landscape.
    - Run the **Network Summary script** (Multi-criteria filtering) to extract the most solid PPIs from the 630 pairs.
3.  **Filtering Thresholds (Excellence)**:
    - `ipSAE_d0_dom >= 0.4` and `cluster_ratio >= 7.0` are the current standards for high-confidence viral motives.
4.  **Key output files**:
    - `interactome_data.csv`: Global metrics (including manual pTM and MSA).
    - `clusters_data.csv`: Interface geometry and clustering.
    - `adeno_interactive_exploration.html`: Visualization of the confidence landscape.
    - `prot_peptide/`: Folder with ChimeraX sessions and aligned PDB models.

---

## Key Dependencies
- `moleculekit`: Structural handling and alignment.
- `sklearn.cluster.DBSCAN`: Interface clustering.
- `pandas`, `numpy`, `plotly`: Data manipulation and visualization.
- `concurrent.futures`: Multi-threaded processing.
