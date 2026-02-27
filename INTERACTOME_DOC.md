# Interactome Module Documentation (`interactome.py`)

The `interactome.py` file is the core of the `virus_interactome` package, containing four main classes that manage the entire pipeline from job creation to complex structural analysis.

---

## 🛠 Project Status & Progress (Session Feb 26, 2026 - FINAL SUMMARY)

### Recent Achievements:
1.  **Adeno Interactome Excellence Analysis**: 
    - Processed 630 heterodimers generated with ColabFold (**rank_001 models only**).
    - Successfully recalculated **manual pTM/ipTM** for all models, eliminating NaNs.
    - Integrated **MSA Metrics** (depth and coverage) into the CSV output.
2.  **InteractomeAnalyzer Upgrades**:
    - **Confidence Tiering**: Categorizes PPIs based on `ipSAE`, `pDockQ2`, and `msa_depth`.
    - **Interactive Landscape**: Plotly HTML plot (`adeno_interactive_exploration.html`) with AlphaFold coloring and hover tooltips.
    - **Structural Pipeline Robustness**: Fixed bugs related to empty chain B selections and pandas index mismatches.
3.  **Filtrado de Red de Excelencia**: 
    - Applied 10 criteria (ipTM, pTM, ipSAE, ipSAE_d0_dom, Cluster points/ratio).
    - **Result**: 84 PPIs passed the excellence threshold out of 630 candidates.
    - **Top Hits (Consensus 10/10)**: `pVI__protease` and `IX__hexon`.
    - **Key Findings**: Discovery of high-ratio interfaces (e.g., `52K__IVa2` with ratio 24.60).

---

## 🔄 Session Continuity (Working Instructions)

### 1. Data & Analysis Locations
- **Models**: `/home/daniel/ppi_data_remote/adeno/5_AF2_multimer/output/` (PDB rank_001 and .a3m files).
- **Results**: `/media/DATA/ppi_data/adeno/3_analysis/2026_02_26_AF2.3/`.
- **Top Candidates**: `analyzer_results/adeno_network_summary.csv` (84 Excellence Hits).

### 2. Excellence Criteria (>= Thresholds)
| ID | Criterion | Logic |
|---|---|---|
| C1-C3 | ipTM / pTM | ipTM >= 0.6 or (ipTM>=0.5 & pTM>=0.5) |
| C4-C5 | ipSAE | Global interface confidence >= 0.5 or 0.25 |
| C6-C7 | ipSAE_d0dom | Local interface confidence >= 0.4 or 0.5 |
| C8 | ipSAE_d0chn | Chain-normalized confidence >= 0.5 |
| C9-C10| Geometry | Cluster points >= 2500, Cluster Ratio >= 7.0 |

### 3. To Resume / Rerun Analysis
```bash
# 1. Update package
git pull origin main

# 2. Run Diagnostic & Pipeline (Tiers, Landscape, ChimeraX)
python analyze_colabfold_data.py

# 3. Generate Excellence Summary (The 84 Hits)
python generate_network_summary.py
```

### 4. Visualization
- Open `adeno_interactive_exploration.html` in a browser for interactive landscape.
- ChimeraX sessions are in `analyzer_results/prot_peptide/` (organized by Binder/ORF).

---

## Key Dependencies
- `moleculekit`: Structural handling and alignment.
- `sklearn.cluster.DBSCAN`: Interface clustering.
- `pandas`, `numpy`, `plotly`: Data manipulation and visualization.
- `concurrent.futures`: Multi-threaded processing.
