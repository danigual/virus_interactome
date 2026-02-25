# Virus Interactome 🧬

A high-throughput Python toolkit for the generation, execution, and structural analysis of viral protein-protein interaction (PPI) networks, covering both **virus-host** and **internal viral (intra-interactome)** systems.

`virus_interactome` bridges the gap between proteomic sequence data and structural "interactomics," providing a complete pipeline to model and analyze how viral factors interact with each other and with host cellular machinery.

---

## 🚀 Core Architecture

The package is organized into four specialized pillars that manage the interactome lifecycle:

### 1. **Proteome Management** (`ProteomeManager`)
* **Data Wrangling:** Automated cleaning, sequence validation, and standardization of FASTA datasets.
* **Physicochemical Profiling:** Compute molecular weight, isoelectric point (pI), instability index, and aromaticity.
* **Identity Analysis:** Multi-processed sequence identity matrices to identify redundant proteins or high-similarity clusters.

### 2. **Job Orchestration** (`InteractomeWriter` & `InteractomeRunner`)
* **Flexible Modes:** 
    * **Intra-interactome:** Systematic analysis of interactions within a single proteome (e.g., all-vs-all viral proteins).
    * **Inter-interactome:** Mapping interactions between two different systems (e.g., virus vs. host).
* **Multi-Engine Support:** Generate native input formats for **AlphaFold 3** (.json), **Boltz2** (.yaml), and **ColabFold** (.fasta/.csv).
* **Stoichiometry:** Support for pairs, homomers (oligomers), and monomers.
* **Status Monitoring:** Real-time tracking of job states (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`).

### 3. **Structural Processing** (`InteractomeProcessor`)
* **Automated Extraction:** Parse `ipTM`, `pTM`, and chain-specific confidence scores from CIF/PDB and JSON outputs.
* **Advanced Metrics:** Calculation of high-confidence interaction markers:
    * **pDockQ & pDockQ2:** Predicted docking quality scores.
    * **ipSAE:** interface-specific Predicted Aligned Error.
    * **LIS:** Local Interaction Strength.
* **Interface Clustering:** Uses **DBSCAN** density-based clustering on the PAE matrix to identify and characterize specific interaction interfaces.

### 4. **Ensemble Analysis** (`InteractomeAnalyzer`)
* **Peptide-Protein Pipeline:** Specialized workflow for identifying and clustering peptide binding sites on larger protein surfaces.
* **Structural Alignment:** Automated superimposition of interaction ensembles based on high-confidence (pLDDT > 70) reference binder structures.
* **Visualization:** Automated generation of **ChimeraX** scripts (`.cxc`) and sessions (`.cxs`) for 3D analysis of binding site clusters.

---

## 📊 Visualization Suite

The package produces publication-ready diagrams:
* **PAE Heatmaps:** With automated chain boundary detection.
* **pLDDT Plots:** Color-coded by confidence bands.
* **Metric Distribution:** Boxplots and iPTM vs pTM scatterplots.
* **Cluster Visualization:** 2D projections of PAE interface clusters.

---

## 🛠️ Quick Start

### 1. Generate Jobs (Intra-interactome Example)
```python
from virus_interactome import ProteomeManager, InteractomeWriter

# Load viral proteome
virus = ProteomeManager("virus.fasta")

# Generate all-vs-all intra-viral pairs for Boltz2
writer = InteractomeWriter(proteome_a=virus)
writer.write_interactome_jobs(engine="boltz2", output_dir="jobs_intra/", mode="intra_pairs")
```

### 2. Process Results
```python
from virus_interactome import InteractomeProcessor, InteractomeRunner

# Check execution status
runner = InteractomeRunner(path_of_inputs="jobs_intra/", path_of_outputs="results_intra/", mode="boltz2")
status = runner.check_run()

# Process completed models and extract pDockQ/ipSAE
processor = InteractomeProcessor(model_list=runner.inputs, engine="boltz2")
processor.process_models(output_path="analysis/")
```

---

## 📦 Installation

```bash
git clone https://github.com/PabloHNieto/virus_interactome.git
cd virus_interactome
pip install -e .
```
