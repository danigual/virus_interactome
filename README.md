# virus_interactome

A Python toolkit for high-throughput structural modeling and analysis of viral protein-protein interaction (PPI) networks. It covers both **intra-viral** (virus vs. virus) and **virus-host** interactomes, bridging proteome sequence data and structural prediction outputs into a quantitative interaction landscape.

Supported prediction engines: **AlphaFold 3**, **Boltz-2**, and **ColabFold**.

---

## Installation

```bash
git clone https://github.com/PabloHNieto/virus_interactome.git
cd virus_interactome
pip install -e .
```

**Dependencies** (installed automatically): `biopython`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `moleculekit`, `PyYAML`, `tqdm`.

> Requires Python ≥ 3.9. A dedicated conda environment is recommended due to the `moleculekit` dependency.

---

## Pipeline Overview

The package is organized around four steps:

1. **Proteome management** — load and validate FASTA files, compute physicochemical properties, calculate sequence identity matrices.
2. **Job generation** — produce engine-ready input files (AF3 `.json`, Boltz-2 `.yaml`, ColabFold `.fasta`) for all-vs-all intra-viral pairs, virus-host pairs, homomers, or single chains.
3. **Result processing** — parse `.cif` model outputs in parallel, extract structural confidence metrics (ipTM, pTM, pDockQ2, ipSAE, LIS family), and run PAE-based interface clustering.
4. **Interactome analysis** — classify interactions by confidence tier, rank and filter by any metric, export network edge lists, and run peptide-protein structural ensemble analysis.

---

## Quick Start

### 1. Generate prediction jobs

```python
from virus_interactome import ProteomeManager, InteractomeWriter

virus = ProteomeManager("virus.fasta")

# All-vs-all intra-viral pairs for AlphaFold 3
writer = InteractomeWriter(proteome_a=virus)
writer.write_interactome_jobs(engine="af3", output_dir="jobs/", mode="intra_pairs")

# Virus-host pairs for ColabFold
host = ProteomeManager("host.fasta")
writer = InteractomeWriter(proteome_a=virus, proteome_b=host)
writer.write_interactome_jobs(engine="colabfold", output_dir="jobs_host/", mode="inter_pairs")
```

### 2. Monitor and run jobs

```python
from virus_interactome import InteractomeRunner

runner = InteractomeRunner(
    path_of_inputs="jobs/",
    path_of_outputs="results/",
    mode="af3"
)
status = runner.check_run()
print(status)
```

### 3. Process structural outputs

```python
from virus_interactome import InteractomeProcessor

processor = InteractomeProcessor(model_list=runner.inputs, engine="af3")
processor.process_models(output_path="analysis/")
# Produces: analysis/interactome_data.csv, analysis/clusters_data.csv
```

### 4. Analyze the interactome

```python
from virus_interactome import InteractomeAnalyzer

analyzer = InteractomeAnalyzer(output_path="analysis/")

# Classify interactions into confidence tiers
tiers = analyzer.get_confidence_tiers()

# Rank by best iLIS score
top = analyzer.get_top_interactions(metric="Best_iLIS", top_n=20)

# Export as network edge list for Cytoscape
analyzer.export_to_network(output_format="cytoscape", output_path="network.csv")

# Visualize the confidence landscape
analyzer.plot_confidence_landscape(output_path="analysis/")
```

---

## Confidence Metrics

Structural confidence is assessed through a combination of metrics:

| Metric | Description |
|---|---|
| `ipTM` / `pTM` | Inter-chain and global TM-score confidence (engine output) |
| `ipSAE` | Interface Symmetrized Aligned Error — primary interaction quality score |
| `pDockQ2` | Predicted docking quality v2 — physical binding plausibility |
| `LIS` / `LIA` | Local Interaction Score / Area (Kim et al. 2024) |
| `cLIS` / `cLIA` | Contact-filtered LIS/LIA (Cβ–Cβ distance ≤ 8 Å) |
| `iLIS` / `iLIA` | `sqrt(LIS × cLIS)` — geometric mean, reduces false positives (Kim et al. 2025) |
| `Best_LIS` / `Best_iLIS` | `max(AB, BA)` — used for final tier classification |

**Literature-validated thresholds:**
- High-confidence (dual): `Best_LIS ≥ 0.203` AND `Best_LIA ≥ 3432`
- High-confidence (single metric): `Best_iLIS ≥ 0.223`

---

## License

MIT
