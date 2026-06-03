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

**Dependencies** (installed automatically): `biopython`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `moleculekit`, `networkx`, `requests`, `PyYAML`, `tqdm`.

> Requires Python ≥ 3.9. A dedicated conda environment is recommended due to the `moleculekit` dependency.

---

## Pipeline Overview

The package is organized around four steps:

1. **Proteome management** — load and validate FASTA files, compute physicochemical properties, calculate sequence identity matrices.
2. **Job generation** — produce engine-ready input files (AF3 `.json`, Boltz-2 `.yaml`, ColabFold `.fasta`) for all-vs-all intra-viral pairs, virus-host pairs, homomers, or single chains. Includes a **pooled ColabFold** mode (`PoolDesigner`) that partitions a proteome into multi-protein pools to maximize GPU utilization.
3. **Result processing** — parse `.cif` model outputs in parallel, extract structural confidence metrics (ipTM, pDockQ2, ipSAE, LIS family), run PAE-based interface clustering, and compute per-residue contact indices. Supports standard pair-wise outputs and pooled ColabFold outputs. Monomer pLDDT extraction also available.
4. **Interactome analysis** — classify interactions by confidence tier, rank and filter by any metric, export network edge lists, compute graph-level topology metrics (degree, betweenness, closeness, eigenvector centrality), cross-validate against experimental PPI databases, and run peptide-protein structural ensemble analysis.

---

## Quick Start

### 1. Generate prediction jobs

```python
from virus_interactome import ProteomeManager, InteractomeWriter, PoolDesigner

virus = ProteomeManager("virus.fasta")

# All-vs-all intra-viral pairs for AlphaFold 3
writer = InteractomeWriter(proteome_a=virus)
writer.write_interactome_jobs(engine="af3", output_dir="jobs/", mode="intra_pairs")

# Virus-host pairs for ColabFold
host = ProteomeManager("host.fasta")
writer = InteractomeWriter(proteome_a=virus, proteome_b=host)
writer.write_interactome_jobs(engine="colabfold", output_dir="jobs_host/", mode="inter_pairs")

# Pooled ColabFold — fits multiple proteins per GPU run (Todor et al. 2026)
writer_pooled = InteractomeWriter(proteome_a=virus)
writer_pooled.write_pooled_jobs(engine="colabfold", output_dir="jobs_pooled/", token_limit=4000)
# Produces: colabfold_pooled_input.csv + pool_manifest.csv
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
from virus_interactome.utils import reorganize_colabfold_outputs

# Standard pair-wise processing
processor = InteractomeProcessor(model_list=runner.inputs, engine="af3")
processor.process_models(output_path="analysis/")
# Produces: analysis/interactome_data.csv, analysis/clusters_data.csv

# Pooled ColabFold processing (reorganize outputs first)
reorganize_colabfold_outputs("cf_output/")
InteractomeProcessor.process_pooled(
    pool_manifest="jobs_pooled/pool_manifest.csv",
    cf_output_dir="cf_output/",
    output_path="analysis/"
)

# Monomer pLDDT extraction
processor.process_monomers(output_path="analysis/")
# Produces: analysis/monomer_data.csv
```

### 4. Analyze the interactome

```python
from virus_interactome import InteractomeAnalyzer, DatabaseClient

analyzer = InteractomeAnalyzer(output_path="analysis/")

# Classify interactions into confidence tiers
tiers = analyzer.get_confidence_tiers()

# Rank by ipSAE
top = analyzer.get_top_interactions(metric="ipSAE_AB", top_n=20)

# Export as network edge list for Cytoscape / Gephi
analyzer.export_to_network(output_format="cytoscape", output_path="network.csv")

# Graph topology — degree, betweenness, closeness, eigenvector centrality
network_df = analyzer.compute_network_properties(weight_col="ipSAE_AB")
analyzer.plot_network(network_df, color_by="betweenness", size_by="degree")

# Cross-validate against an experimental PPI database
known = DatabaseClient.from_file("experimental_ppis.csv", col_a="protein_A", col_b="protein_B")
validated = analyzer.validate_against_database(known)
print(analyzer.validation_summary(validated, known_ppis=known))

# Visualize the confidence landscape
analyzer.plot_confidence_landscape(output_path="analysis/")
```

### 5. Structural homology search with Foldseek

```python
from virus_interactome import FoldseekClient

client = FoldseekClient()
results = client.search(
    cif_path="models/proteinA_model.cif",
    databases=["pdb100", "afdb50"],
    out_dir="foldseek_results/"
)
```

---

## Confidence Metrics

Structural confidence is assessed through a combination of metrics:

| Metric | Description |
|---|---|
| `ipTM` / `pTM` | Inter-chain and global TM-score confidence (engine output) |
| `ipSAE` | Interface Symmetrized Aligned Error — primary interaction quality score |
| `ipSAE_d0dom` | ipSAE with domain-length-normalized d0 — preferred for ranking |
| `pDockQ2` | Predicted docking quality v2 — physical binding plausibility |
| `LIS` / `LIA` | Local Interaction Score / Area (Kim et al. 2024) |
| `Best_LIS` / `Best_LIA` | `max(AB, BA)` — used for tier classification |
| `msa_depth` | Effective MSA depth — evolutionary support (NaN for AF3/Boltz) |
| `cluster_ratio` | PAE cluster aspect ratio — high values suggest linear/peptide interfaces |

**Literature-validated thresholds (tier classification):**
- High-confidence (dual): `Best_LIS ≥ 0.203` AND `Best_LIA ≥ 3432`
- ipSAE primary cutoffs: see `get_confidence_tiers()` docstring

---

## References

- Dunbrack R.L. (2025). *Res ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it.* bioRxiv. https://doi.org/10.1101/2025.02.10.637595
- Kim D.W. et al. (2024). *Systematic assessment of protein-protein interaction prediction using AlphaFold-Multimer with LIS score.* bioRxiv. https://doi.org/10.1101/2024.02.19.580970
- Todor et al. (2026). *Pooled ColabFold for large-scale interactome screening.* (pooled job design methodology)
- Qi Y. et al. (2026). *Atlas of predicted protein complex structures across kingdoms.* Nature Communications. https://doi.org/10.1038/s41467-026-70884-4

---

## License

MIT
