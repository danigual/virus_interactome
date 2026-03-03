# HAdV-5 Interactome Analysis Pipeline

Complete description of the pipeline executed for Human Adenovirus 5 (HAdV-5), from proteome preparation to downstream structural analysis. Implemented in the `virus_interactome` package around the four core classes: `ProteomeManager`, `InteractomeWriter`, `InteractomeRunner`, and `InteractomeProcessor` / `InteractomeAnalyzer`.

---

## Stage 0 — Proteome Preparation (`ProteomeManager`)

**Input:** FASTA file of the viral proteome.
`HAdV5_AC_000008_1_modified.fa`

**Actions:**
1. Load sequences from FASTA (`ProteomeManager(fasta_path)`).
2. Validate residues — only standard amino acids are accepted; invalid sequences are quarantined in `invalid_sequences`.
3. (Optional) Compute pairwise sequence identity matrix to detect high-similarity pairs (`high_similarity_pairs`) that may inflate the interactome with trivial/redundant predictions.
4. (Optional) Compute per-protein physicochemical properties (MW, pI, instability index) via BioPython.

**Key output:** A `ProteomeManager` instance with a clean `sequences` dict `{id: aa_sequence}`.

---

## Stage 1 — Input Generation (`InteractomeWriter`)

**Goal:** Generate one input file per protein pair (heterodimer) to be submitted to a folding engine.

```python
writer = InteractomeWriter(proteome_a="HAdV5_AC_000008_1_modified.fa")
writer.write_interactome_jobs(
    engine="af3",           # or "boltz2"
    output_dir="2_AF/input/",
    mode="intra_pairs",     # all unique (A, B) pairs within the proteome
)
```

**Internal logic:**
- `generate_intra_pairs()` — `combinations(ids, 2)`, canonical ordering (alphabetical) to avoid duplicates `(A,B)` vs `(B,A)`.
- For each pair, builds a `seq_list = [(idA, seqA, countA), (idB, seqB, countB)]`.
- Residue count validation: pairs exceeding the engine limit (`af3_threshold=5000 aa` or `boltz_threshold=1600 aa`) are logged/skipped.
- Writes one `.json` (AF3) or `.yaml` (Boltz2) per pair following the naming convention `{idA}__{idB}`.
- Saves an `index.csv` with metadata: `engine, mode, name, idA, idB, countA, countB, total_residues, warnings, file_path`.

**Also supports:**
- `mode="homomers"` — generates `(protein, n_copies)` for n in `[nmin, nmax]`.
- `mode="single"` — monomeric folding.
- `mode="inter_pairs"` — all pairs between two different proteomes (virus-host).

---

## Stage 2 — Structure Prediction (External)

Prediction runs were submitted externally to the folding engines. The pipeline monitors and manages these runs via `InteractomeRunner`.

### AF3 (AlphaFold 3)
- Input: `.json` files generated in Stage 1.
- Output per pair: a subdirectory `{idA}__{idB}/` containing up to **10 models** in `.cif` format (`*model_0.cif` … `*model_9.cif`).

### Boltz2
- Input: `.yaml` files.
- Output per pair: a subdirectory with up to **5 models** in `.cif` format.

### Run monitoring (`InteractomeRunner.check_run`)
```python
runner = InteractomeRunner(path_of_inputs="2_AF/input/", path_of_outputs="2_AF/output/", mode="af3")
status_df = runner.check_run(expected_models=10)
# Returns DataFrame: PPI | num_chain | num_aa | num_models | status
# status ∈ {COMPLETED, RUNNING, PENDING, FAILED}
```

Missing/failed jobs can be re-queued with `runner.write_missing_jobs()`.

---

## Stage 3 — Per-Model Processing (`InteractomeProcessor`)

**Goal:** Parse every `.cif` model, compute all quality metrics, cluster the PAE interface matrix, and save consolidated CSVs. Runs in parallel via `ProcessPoolExecutor`.

```python
from glob import glob
model_files = glob("2_AF/output/**/*model*.cif", recursive=True)

processor = InteractomeProcessor(model_list=model_files, engine="af3")
processor.process_models(output_path="3_results/", max_workers=8)
```

### Per-model pipeline (`process_ppi`)

For each `.cif` file:

1. **Metadata parsing** — extracts `PPI` id and model number from the directory/filename convention `{idA}__{idB}/..._model_N.cif`.

2. **Full data loading** — engine-specific parsers (`process_full_data_af3`, `process_full_data_boltz`, `process_full_data_colabfold`) extract:
   - `pae` — full PAE matrix (N×N, Å).
   - `ca_plddts` — per-residue pLDDT scores.
   - `token_chain_ids` — chain assignment per residue.
   - `chain_boundaries_by_res` — chain boundary indices.
   - `iptm`, `ptm` — global confidence scalars.
   - `iptm_chain_pair` — per-chain-pair ipTM matrix.
   - `msa_depth`, `msa_coverage` — MSA statistics.

3. **Plotting (per model):**
   - pLDDT profile → `*_plddt.png`
   - PAE matrix with chain delimiters → `*_pae.png`

4. **Interface metrics** (only for heterodimers, i.e., 2-chain models):
   - `calculate_all_metrics()` from `metrics.py` — computes:
     - `ipSAE_AB`, `ipSAE_d0_dom_AB` — interface SAE scores (TM-score-like, symmetrized A↔B).
     - `pDockQ2_AB` — docking quality score.
     - `pLDDT_mean` — mean pLDDT across the complex.
     - `ipTM`, `pTM_chain_A`, `pTM_chain_B`.

5. **PAE interface clustering (DBSCAN):**
   - Extracts the inter-chain PAE submatrix (A→B and B→A, then symmetrized by mean).
   - Thresholds low-PAE regions (default < 15 Å) and runs DBSCAN (`eps=10.0`, `min_samples=5`).
   - `cluster_info()` computes per-cluster geometry: `x_len`, `y_len`, `center_x`, `center_y`, `cluster_ratio` (aspect ratio = max_side / min_side).
   - Cluster plot → `*_cluster.png`

6. **Resume logic** — if `interactome_data.csv` already exists, already-processed folders are skipped.

### Outputs
| File | Content |
|---|---|
| `interactome_data.csv` | One row per model: `PPI, ORF_A, ORF_B, Folder, Model_num, ipTM, pTM, ipSAE_AB, ipSAE_d0_dom_AB, pDockQ2_AB, pLDDT_mean, msa_depth, msa_coverage, …` |
| `clusters_data.csv` | One row per PAE cluster: `PPI, model_num, path, cluster_id, num_points, x_len, y_len, x_min, x_max, y_min, y_max, center_x, center_y, cluster_ratio` |

---

## Stage 4 — Analysis and Prioritization (`InteractomeAnalyzer`)

**Goal:** Load the processed CSVs and apply biological filters to identify high-confidence interactions and candidate peptide-binding sites.

```python
analyzer = InteractomeAnalyzer(output_path="4_analysis/")
analyzer.interactome_path = "3_results/interactome_data.csv"
analyzer.cluster_path    = "3_results/clusters_data.csv"
```

### 4.1 Confidence Tier Classification

```python
tiered_df = analyzer.get_confidence_tiers(
    ipsae_threshold=0.5,
    pdockq2_threshold=0.23,
    msa_threshold=20
)
```

| Tier | Criteria |
|---|---|
| **Tier 1 — High Confidence** | ipSAE > 0.5 AND pDockQ2 > 0.23 AND msa_depth > 20 |
| **Tier 2 — Specific/Novel** | ipSAE > 0.5 AND pDockQ2 > 0.23 AND msa_depth ≤ 20 |
| **Tier 3 — Weak/Dynamic** | ipSAE > 0.5 AND pDockQ2 ≤ 0.23 |
| **Low Confidence** | ipSAE ≤ 0.5 |

### 4.2 Confidence Landscape Visualization

```python
analyzer.plot_confidence_landscape()      # static matplotlib scatter
analyzer.plot_interactive_landscape()     # interactive plotly HTML
```

- X-axis: pDockQ2 | Y-axis: ipSAE_d0_dom | Size: √msa_depth | Color: pLDDT (AF colorscale: orange < 50, yellow 50–70, cyan 70–90, blue > 90).
- Jitter applied to prevent point overlap for identical predictions.

### 4.3 Candidate Peptide-Protein Detection (`_get_candidate_clusters`)

Applies geometric filtering on the PAE clusters to identify elongated interfaces characteristic of linear peptide-binding motifs:

```python
# Called internally by run_full_pipeline / analyze_peptide_proteins_pairs
candidates = analyzer._get_candidate_clusters(
    cluster_ratio_threshold=7.0,   # aspect ratio x_len/y_len or y_len/x_len ≥ 7
    min_peptide_len=5              # minimum cluster dimension in residues
)
```

For each candidate, the shorter dimension → **Peptide**, the longer → **Binder** (receptor protein).

### 4.4 Peptide-Protein Structural Analysis (`analyze_peptide_proteins_pairs`)

For each candidate binder protein:

1. **Structure curation** — extract peptide + binder interface residues from the full `.cif` using `moleculekit` (`_curate_protein_peptide_models`); save trimmed `.pdb` to `prot_peptide/{binder}/filtered/`.
2. **Reference structure selection** — pick the model with the best pLDDT as reference for alignment (`_get_reference_structure_for_binder`); saved to `prot_peptide/{binder}/reference_{binder}.pdb`.
3. **Backbone alignment** — align all filtered models onto the reference by the binder's Cα atoms (`_create_binder_alignments`); save to `prot_peptide/{binder}/aligned/`.
4. **Spatial clustering (DBSCAN)** on peptide centroid coordinates across aligned models (`cluster_protein_peptides`): groups poses that converge on the same binding site.
5. **ChimeraX session generation** (`_create_chimera_session`) — writes a `.cxs` script that:
   - Loads the reference + all aligned models.
   - Colors models by spatial cluster.
   - Displays peptide centroids as spheres.
6. **Summary CSV** — `peptide_binder_info.csv` with cluster label, center coordinates, and interface residues per candidate.

### 4.5 Full Pipeline Entry Point

```python
analyzer.run_full_pipeline(
    ipsae_filter=0.5,          # informational threshold (does NOT filter structural step)
    cluster_ratio_threshold=7.0,
    min_peptide_len=5
)
```

---

## Data Flow Summary

```
HAdV5.fa
    │
    ▼
[ProteomeManager]
    │  clean sequences dict
    ▼
[InteractomeWriter]
    │  {idA}__{idB}.json / .yaml   +   index.csv
    ▼
[AF3 / Boltz2]  (external HPC/GPU)
    │  {idA}__{idB}/*model_N.cif
    ▼
[InteractomeRunner.check_run]   ← monitors, re-queues missing jobs
    │
    ▼
[InteractomeProcessor.process_models]
    │  interactome_data.csv   (metrics per model)
    │  clusters_data.csv      (PAE clusters per model)
    ▼
[InteractomeAnalyzer]
    ├── get_confidence_tiers()          → tiered_df
    ├── plot_confidence_landscape()     → landscape.png / .html
    └── analyze_peptide_proteins_pairs()
            ├── filtered PDBs
            ├── aligned PDBs
            ├── ChimeraX sessions
            └── peptide_binder_info.csv
```

---

## Key Metrics Reference

| Metric | Source | Meaning |
|---|---|---|
| `ipSAE_AB` | `metrics.py` | Interface Symmetrized Aligned Error — TM-score-like, range [0,1], higher = better interface alignment |
| `ipSAE_d0_dom_AB` | `metrics.py` | ipSAE with domain-length-normalized d0 — more discriminative for short interfaces |
| `pDockQ2_AB` | `metrics.py` | Docking quality score — proxy for physical binding plausibility |
| `pLDDT_mean` | `metrics.py` | Mean per-residue confidence across the complex |
| `ipTM` | AF3/Boltz output | Inter-chain predicted TM-score |
| `pTM` | AF3/Boltz output | Global predicted TM-score |
| `msa_depth` | AF3/Boltz output | Number of effective sequences in MSA — evolutionary support |
| `cluster_ratio` | `InteractomeProcessor` | PAE cluster aspect ratio — used to detect peptide-like linear interfaces |

---

## File and Directory Conventions

```
ppi_data/adeno/
├── 0_proteomes/curated/
│   └── HAdV5_AC_000008_1_modified.fa
├── 2_AF/
│   ├── input/
│   │   ├── index.csv
│   │   ├── af3_{idA}__{idB}.json
│   │   └── ...
│   └── output/
│       └── {idA}__{idB}/
│           ├── *model_0.cif ... *model_9.cif
│           ├── *_plddt.png
│           ├── *_pae.png
│           └── *_cluster.png
└── 3_results/
    ├── interactome_data.csv
    ├── clusters_data.csv
    └── prot_peptide/
        └── {binder}/
            ├── reference_{binder}.pdb
            ├── filtered/
            └── aligned/
```
