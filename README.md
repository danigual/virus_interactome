# Virus Interactome
A Python-based toolkit for the analysis and visualization of virus-host protein-protein interaction (PPI) networks.

---

## 🧬 Overview

`virus_interactome` is a Python package designed to help researchers process complex viral interaction data. It provides a structured workflow to identify key host proteins targeted by viral factors, analyze network topology, and visualize the interface between pathogens and host cells.

## ✨ Key Features

* **Data Wrangling:** Automated cleaning and standardization of PPI datasets.
* **Network Analysis:** Compute centrality metrics (Degree, Betweenness, Eigenvector) to find viral "hubs."
* **Module Detection:** Identify functional clusters within the host-virus interactome.
* **Visualization:** Integration with `matplotlib` and `networkx` for high-quality network diagrams.

## 🚀 Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/PabloHNieto/virus_interactome.git](https://github.com/PabloHNieto/virus_interactome.git)
cd virus_interactome
pip install -e .
```

## 🛠️ Quick Start
The library is organized into specialized classes that handle different stages of the interactome pipeline:

1. **Proteome**: manages sequence data and protein metadata (e.g., UniProt IDs, sequences).
2. **InteractomeProcessor**: handles data normalization, filtering of low-confidence hits, and format conversion.
3. **InteractomeRunner**: the execution engine that maps interactions onto the proteome and builds the graph.
4. **InteractomeWritter**: exports processed interactomes into standard formats (CSV, JSON, or Cytoscape-compatible files).
5. **InteractomeAnalyzer**: performs statistical and topological analysis, such as degree centrality and cluster detection.

```python
from virus_interactome import (
    Proteome, 
    InteractomeProcessor, 
    InteractomeRunner, 
    InteractomeAnalyzer
)

# 1. Initialize Proteome data
proteome = Proteome(fasta_path="path/to/species.fasta")

# 2. Process raw interaction data
processor = InteractomeProcessor(raw_data="raw_interactions.csv")
clean_data = processor.clean_and_filter()

# 3. Run the interactome construction
runner = InteractomeRunner(proteome=proteome, interactions=clean_data)
interactome_result = runner.execute()

# 4. Analyze the resulting network
analyzer = InteractomeAnalyzer(interactome_result)
hubs = analyzer.calculate_centrality()
```
