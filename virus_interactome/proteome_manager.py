from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from Bio import pairwise2, SeqIO
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pathlib import Path
from typing import Callable, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from moleculekit.molecule import Molecule
import concurrent.futures
from functools import partial
import os

from .utils import process_full_data_af3, process_full_data_boltz, process_full_data_colabfold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

class ProteomeManager:
    """
    Class to manage proteomes and prepare data for interactome analysis.
    """
    
    def __init__(self, fasta_file: str | None = None):
        """
        Initialize ProteomeManager. Optionally load a proteome from a FASTA file.

        Parameters
        ----------
        fasta_file : str or None
            Path to FASTA file. If None, no proteome is loaded.
        """
        self.sequences = {}
        self.invalid_sequences = {}
        self.high_similarity_pairs = []
        self.identity_matrix = None
        self.identity_table = None
        self.sequence_properties = None
        self._ids_cache: Optional[Tuple[str, ...]] = None
        self._order_mode: str = "insertion"
        self.model_info_extended = None
        self.model_info_extended = None

        if fasta_file:
            self.load_proteome(fasta_file)
        
        self._file_path = fasta_file

    @property
    def ids(self) -> Tuple[str, ...]:
        """
        Immutable tuple of valid protein IDs.
        Deterministic order: FASTA insertion order or alphabetical (configurable).
        """
        if self._ids_cache is None:
            self._ids_cache = tuple(sorted(self.sequences.keys()))
        return self._ids_cache

    @staticmethod
    def _check_sequence_validity(seq: str) -> bool:
        return all(residue in VALID_AMINO_ACIDS for residue in seq)
    
    @property
    def file_path(self) -> Optional[str]:
        return self._file_path
    
    @file_path.setter
    def file_path(self, fasta_file: str) -> None:
        self.load_proteome(fasta_file)

    @staticmethod
    def _get_orf_id_from_path(path: str, engine: str) -> Optional[str]:
        """Return the ORF ID encoded in a model file path (engine-specific)."""
        if engine.lower() in ("af3", "boltz"):
            return Path(path).parent.name
        elif engine.lower() == "colabfold":
            basename = os.path.basename(path)
            for part in basename.split("_"):
                if "ORF" in part:
                    return part
            if "_model_" in basename:
                return basename.split("_model_")[0]
            return basename
        return None

    @staticmethod
    def _compute_identity(args):
        from Bio import pairwise2
        i, j, seq1, seq2 = args
        alignments = pairwise2.align.globalxx(seq1, seq2)
        score = alignments[0].score
        max_len = max(len(seq1), len(seq2))
        return i, j, score / max_len
    
    def __str__(self) -> str:
        summary_data = self.summary()
        return (
            f"ProteomeManager Summary:\n"
            f"  Total sequences: {summary_data['total_sequences']}\n"
            f"  Total residues: {summary_data['total_residues']}\n"
            f"  Average length: {summary_data['average_length']}\n"
            f"  Min length: {summary_data['min_length']}\n"
            f"  Max length: {summary_data['max_length']}\n"
            f"  Invalid sequences: {summary_data['invalid_sequences']}\n"
            f"  High similarity pairs: {summary_data['high_similarity_pairs']}"
        )
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def seq_from_id(self, protein_id: str) -> str:
        """
        Retrieve the amino acid sequence for a given protein ID.

        Parameters
        ----------
        protein_id : str
            The protein ID to look up.

        Returns
        -------
        str
            The amino acid sequence corresponding to the protein ID.

        Raises
        ------
        KeyError
            If the protein ID is not found in the proteome.
        """
        if protein_id not in self.sequences:
            raise KeyError(f"Protein ID '{protein_id}' not found in proteome.")
        return self.sequences[protein_id]

    # -------------------------
    # 1. Loading and validation
    # -------------------------

    @staticmethod
    def normalize_fasta_headers(
        inputpath: str,
        outputpath: str,
        header_parser: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Rewrite FASTA headers using a custom or default parser.

        Reads *inputpath* line by line and replaces each header line with a
        standardised ``>{new_id}|{original_header}`` format. Sequence lines
        are written unchanged.

        Parameters
        ----------
        inputpath : str
            Path to the input FASTA file.
        outputpath : str
            Path for the cleaned output FASTA file.
        header_parser : callable, optional
            ``f(header: str) -> str`` where *header* is the full header string
            **without** the leading ``>``.  Must return the new protein ID.
            If *None*, the default NCBI parser is used: extracts the value of
            the ``protein=`` field and replaces spaces, dots and slashes with
            underscores.

        Raises
        ------
        FileNotFoundError
            If *inputpath* does not exist.
        ValueError
            If *header_parser* returns an empty string for a header.
        """
        if not Path(inputpath).is_file():
            raise FileNotFoundError(f"Input file not found: {inputpath}")

        def _default_ncbi_parser(header: str) -> str:
            start = header.find("protein=")
            if start == -1:
                return header.split()[0]
            end = header.find("]", start)
            raw = header[start + 8 : end] if end != -1 else header[start + 8 :]
            return raw.replace(" ", "_").replace(".", "_").replace("/", "_")

        parser = header_parser if header_parser is not None else _default_ncbi_parser

        with open(inputpath, "r") as inf, open(outputpath, "w") as outf:
            for line in inf:
                if line.startswith(">"):
                    header = line[1:].rstrip("\n")
                    new_id = parser(header)
                    if not new_id:
                        raise ValueError(
                            f"header_parser returned an empty ID for header: {header!r}"
                        )
                    outf.write(f">{new_id}|{header}\n")
                else:
                    outf.write(line)

    def load_proteome(self, fasta_file: str) -> dict:
        """
        Loads a FASTA file and extracts protein sequences into a dictionary.

        This method parses a FASTA file and builds a dictionary mapping protein IDs to their
        amino acid sequences. It uses the first segment of the ID (before the first '|') as the key.
        Duplicate IDs will trigger a warning and overwrite previous entries.

        Parameters
        ----------
        fasta_file : str
            Path to the FASTA file containing protein sequences.

        Returns
        -------
        dict
            Dictionary where keys are protein IDs and values are amino acid sequences.

        Raises
        ------
        FileNotFoundError
            If the FASTA file does not exist.
        UserWarning
            If duplicate protein IDs are found.
        """
        from Bio import SeqIO
        
        if fasta_file is None:
            return

        if not Path(fasta_file).is_file():
            logger.error(f"FASTA file not found: {fasta_file}")
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        proteome_dict = {}
        for protein in SeqIO.parse(fasta_file, "fasta"):
            short_id = protein.id.split('|')[0]

            if not self._check_sequence_validity(protein.seq):
                msg = f"Invalid amino acids in {short_id}. Sequence skipped."
                logger.warning(msg)
                self.invalid_sequences[short_id] = str(protein.seq)
                continue

            if short_id in proteome_dict:
                logger.warning(f"Duplicate protein ID '{short_id}' found. Overwriting previous entry.")
            proteome_dict[short_id] = str(protein.seq)

        self.sequences = proteome_dict
        self._file_path = fasta_file
        logger.info(f"Proteome loaded successfully with {len(proteome_dict)} proteins.")

        if self.sequences:
            logger.info("Computing sequence identity matrix...")
            self.compute_identity_matrix(n_jobs=10, similarity_threshold=0.95)
        # self.identity_matrix = similarity_matrix
        # self.identity_table = similarity_data
        # logger.info(f"Proteome loaded successfully with {len(proteome_dict)} proteins.")
        # self.sequence_properties = self.compute_properties()
        return proteome_dict
    
    def compute_identity_matrix(self, n_jobs: int = 4,  similarity_threshold: float = 0.95) -> pd.DataFrame:
        """
        Compute pairwise sequence identity using multiprocessing.
        Only computes unique pairs (i < j), avoids redundant and self-comparisons.
        Stores result in self.identity_matrix and returns it.
        """
        from concurrent.futures import ProcessPoolExecutor
        from itertools import combinations

        if not self.sequences:
            raise ValueError("Proteome is empty. Load a proteome first.")

        labels = list(self.sequences.keys())
        sequences = list(self.sequences.values())
        n = len(sequences)
        matrix = np.eye(n)  # Initialize with 1.0 on diagonal
        df_similarity = []

        # Prepare unique pairs
        pairs = [(i, j, sequences[i], sequences[j]) for i, j in combinations(range(n), 2)]

        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, j, identity in tqdm(executor.map(self._compute_identity, pairs), total=len(pairs), desc="Computing identities"):
                matrix[i][j] = identity
                matrix[j][i] = identity

                df_similarity.append([labels[i], labels[j], identity])

                if similarity_threshold and identity >= similarity_threshold:
                    msg = f"High similarity detected between {labels[i]} and {labels[j]}: {identity:.2f}"
                    logger.warning(msg)
                    self.high_similarity_pairs.append((labels[i], labels[j], identity))

        df_similarity_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
        df_similarity = pd.DataFrame(df_similarity, columns=['ORF1', 'ORF2', 'Identity'])
        self.high_similarity_pairs = pd.DataFrame(self.high_similarity_pairs, columns=['ORF1', 'ORF2', 'Identity'])
        self.identity_matrix = df_similarity
        self.identity_table = df_similarity
        return df_similarity, df_similarity_matrix

    def plot_identity_heatmap(self, output_image: str, cmap: str = "coolwarm", vmin: float = 0.2, vmax: float = 1.0, show: bool = False):
        """
        Plot heatmap from self.identity_matrix using Matplotlib only.

        Parameters
        ----------
        output_image : str
            Path to save the heatmap image.
        cmap : str
            Colormap for the heatmap.
        vmin, vmax : float
            Min and max values for color scale.
        show : bool
            If True, display the plot interactively instead of saving.
        """
        import matplotlib.pyplot as plt
        if self.identity_matrix is None:
            raise ValueError("Identity matrix not computed. Run compute_identity_matrix() first.")

        fig, ax = plt.subplots(figsize=(14, 14))
        cax = ax.imshow(self.identity_matrix.values, cmap=cmap, vmin=vmin, vmax=vmax)

        # Add colorbar
        fig.colorbar(cax, ax=ax)

        # Set ticks and labels
        ax.set_xticks(range(len(self.identity_matrix.columns)))
        ax.set_yticks(range(len(self.identity_matrix.index)))
        ax.set_xticklabels(self.identity_matrix.columns, rotation=90)
        ax.set_yticklabels(self.identity_matrix.index)

        ax.set_title("Sequence Identity Heatmap", fontsize=14)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(output_image, dpi=300)
            plt.close()

    def get_sequence(self, protein_id: str) -> str:
        """
        Retrieve a protein sequence by its ID.
        Raises KeyError if the ID is not found.
        """
        if protein_id not in self.sequences:
            raise KeyError(f"Protein ID '{protein_id}' not found in proteome.")
        return self.sequences[protein_id]

    def get_ids(self) -> list[str]:
        """
        Return a list of all protein IDs in the proteome.
        """
        return list(self.sequences.keys())
    
    def filter_by_regex(self, pattern: str, return_sequences: bool = False) -> dict | list[str]:
        """
        Filter protein IDs by a regex pattern.

        Parameters
        ----------
        pattern : str
            Regular expression to match protein IDs.
        return_sequences : bool, optional
            If True, return a dictionary {id: sequence}. If False, return a list of IDs.

        Returns
        -------
        dict or list
            Matching IDs (and sequences if return_sequences=True).
        """
        import matplotlib.pyplot as plt
        regex = re.compile(pattern)
        matches = {pid: seq for pid, seq in self.sequences.items() if regex.search(pid)}
        return matches if return_sequences else list(matches.keys())

    def compute_properties(self) -> pd.DataFrame:
        """
        Compute physicochemical properties for all sequences in the proteome.

        Returns
        -------
        pd.DataFrame
            Table with columns:
            ['id', 'length', 'molecular_weight', 'isoelectric_point',
            'instability_index', 'gravy', 'aromaticity']
        """
        from Bio.SeqUtils import molecular_weight, IsoelectricPoint
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        if not self.sequences:
            raise ValueError("Proteome is empty. Load a proteome first.")

        data = []
        for pid, seq in self.sequences.items():
            analysis = ProteinAnalysis(seq)
            length = len(seq)
            mw = molecular_weight(seq, seq_type="protein")
            pI = IsoelectricPoint(seq).pi()
            instability = analysis.instability_index()
            gravy = analysis.gravy()
            aromaticity = analysis.aromaticity()

            data.append({
                "id": pid,
                "length": length,
                "molecular_weight": mw,
                "isoelectric_point": pI,
                "instability_index": instability,
                "gravy": gravy,
                "aromaticity": aromaticity
            })

        return pd.DataFrame(data).set_index("id")

    # -------------------------
    # 2. Export
    # -------------------------
    def summary(self) -> None:
        """
        Generate a summary of the proteome information.

        Returns
        -------
        dict
            Dictionary with proteome metrics.
        """
                
        if not self.sequences:
                return {
                    "total_sequences": 0,
                    "total_residues": 0,
                    "average_length": 0,
                    "min_length": 0,
                    "max_length": 0,
                    "invalid_sequences": len(self.invalid_sequences),
                    "high_similarity_pairs": len(self.high_similarity_pairs)
                }

        lengths = [len(seq) for seq in self.sequences.values()]
        total_sequences = len(self.sequences)
        total_residues = sum(lengths)
        avg_length = total_residues / total_sequences

        return {
            "total_sequences": total_sequences,
            "total_residues": total_residues,
            "average_length": round(avg_length, 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "invalid_sequences": len(self.invalid_sequences),
            "high_similarity_pairs": len(self.high_similarity_pairs)
        }

    ##
    # -------------------------
    # 4. Find matches in PDB
    # ------------------------
    def screen_proteome_against_pdb(self, score_cutoff=0.15, **kwargs) -> pd.DataFrame:
        """
        Runs the PDB search for all sequences in the manager and aggregates results.
        
        Parameters
        ----------
        top_n : int
            Only perform heavy local alignment on the top N hits per protein 
            to save time/bandwidth.
        **kwargs : 
            Passed to search_pdb_sequence (evalue_cutoff, identity_cutoff, etc.)
        """
        all_results = []
        
        # tqdm gives you a nice progress bar for the proteome loop
        from tqdm import tqdm
        
        logger.info(f"Starting PDB screening for {len(self.sequences)} sequences...")
        
        for protein_id, sequence in tqdm(self.sequences.items(), desc="Proteome Search"):
            # 1. Get hits from API
            df_hits = self.search_pdb_sequence(sequence, protein_name=protein_id, **kwargs)
            
            if df_hits.empty:
                continue
                
            # 2. Limit to Top N hits to avoid downloading hundreds of PDBs per query
            df_top = df_hits.loc[df_hits.score >= score_cutoff, :].reset_index(drop=True)
            logger.info(f"Found {len(df_top)} hits above score {score_cutoff} for protein {protein_id}.")
            # df_top = df_top.head(5) # Limit to top 5 hits. Just for testing

            for index, row in df_top.iterrows():
                pdb_id = row['PDB_code']
                chain_id = row['PDB_chain']
                mol = Molecule(pdb_id)
                chain_seq = mol.sequence()[chain_id]
                align_info = ProteomeManager.align_sequences(sequence, chain_seq)
                df_top.at[index, 'alignment_score'] = align_info['score']
                df_top.at[index, 'coverage'] = align_info['coverage']
                df_top.at[index, 'identity'] = align_info['identity']
                df_top.at[index, 'gaps'] = align_info['gaps']
            
            # 3. Store result
            all_results.append(df_top)
            
        if not all_results:
            logger.warning("No PDB matches found for any sequence.")
            return pd.DataFrame()
            
        # Combine everything into one master table
        master_df = pd.concat(all_results, ignore_index=True)
        master_df.reset_index(inplace=True, drop=True)
        
        # Sort by best alignment score across the whole proteome
        master_df = master_df.sort_values(by=["protein_name", "alignment_score"], ascending=[True, False])
        
        return master_df
    
    @staticmethod
    def search_pdb_sequence(sequence: str, protein_name: str = "unknown", **kwargs) -> pd.DataFrame:
        """
        Performs a sequence search against the PDB and returns a detailed DataFrame.
        
        Parameters
        ----------
        sequence : str
            The amino acid sequence to search.
        protein_name : str
            Identifier for the input protein (used in the resulting dataframe).
        **kwargs : 
            evalue_cutoff (float), identity_cutoff (float).
        """
        import requests

        # API Endpoint for Search
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
        # Default parameters
        evalue_cutoff = kwargs.get("evalue_cutoff", 1.0)
        identity_cutoff = kwargs.get("identity_cutoff", 0.1)

        # Build the Search Query JSON
        query_payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": evalue_cutoff,
                    "identity_cutoff": identity_cutoff,
                    "target": "pdb_protein_sequence",
                    "value": sequence
                }
            },
            "request_options": {
                "return_all_hits": True,
                "scoring_strategy": "sequence"
            },
            "return_type": "polymer_entity"
        }

        try:
            response = requests.post(search_url, json=query_payload)
            response.raise_for_status()
            search_results = response.json()
        except Exception as e:
            logging.error(f"PDB Search failed: {e}")
            return pd.DataFrame()

        # Parse the results into rows
        rows = []
        if "result_set" in search_results:
            for hit in search_results["result_set"]:
               

                # Extracting requested fields
                pdb_id_entity = hit["identifier"] # Format: 4HHB_1
                
                rows.append({
                    "protein_name": protein_name,
                    "PDB_ID": pdb_id_entity,
                    "PDB_code": pdb_id_entity.split('_')[0] if '_' in pdb_id_entity else '',
                    "PDB_chain": pdb_id_entity.split('_')[1] if '_' in pdb_id_entity else '',
                    "score": hit.get("score"),
                })

        df = pd.DataFrame(rows)
            
        return df
    
    @staticmethod 
    def align_sequences(seq1, seq2):
        from Bio import Align
        aligner = Align.PairwiseAligner(scoring="blastp")
        # aligner.mode = 'local'  # Use 'global' or 'local'

        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]

        # Extracting details
        score = best_alignment.score

        counts = best_alignment.counts()
        matches = counts.identities
        length = best_alignment.length
        identity = (matches / length) * 100 if length > 0 else 0

        # 2. Query Coverage
        # Check how much of the original query length is involved in the alignment
        query_len = len(seq1)
        # We sum the lengths of the aligned segments in the query
        query_covered_bases = sum(end - start for start, end in best_alignment.aligned[1])
        coverage = (query_covered_bases / query_len) * 100

        # 3. Gaps
        gaps = counts.gaps

        return {
            "score": score,
            "identity": identity,
            "coverage": coverage,
            "gaps": gaps,
        }

    # -------------------------
    # 4. Monomer prediction
    # -------------------------
    def create_af3_jobs(self, output_dir: str, template: str) -> None:
        """
        Generate input files for AlphaFold3 jobs.
        """
        pass

    def create_boltz_jobs(self, output_dir: str, template: str) -> None:
        """
        Generate input files for Boltz2 jobs.
        """
        pass

    def load_model_info_monomer(self, model_dir: str, engine: str) -> pd.DataFrame:
        """
        Load and parse model information from AlphaFold/Boltz output directories.
        """

        model_number = None
        if engine.lower() == "af3":
            full_data = process_full_data_af3(model_dir)
            orf_id = Path(model_dir).parent.name
        elif engine.lower() == "boltz":
            full_data = process_full_data_boltz(model_dir)
            orf_id = Path(model_dir).parent.name
        elif engine.lower() == "colabfold":
            full_data = process_full_data_colabfold(model_dir)
            orf_id = ProteomeManager._get_orf_id_from_path(model_dir, "colabfold")
            model_number = int(os.path.basename(model_dir).split("model_")[1].split("_")[0])
        else:
            raise ValueError("engine should be 'AF3', 'Boltz' or 'ColabFold'")
        
        summary_dict = {"ORF": orf_id,
                "Model_num": model_number, 
                "ipTM": full_data.get("iptm_chain_pair", 0)[0][0], 
                "pTM": full_data.get("ptm", 0),
                "mean_plddt": np.mean(full_data.get("ca_plddts", 0)),
                "mean_pae": np.mean(full_data.get("pae", 0)),
                }
        
        return summary_dict
    
    def load_model_info(self, model_dir: str, engine: str = "AF3", file_ext: str = "cif") -> pd.DataFrame:
        """
        Load and parse model information from AlphaFold/Boltz/ColabFold output directories.

        Only processes ORFs that have both a sequence in ``self.sequences`` and at least one
        model file on disk.  ORFs missing either are skipped with a warning.  If no proteome
        has been loaded, all found model files are processed.

        Parameters
        ----------
        model_dir : str
            Root directory containing one sub-folder per ORF, each holding model files.
        engine : str
            One of ``"AF3"``, ``"Boltz"``, or ``"ColabFold"`` (case-insensitive).
        file_ext : str
            Extension of model files to search for (default ``"cif"``; use ``"pdb"`` if needed).

        Returns
        -------
        pd.DataFrame
            Per-model summary rows (same as ``self.model_info``).
        """
        from glob import glob

        all_model_data = glob(f"{model_dir}/*/*.{file_ext}")

        # Build orf_id → [file paths] mapping
        orf_to_files: dict[str, list[str]] = {}
        for f in all_model_data:
            orf_id = self._get_orf_id_from_path(f, engine)
            if orf_id is not None:
                orf_to_files.setdefault(orf_id, []).append(f)

        if self.sequences:
            sequence_orfs = set(self.sequences.keys())
            model_orfs_raw = set(orf_to_files.keys())

            # Defensive: resolve case-insensitive matches before flagging as missing
            seq_lower_map = {k.lower(): k for k in sequence_orfs}
            resolved_orf_to_files: dict[str, list[str]] = {}
            for raw_orf, files in orf_to_files.items():
                if raw_orf in sequence_orfs:
                    resolved_orf_to_files.setdefault(raw_orf, []).extend(files)
                elif raw_orf.lower() in seq_lower_map:
                    canonical = seq_lower_map[raw_orf.lower()]
                    logger.warning(
                        "ORF ID case mismatch: model dir '%s' matched to sequence key '%s'. "
                        "Using canonical ID.", raw_orf, canonical
                    )
                    resolved_orf_to_files.setdefault(canonical, []).extend(files)
                else:
                    resolved_orf_to_files.setdefault(raw_orf, []).extend(files)

            model_orfs = set(resolved_orf_to_files.keys())

            for orf in sorted(sequence_orfs - model_orfs):
                logger.warning("ORF '%s': sequence loaded but no model files found — skipping.", orf)
            for orf in sorted(model_orfs - sequence_orfs):
                logger.warning("ORF '%s': model files found but no sequence in proteome — skipping.", orf)

            valid_orfs = sequence_orfs & model_orfs
            filtered_files = [f for orf in valid_orfs for f in resolved_orf_to_files[orf]]
        else:
            logger.info("No sequences loaded; processing all model files found in '%s'.", model_dir)
            filtered_files = all_model_data

        if not filtered_files:
            logger.warning("No valid model files to process after cross-checking sequences and structures.")
            return pd.DataFrame()

        model_df = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            worker = partial(self.load_model_info_monomer, engine=engine)
            for res in tqdm(executor.map(worker, filtered_files), total=len(filtered_files), desc="Loading model info"):
                model_df.append(res)

        df = pd.DataFrame(model_df)
        grouped_df = df.groupby("ORF").mean().drop("Model_num", axis=1, errors="ignore").reset_index()
        self.model_info = df
        self.model_info_extended = grouped_df
        return df

    
    