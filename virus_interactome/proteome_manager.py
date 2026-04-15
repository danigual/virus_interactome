
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
import matplotlib.pyplot as plt
import re

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
        self.file_path = None
        self.identity_matrix = None
        self.sequence_properties = None
        self._ids_cache: Optional[Tuple[str, ...]] = None
        self._order_mode: str = "insertion"

        if fasta_file:
            self.load_proteome(fasta_file)

    @property
    def ids(self) -> Tuple[str, ...]:
        """
        Immutable tuple of valid protein IDs.
        Deterministic order: FASTA insertion order or alphabetical (configurable).
        """
        if self._ids_cache is None:
            if self._order_mode == "sorted":
                self._ids_cache = tuple(sorted(self.sequences.keys()))
            else:
                # dict preserves insertion order (CPython 3.7+)
                self._ids_cache = tuple(self.sequences.keys())
        return self._ids_cache

    @staticmethod
    def _check_sequence_validity(seq: str) -> bool:
        return all(residue in VALID_AMINO_ACIDS for residue in seq)
    
    @staticmethod
    def _compute_identity(args):
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
        self.file_path = fasta_file
        logger.info(f"Proteome loaded successfully with {len(proteome_dict)} proteins.")
        return proteome_dict
    
    def compute_identity_matrix(self, n_jobs: int = 4,  similarity_threshold: float = 0.95) -> pd.DataFrame:
        """
        Compute pairwise sequence identity using multiprocessing.
        Only computes unique pairs (i < j), avoids redundant and self-comparisons.
        Stores result in self.identity_matrix and returns it.
        """
        if not self.sequences:
            raise ValueError("Proteome is empty. Load a proteome first.")

        labels = list(self.sequences.keys())
        sequences = list(self.sequences.values())
        n = len(sequences)
        matrix = np.eye(n)  # Initialize with 1.0 on diagonal

        # Prepare unique pairs
        pairs = [(i, j, sequences[i], sequences[j]) for i, j in combinations(range(n), 2)]

        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, j, identity in tqdm(executor.map(self._compute_identity, pairs), total=len(pairs), desc="Computing identities"):
                matrix[i][j] = identity
                matrix[j][i] = identity

                if similarity_threshold and identity >= similarity_threshold:
                    msg = f"High similarity detected between {labels[i]} and {labels[j]}: {identity:.2f}"
                    logger.warning(msg)
                    self.high_similarity_pairs.append((labels[i], labels[j], identity))

        df_similarity = pd.DataFrame(matrix, index=labels, columns=labels)
        self.identity_matrix = df_similarity
        return df_similarity

    
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

    