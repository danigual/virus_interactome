from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from Bio import SeqIO, Align
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
from .model import Engine, Model


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
        self._sequences = {}
        self.invalid_sequences = {}
        self.high_similarity_pairs = []
        # self.identity_matrix = None
        self.identity_table = None
        self.sequence_properties = None
        self._ids_cache: Optional[Tuple[str, ...]] = None
        self._order_mode: str = "insertion"
        self.model_info_by_model: Optional[pd.DataFrame] = None
        self.model_info_by_orf: Optional[pd.DataFrame] = None
        # self.model_info_extended: Optional[pd.DataFrame] = None
        self._model_engine: Optional[str] = None

        if fasta_file:
            self.load_proteome(fasta_file)
        
        self._file_path = fasta_file

    @property
    def sequences(self) -> dict:
        return self._sequences
    
    @property
    def ids(self) -> Tuple[str, ...]:
        """
        Immutable tuple of valid protein IDs.
        Deterministic order: FASTA insertion order or alphabetical (configurable).
        """
        if self._ids_cache is None:
            if self._order_mode == "insertion":
                self._ids_cache = tuple(self._sequences.keys())
            else:
                self._ids_cache = tuple(sorted(self._sequences.keys()))
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
    
    @sequences.setter
    def sequences(self, sequences: dict) -> None:
        self._sequences = sequences
        self._ids_cache = None

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
        from Bio import Align
        i, j, seq1, seq2 = args
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1
        aligner.mismatch_score = 0
        aligner.open_gap_score = 0
        aligner.extend_gap_score = 0
        score = aligner.score(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return i, j, score / max_len
    
    def __repr__(self) -> str:
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
        return len(self._sequences)
    
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
        if protein_id not in self._sequences:
            raise KeyError(f"Protein ID '{protein_id}' not found in proteome.")
        return self._sequences[protein_id]

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

        self._sequences = proteome_dict
        self._file_path = fasta_file
        logger.info(f"Proteome loaded successfully with {len(proteome_dict)} proteins.")

        # if self._sequences:
        #     logger.info("Computing sequence identity matrix...")
        #     self.compute_identity_matrix(n_jobs=10, similarity_threshold=0.95)
        # self.identity_matrix = similarity_matrix
        # self.identity_table = similarity_data
        # logger.info(f"Proteome loaded successfully with {len(proteome_dict)} proteins.")
        # self.sequence_properties = self.compute_properties()
        return proteome_dict
    
    def compute_identity(self, n_jobs: int = 4,  similarity_threshold: float = 0.95) -> pd.DataFrame:
        """
        Compute pairwise sequence identity using multiprocessing.
        Only computes unique pairs (i < j), avoids redundant and self-comparisons.
        Stores result in self.identity_matrix and returns it.
        """
        from concurrent.futures import ProcessPoolExecutor
        from itertools import combinations

        if not self._sequences:
            raise ValueError("Proteome is empty. Load a proteome first.")

        labels = list(self._sequences.keys())
        sequences = list(self._sequences.values())
        n = len(sequences)
        # matrix = np.eye(n)  # Initialize with 1.0 on diagonal
        df_similarity = []

        # Prepare unique pairs
        pairs = [(i, j, sequences[i], sequences[j]) for i, j in combinations(range(n), 2)]

        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, j, identity in tqdm(executor.map(self._compute_identity, pairs), total=len(pairs), desc="Computing identities"):
                # matrix[i][j] = identity
                # matrix[j][i] = identity

                df_similarity.append([labels[i], labels[j], identity])

                if similarity_threshold and identity >= similarity_threshold:
                    msg = f"High similarity detected between {labels[i]} and {labels[j]}: {identity:.2f}"
                    logger.warning(msg)
                    self.high_similarity_pairs.append((labels[i], labels[j], identity))

        # df_similarity_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
        df_similarity = pd.DataFrame(df_similarity, columns=['ORF1', 'ORF2', 'Identity'])
        # self.identity_matrix = df_similarity_matrix
        self.identity_table = df_similarity
        return df_similarity

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
        import re
        regex = re.compile(pattern)
        matches = {pid: seq for pid, seq in self._sequences.items() if regex.search(pid)}
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
        if not self._sequences:
            raise ValueError("Proteome is empty. Load a proteome first.")

        data = []
        for pid, seq in self._sequences.items():
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
                
        if not self._sequences:
                return {
                    "total_sequences": 0,
                    "total_residues": 0,
                    "average_length": 0,
                    "min_length": 0,
                    "max_length": 0,
                    "invalid_sequences": len(self.invalid_sequences),
                    "high_similarity_pairs": len(self.high_similarity_pairs)
                }

        lengths = [len(seq) for seq in self._sequences.values()]
        total_sequences = len(self._sequences)
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
        
        logger.info(f"Starting PDB screening for {len(self._sequences)} sequences...")
        
        for protein_id, sequence in tqdm(self._sequences.items(), desc="Proteome Search"):
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
    def describe_orf(
        self,
        orf_ids: str | list[str],
        identity_threshold: Optional[float] = None,
        interactome_df: Optional[pd.DataFrame] = None,
        foldseek_df: Optional[pd.DataFrame] = None,
        iptm_threshold: float = 0.6,
    ) -> None:
        """
        Print an interactive profile for one or more ORFs.

        Parameters
        ----------
        orf_ids : str or list[str]
            ORF ID(s) to describe.
        identity_threshold : float, optional
            0–100 identity cutoff for the final similarity prompt.
            If provided, skips the interactive input() call.
        interactome_df : pd.DataFrame, optional
            Output of InteractomeProcessor — must contain a 'PPI' column.
        foldseek_df : pd.DataFrame, optional
            Output of run_foldseek_search — must contain a 'protein_id' column.
        iptm_threshold : float
            Default ipTM cutoff for the PPI interactive filter (default 0.6).
        """
        if isinstance(orf_ids, str):
            orf_ids = [orf_ids]

        SEP = "=" * 62

        for orf_id in orf_ids:
            print(f"\n{SEP}")
            print(f"  ORF: {orf_id}")
            print(SEP)

            if orf_id not in self._sequences:
                print(f"  [!] '{orf_id}' not found in proteome.")
                continue

            seq = self._sequences[orf_id]

            # ── SEQUENCE ──────────────────────────────────────────────
            print("\n[SEQUENCE]")
            analysis = ProteinAnalysis(seq)
            mw = molecular_weight(seq, seq_type="protein")
            pi = IsoelectricPoint(seq).pi()
            charge_ph7 = IsoelectricPoint(seq).charge_at_pH(7.0)
            preview = seq[:30] + ("..." if len(seq) > 30 else "")
            print(f"  Length      : {len(seq)} aa")
            # print(f"  Preview     : {preview}")
            print(f"  MW          : {mw:.1f} Da")
            print(f"  pI          : {pi:.2f}")
            print(f"  Charge@pH7  : {charge_ph7:+.2f}")
            print(f"  GRAVY       : {analysis.gravy():.3f}")
            print(f"  Aromaticity : {analysis.aromaticity():.3f}")
            instability = analysis.instability_index()
            print(f"  Instability : {instability:.1f} ({'unstable' if instability > 40 else 'stable'})")
            aa_counts = analysis.count_amino_acids()
            aa_str = ", ".join(f"{aa}:{round(100 * cnt / len(seq), 2)}" for aa, cnt in sorted(aa_counts.items()) if cnt > 0)
            print(f"  AA comp (%) : {aa_str}")

            # ── MODELS ────────────────────────────────────────────────
            print("\n[MODELS]")
            if self.model_info_by_model is not None and orf_id in self.model_info_by_model["id"].values:
                row = self.model_info_by_model[self.model_info_by_model["id"] == orf_id].iloc[0]
                n_models = self.model_info_by_model[self.model_info_by_model["id"] == orf_id].shape[0]
                print(f"  Engine      : {self._model_engine or 'unknown'}")
                print(f"  N models    : {n_models}")
                for col in ["pTM", "mean_plddt", "mean_pae", "ptm", "iptm"]:
                    if col in row.index:
                        print(f"  {col:<12}: {row[col]:.3f}")
            else:
                print("  No model data. Run load_model_info() first.")

            # ── SIMILARITY ────────────────────────────────────────────
            print("\n[SIMILARITY]  ")

            if self.identity_table is None:
                self.compute_identity()
                # print("  Not computed. Run compute_identity_matrix() first.")
            identity_sel = self.identity_table[
                (self.identity_table["ORF1"] == orf_id) | (self.identity_table["ORF2"] == orf_id)
            ].sort_values(by="Identity", ascending=False)
            print("TOP 5 similar ORFs:\n")
            print(identity_sel.head(5).to_string(index=False))

            print("\nIdentity distribution:\n")

            bins = np.arange(0, 1.1, 0.1)

            counts, _ = np.histogram(identity_sel.Identity, bins=bins)

            percentages = counts / counts.sum() * 100

            
            STAR_SCALE = 5.0  # 1 star per 1%

            lines = []
            for i, p in enumerate(percentages):
                stars = "*" * int(round(p / STAR_SCALE))
                lines.append(f"{bins[i]:.1f}–{bins[i+1]:.1f}: {p:6.2f}%  {stars}")

            summary = "\n".join(lines)

            print(summary)

            # ── PPIs ──────────────────────────────────────────────────
            if interactome_df is not None:
                print("\n[PPIs]")
                ppi_mask = interactome_df["PPI"].apply(lambda p: orf_id in str(p).split("__"))
                df_ppi = interactome_df[ppi_mask]
                print(f"  Total PPIs  : {len(df_ppi)}")
                if "Tier" in df_ppi.columns:
                    for tier, cnt in df_ppi["Tier"].value_counts().items():
                        print(f"    {tier}: {cnt}")
                if "ipTM" in df_ppi.columns and not df_ppi.empty:
                    try:
                        raw = input(f"\n  Show PPIs with ipTM > {iptm_threshold} ? [Enter=yes / type threshold / 'n'=skip]: ").strip()
                        if raw.lower() == "n":
                            pass
                        else:
                            threshold = float(raw) if raw else iptm_threshold
                            show_cols = [c for c in ["PPI", "ipTM", "Tier"] if c in df_ppi.columns]
                            df_filtered = df_ppi[df_ppi["ipTM"] > threshold][show_cols]
                            print(df_filtered.to_string(index=False) if not df_filtered.empty else "  (no results above threshold)")
                    except (EOFError, ValueError):
                        pass

            # ── FOLDSEEK ──────────────────────────────────────────────
            if foldseek_df is not None:
                print("\n[FOLDSEEK]")
                fs_rows = foldseek_df[foldseek_df["protein_id"] == orf_id]
                if fs_rows.empty:
                    print("  No hits found.")
                else:
                    cols = [c for c in ["rank", "target", "fident", "alnlen", "evalue", "bits"] if c in fs_rows.columns]
                    print(fs_rows[cols].to_string(index=False))

    def _load_model_info_monomer(self, model_path: str, engine: str) -> pd.DataFrame:
        """
        Load and parse model information from AlphaFold/Boltz output directories.
        """
        model = Model(model_path, engine=engine)
        return model.summary()
    
    def load_model_info(self, model_dir: str, engine: str | Engine = "AF3") -> pd.DataFrame:
        """
        Load and parse model information from AlphaFold/Boltz/ColabFold output directories.

        Only processes ORFs that have both a sequence in ``self._sequences`` and at least one
        model file on disk.  ORFs missing either are skipped with a warning.  If no proteome
        has been loaded, all found model files are processed.

        Parameters
        ----------
        model_dir : str
            Root directory containing one sub-folder per ORF, each holding model files.
        engine : str
            One of ``"AF3"``, ``"Boltz"``, or ``"ColabFold"`` (case-insensitive).

        Returns
        -------
        pd.DataFrame
            Per-model summary rows.
        """
        from glob import glob
        engine = Engine(engine.lower()) if isinstance(engine, str) else engine

        if engine == Engine.AF3 or engine == Engine.BOLTZ:
            file_ext = "cif"
        elif engine == Engine.COLABFOLD:
            file_ext = "pdb"

        all_model_data = {}
        all_models = []
        for seq_id in self.ids:
            seq_id_models = glob(f"{model_dir}/{seq_id}/*.{file_ext}")
            if len(seq_id_models) == 0:
                logger.warning(f"ORF '{seq_id}': no model files found in '{model_dir}' matching '*{seq_id}*/*.{file_ext}'.")
            all_model_data[seq_id] = seq_id_models
            all_models.extend(seq_id_models)

        self.orf_models = all_model_data

        if len(all_models) == 0:
            logger.warning(f"No model files found in '{model_dir}' matching '*/*.{file_ext}'.")
            return pd.DataFrame()

        model_df = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            worker = partial(self._load_model_info_monomer, engine=engine)
            for res in tqdm(executor.map(worker, all_models), total=len(all_models), desc="Loading model info"):
                model_df.append(res)

        df = pd.DataFrame(model_df)
        grouped_df = df.drop(["model_num", "path", "engine"], axis=1, errors="ignore").groupby("id").mean().reset_index()
        self.model_info_by_model = df.sort_values(by="id").reset_index(drop=True)
        self.model_info_by_orf = grouped_df.sort_values(by="id").reset_index(drop=True)
        self._model_engine = engine
        return df

    def view(self, orf_id: str, mode: str = "plddt") -> None:
        """
        Visualize all models for a given ORF in ChimeraX.
        Models are opened as separate structures and arranged in a grid.

        Parameters
        ----------
        orf_id : str
            The ORF to visualize.
        mode : "plddt" | "chain" | "rainbow"
            Coloring scheme passed to each model.
        """
        import subprocess
        import tempfile

        if not hasattr(self, "orf_models") or orf_id not in self.orf_models:
            raise ValueError(
                f"No model files found for '{orf_id}'. Run load_model_info() first."
            )

        model_paths = self.orf_models[orf_id]
        if not model_paths:
            raise ValueError(f"orf_models['{orf_id}'] is empty.")

        lines = []
        for i, path in enumerate(sorted(model_paths), start=1):
            lines.append(f"open {path}")

        # color all opened models
        if mode == "plddt":
            lines.append("color bfactor palette alphafold")
        elif mode == "chain":
            lines.append("rainbow chain")
        elif mode == "rainbow":
            lines.append("rainbow")

        lines.append(f"mm #2-{len(model_paths)} to #1")
        mod = Model(model_paths[0], engine=self._model_engine)
        lines += [f"alphafold pae #1 file {mod._extra_files['scores']} palette paegreen"]
        lines.append("view")

        script = "\n".join(lines)

        # write to a temp file so we don't pollute the working directory
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cxc", delete=False
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        logger.info(f"Launching ChimeraX for {len(model_paths)} models of '{orf_id}'")
        subprocess.run(["chimerax", tmp_path])

    def filter_proteome(self, orfs_to_keep: list[str] | str) -> None:
        """
        Filter the proteome to only include specified ORFs.

        Parameters
        ----------
        orfs_to_keep : list[str]
            List of ORF IDs to retain in the proteome.  All others will be removed.
        """
        if not self._sequences:
            logger.warning("Proteome is empty. Nothing to filter.")
            return
        
        missing_orfs = [orf for orf in orfs_to_keep if orf not in self._sequences]
        if missing_orfs:
            logger.warning(f"The following ORFs were not found in the proteome and will be skipped: {missing_orfs}")

        filtered_sequences = {orf: seq for orf, seq in self._sequences.items() if orf in orfs_to_keep}
        
        if not filtered_sequences:
            logger.warning("No valid ORFs found to keep. Proteome will be empty.")
        
        self._sequences = filtered_sequences

    def save(self, path: str) -> None:
        """
        Save the ProteomeManager to disk.
        Pickle preserves all attributes including model_info, identity_table, etc.
        """
        import pickle
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"ProteomeManager saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ProteomeManager":
        """
        Load a ProteomeManager from a pickle file.
        Usage: pm = ProteomeManager.load("my_proteome.pkl")
        """
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ProteomeManager, got {type(obj)}")
        logger.info(f"ProteomeManager loaded from {path} ({len(obj)} sequences)")
        return obj
    
    def filter(
        self,
        ids: list[str] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        regex: str | None = None,
    ) -> "ProteomeManager":
        """
        Return a new ProteomeManager containing only the matching sequences.
        All available derived data (model_info, identity_table) is subset accordingly.

        Parameters
        ----------
        ids : list of str, optional
            Explicit list of IDs to keep.
        min_length, max_length : int, optional
            Filter by sequence length.
        regex : str, optional
            Keep IDs matching this pattern.
        """
        import re
        
        # start with all, then narrow
        selected = set(self._sequences.keys())

        if ids is not None:
            missing = set(ids) - selected
            if missing:
                logger.warning(f"IDs not found in proteome and will be ignored: {missing}")
            selected &= set(ids)

        if min_length is not None:
            selected = {pid for pid in selected if len(self._sequences[pid]) >= min_length}

        if max_length is not None:
            selected = {pid for pid in selected if len(self._sequences[pid]) <= max_length}

        if regex is not None:
            pattern = re.compile(regex)
            selected = {pid for pid in selected if pattern.search(pid)}

        if not selected:
            logger.warning("filter() produced an empty ProteomeManager.")

        # build new instance — no fasta file, inject sequences directly
        new_pm = ProteomeManager()
        new_pm._sequences = {pid: self._sequences[pid] for pid in self._sequences if pid in selected}
        new_pm._order_mode = self._order_mode
        new_pm._model_engine = self._model_engine

        # carry over derived data if available, subset to matching ids
        if self.identity_table is not None:
            mask = (
                self.identity_table["ORF1"].isin(selected) &
                self.identity_table["ORF2"].isin(selected)
            )
            new_pm.identity_table = self.identity_table[mask].reset_index(drop=True)
            new_pm.high_similarity_pairs = [
                (o1, o2, i) for o1, o2, i in self.high_similarity_pairs
                if o1 in selected and o2 in selected
            ]

        if self.model_info_by_orf is not None:
            new_pm.model_info_by_orf = self.model_info_by_orf[
                self.model_info_by_orf["id"].isin(selected)
            ].reset_index(drop=True)

        if self.model_info_by_model is not None:
            new_pm.model_info_by_model = self.model_info_by_model[
                self.model_info_by_model["id"].isin(selected)
            ].reset_index(drop=True)

        if hasattr(self, "orf_models"):
            new_pm.orf_models = {k: v for k, v in self.orf_models.items() if k in selected}

        if self.sequence_properties is not None:
            new_pm.sequence_properties = self.sequence_properties[
                self.sequence_properties.index.isin(selected)
            ]

        new_pm.invalid_sequences = dict(self.invalid_sequences)

        logger.info(f"filter() → {len(new_pm)} sequences kept from {len(self)}")
        return new_pm
    
    