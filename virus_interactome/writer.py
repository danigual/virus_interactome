import csv
import json
import logging
import warnings
import yaml

from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .proteome_manager import ProteomeManager
from .utils import check_sequence_validity

logger = logging.getLogger(__name__)


class InteractomeWriter:
    """
    Generates input files for protein folding engines from one or two proteomes.

    Attributes:
        proteome_a (ProteomeManager): First proteome.
        proteome_b (Optional[ProteomeManager]): Second proteome (inter mode only).
        mode (str): "intra" or "inter".
    """

    _ENGINE_CONFIG: Dict[str, Dict[str, Any]] = {
        "af3":       {"ext": "json", "default_threshold": 5000},
        "boltz2":    {"ext": "yaml", "default_threshold": 1600},
        "colabfold": {"ext": "csv",  "default_threshold": 10000},
    }

    def __init__(
        self,
        proteome_a: Union[str, ProteomeManager],
        proteome_b: Union[str, ProteomeManager, None] = None,
    ):
        self.proteome_a: Optional[ProteomeManager] = None
        self.proteome_b: Optional[ProteomeManager] = None
        self.mode = "intra"

        if isinstance(proteome_a, str):
            self.proteome_a = ProteomeManager(proteome_a)
        elif isinstance(proteome_a, ProteomeManager):
            self.proteome_a = proteome_a
        else:
            raise ValueError("proteome_a must be a string path or a ProteomeManager instance")

        if isinstance(proteome_b, str):
            self.proteome_b = ProteomeManager(proteome_b)
            self.mode = "inter"
        elif isinstance(proteome_b, ProteomeManager):
            self.proteome_b = proteome_b
            self.mode = "inter"
        elif proteome_b is not None:
            raise ValueError("proteome_b must be a string, ProteomeManager, or None")

    # ── Pair / job generators ─────────────────────────────────────────────────

    def generate_intra_pairs(self) -> Iterable[Tuple[str, str]]:
        """Unique unordered heteromeric pairs within Proteome A."""
        if self.proteome_a is None:
            raise ValueError("generate_intra_pairs() requires a valid proteome_a.")
        return combinations(list(self.proteome_a.sequences.keys()), 2)

    def generate_inter_pairs(self) -> Iterable[Tuple[str, str]]:
        """Cartesian product of sequences between Proteome A and Proteome B."""
        if self.mode != "inter" or self.proteome_b is None:
            raise ValueError("generate_inter_pairs() requires 'inter' mode with a valid proteome_b.")
        ids_a = list(self.proteome_a.sequences.keys())
        ids_b = list(self.proteome_b.sequences.keys())
        return product(ids_a, ids_b)

    def generate_homo_mers(self, nmin: int = 2, nmax: int = 6) -> Iterable[Tuple[str, int]]:
        """Homomeric configurations (oligomers) for sequences in Proteome A."""
        if self.mode == "inter":
            raise ValueError("Homomers can only be computed in 'intra' mode.")
        if nmin < 2:
            raise ValueError("nmin cannot be lower than 2.")
        if nmax < nmin:
            raise ValueError(f"nmax ({nmax}) must be >= nmin ({nmin}).")
        for pid in list(self.proteome_a.sequences.keys()):
            for n in range(nmin, nmax + 1):
                yield (pid, n)

    def generate_single_run(
        self,
        source: str = "a",
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None,
    ) -> Iterable[Tuple[str, str, int]]:
        """Yields (protein_id, sequence, 1) for monomer folding."""
        valid_sources = {"a", "b", "both"}
        if source not in valid_sources:
            raise ValueError(f"source must be one of {valid_sources}")
        if source in {"b", "both"}:
            if self.mode != "inter" or self.proteome_b is None:
                raise ValueError("source='b' or 'both' requires 'inter' mode with a proteome_b.")

        if source in {"a", "both"}:
            for pid in (ids_a or self.proteome_a.ids):
                yield (pid, self.proteome_a.sequences[pid], 1)

        if source in {"b", "both"}:
            for pid in (ids_b or self.proteome_b.ids):
                yield (pid, self.proteome_b.sequences[pid], 1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_colabfold_seq_str(seq_list: List[Tuple[str, str, int]]) -> str:
        """Colon-separated multimer sequence string for ColabFold."""
        parts = []
        for _, seq, count in seq_list:
            parts.extend([seq] * count)
        return ":".join(parts)

    def _build_seq_list_from_entry(
        self,
        kind: str,
        entry,
        mode: str,
        counts_map: Optional[Dict[str, int]],
    ) -> Tuple[list, str, str, str]:
        """Build (seq_list, name, idA, idB) from an iterator entry."""
        seq_list = []
        name = ""
        idA, idB = "", ""

        if kind == "pair":
            idA_raw, idB_raw = entry
            prot_A = self.proteome_a
            prot_B = self.proteome_b if mode == "inter_pairs" else self.proteome_a

            if mode == "intra_pairs" and idA_raw > idB_raw:
                idA, idB = idB_raw, idA_raw
                prot_A, prot_B = prot_B, prot_A
            else:
                idA, idB = idA_raw, idB_raw

            cntA = counts_map.get(idA, 1) if counts_map else 1
            cntB = counts_map.get(idB, 1) if counts_map else 1
            seq_list = [(idA, prot_A.sequences[idA], cntA), (idB, prot_B.sequences[idB], cntB)]
            name = f"{idA}__{idB}"

        elif kind == "homo":
            pid, copies = entry
            seq_list = [(pid, self.proteome_a.sequences[pid], copies)]
            name = f"{pid}__{copies}"
            idA = pid

        elif kind == "single":
            pid, seq, cnt = entry
            seq_list = [(pid, seq, cnt)]
            name = pid
            idA = pid

        return seq_list, name, idA, idB

    def _make_iterator(
        self,
        mode: str,
        nmin: int,
        nmax: int,
        ids_a: Optional[list],
        ids_b: Optional[list],
    ):
        """Build job iterator from mode string."""
        if mode == "intra_pairs":
            pairs = self.generate_intra_pairs()
            if ids_a:
                pairs = (p for p in pairs if p[0] in ids_a and p[1] in ids_a)
            return (("pair", p) for p in pairs)
        elif mode == "inter_pairs":
            return (("pair", p) for p in self.generate_inter_pairs())
        elif mode == "homomers":
            return (("homo", h) for h in self.generate_homo_mers(nmin=nmin, nmax=nmax))
        elif mode == "single":
            source = "both" if self.mode == "inter" else "a"
            return (("single", s) for s in self.generate_single_run(source=source, ids_a=ids_a, ids_b=ids_b))
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Choose from: intra_pairs, inter_pairs, homomers, single.")

    def _write_job_file(
        self,
        engine_lower: str,
        seq_list: list,
        name: str,
        save_path: str,
        residue_threshold: int,
    ) -> None:
        """Dispatch file writing to the correct engine handler (af3 / boltz2)."""
        if engine_lower == "af3":
            self.get_af3_input(seq_list, job_name=name, save_path=save_path,
                               residue_threshold=residue_threshold)
        elif engine_lower == "boltz2":
            self.get_boltz2_input(seq_list, save_path=save_path,
                                  residue_threshold=residue_threshold)

    def _write_colabfold_csv_jobs(
        self,
        iterator,
        out_path: Path,
        mode: str,
        counts_map: Optional[Dict[str, int]],
        residue_threshold: int,
        skip_over_threshold: bool,
        index_name: str,
        csv_name: str = "colabfold_input.csv",
    ) -> List[dict]:
        """
        Write a single ColabFold input CSV containing all jobs.

        ColabFold format::

            id,sequence
            ProtA__ProtB,SEQUENCEA:SEQUENCEB

        Run with: ``colabfold_batch colabfold_input.csv <output_dir>``
        """
        metas: List[dict] = []
        cf_csv_path = out_path / csv_name
        index_path = out_path / index_name

        fieldnames = ["engine", "mode", "name", "idA", "idB",
                      "countA", "countB", "total_residues", "warnings", "file_path"]

        with open(cf_csv_path, "w", newline="", encoding="utf-8") as cf_fh, \
             open(index_path, "w", newline="", encoding="utf-8") as idx_fh:

            cf_writer = csv.writer(cf_fh)
            cf_writer.writerow(["id", "sequence"])

            idx_writer = csv.DictWriter(idx_fh, fieldnames=fieldnames)
            idx_writer.writeheader()

            for kind, entry in iterator:
                seq_list, name, idA, idB = self._build_seq_list_from_entry(
                    kind, entry, mode, counts_map
                )
                total_res = sum(len(s) * c for _, s, c in seq_list)
                is_over_limit = total_res > residue_threshold
                warns = []

                if not (skip_over_threshold and is_over_limit):
                    seq_str = self._build_colabfold_seq_str(seq_list)
                    cf_writer.writerow([name, seq_str])

                if is_over_limit:
                    warns.append(f"Skipped: total residues {total_res} exceed {residue_threshold}")

                meta = {
                    "engine": "colabfold",
                    "mode": mode,
                    "name": name,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                    "warnings": "|".join(warns),
                    "file_path": str(cf_csv_path),
                }
                metas.append(meta)
                idx_writer.writerow(meta)

        logger.info(f"[ColabFold] Written {len(metas)} jobs to {cf_csv_path}")
        return metas

    # ── Public write method ───────────────────────────────────────────────────

    def write_interactome_jobs(
        self,
        engine: str,
        output_dir: str,
        *,
        mode: str = "intra_pairs",
        nmin: int = 2,
        nmax: int = 6,
        counts_map: Optional[Dict[str, int]] = None,
        ids_a: Optional[List[str]] = None,
        ids_b: Optional[List[str]] = None,
        residue_threshold: Optional[int] = None,
        skip_over_threshold: bool = False,
        filename_fmt: str = "{engine}_{name}.{ext}",
        index_name: str = "index.csv",
    ) -> List[dict]:
        """
        Generate input files for a folding engine from the proteome pair(s).

        For AF3 and Boltz2, writes one file per job. For ColabFold, writes a
        single ``colabfold_input.csv`` suitable for one ``colabfold_batch`` call.

        Args:
            engine: Target engine. One of ``'af3'``, ``'boltz2'``, ``'colabfold'``.
            output_dir: Directory where files and the index CSV are saved.
            mode: Job generation strategy: ``'intra_pairs'``, ``'inter_pairs'``,
                ``'homomers'``, or ``'single'``. Defaults to ``'intra_pairs'``.
            nmin: Min stoichiometry for homomers. Defaults to 2.
            nmax: Max stoichiometry for homomers. Defaults to 6.
            counts_map: Custom stoichiometry per protein ID.
            ids_a: Filter to specific IDs in Proteome A.
            ids_b: Filter to specific IDs in Proteome B.
            residue_threshold: Max total residues per job. Defaults to the
                engine-specific value (AF3: 5000, Boltz2: 1600, ColabFold: 10000).
            skip_over_threshold: If True, jobs exceeding the threshold are not written.
            filename_fmt: Filename template for AF3/Boltz2 files.
            index_name: Name of the metadata index CSV. Defaults to ``'index.csv'``.

        Returns:
            List of metadata dicts for every job processed.
        """
        engine_lower = engine.lower()
        if engine_lower not in self._ENGINE_CONFIG:
            raise ValueError(
                f"Unsupported engine '{engine}'. Must be one of: {list(self._ENGINE_CONFIG.keys())}"
            )

        cfg = self._ENGINE_CONFIG[engine_lower]
        ext = cfg["ext"]
        threshold = residue_threshold if residue_threshold is not None else cfg["default_threshold"]

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        iterator = self._make_iterator(mode, nmin, nmax, ids_a, ids_b)

        # ColabFold uses a single batch CSV — separate code path
        if engine_lower == "colabfold":
            return self._write_colabfold_csv_jobs(
                iterator, out_path, mode, counts_map,
                threshold, skip_over_threshold, index_name,
            )

        # AF3 / Boltz2 — one file per job
        metas: List[dict] = []
        index_path = out_path / index_name

        with open(index_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["engine", "mode", "name", "idA", "idB",
                          "countA", "countB", "total_residues", "warnings", "file_path"]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()

            for kind, entry in iterator:
                seq_list, name, idA, idB = self._build_seq_list_from_entry(
                    kind, entry, mode, counts_map
                )
                total_res = sum(len(s) * c for _, s, c in seq_list)
                is_over_limit = total_res > threshold
                warns = []
                save_path_str = ""

                if not (skip_over_threshold and is_over_limit):
                    base_name = filename_fmt.format(engine=engine_lower, name=name, ext=ext)
                    full_save_path = out_path / base_name
                    save_path_str = str(full_save_path)
                    self._write_job_file(engine_lower, seq_list, name, save_path_str, threshold)

                if is_over_limit:
                    warns.append(f"Skipped: total residues {total_res} exceed {threshold}")

                meta = {
                    "engine": engine_lower,
                    "mode": mode,
                    "name": name,
                    "idA": idA,
                    "idB": idB,
                    "countA": seq_list[0][2],
                    "countB": seq_list[1][2] if len(seq_list) > 1 else "",
                    "total_residues": total_res,
                    "warnings": "|".join(warns),
                    "file_path": save_path_str,
                }
                metas.append(meta)
                w.writerow(meta)

        return metas

    # ── Static input builders ─────────────────────────────────────────────────

    @staticmethod
    def check_input(
        seq_list: List[Tuple[str, str, int]],
        residue_threshold: int = 5000,
    ) -> Tuple[bool, Optional[str]]:
        """Validate seq_list and warn if total residues exceed threshold."""
        if not seq_list:
            return False, "Sequence list cannot be empty."
        total_res = 0
        for chain_id, seq, count in seq_list:
            if count < 1:
                return False, f"Count for {chain_id} must be at least 1."
            if not check_sequence_validity(seq):
                return False, f"{chain_id} is not a valid protein sequence."
            total_res += len(seq) * count
        if total_res > residue_threshold:
            warnings.warn(
                f"Total residues {total_res} exceed recommended maximum ({residue_threshold}).",
                category=UserWarning,
                stacklevel=2,
            )
        return True, None

    @staticmethod
    def get_af3_input(
        seq_list: List[Tuple[str, str, int]],
        job_name: str = "AF3_job",
        residue_threshold: int = 5000,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and optionally save an AlphaFold 3 JSON input."""
        is_valid, err = InteractomeWriter.check_input(seq_list, residue_threshold)
        if not is_valid:
            raise ValueError(f"Invalid input for AF3: {err}")

        sequences = [
            {"proteinChain": {"id": cid, "count": cnt, "sequence": seq}}
            for cid, seq, cnt in seq_list
        ]
        data = {"name": job_name, "sequences": sequences, "modelSeeds": []}

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        return data

    @staticmethod
    def get_boltz2_input(
        seq_list: List[Tuple[str, str, int]],
        residue_threshold: int = 1600,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and optionally save a Boltz2 YAML input."""
        is_valid, err = InteractomeWriter.check_input(seq_list, residue_threshold)
        if not is_valid:
            raise ValueError(f"Invalid input for Boltz: {err}")

        def _chain_id_gen():
            from string import ascii_uppercase
            for c in ascii_uppercase:
                yield c
            for r in range(2, 4):
                for combo in product(ascii_uppercase, repeat=r):
                    yield "".join(combo)

        chain_gen = _chain_id_gen()
        seqs2yaml = []
        for _, seq, count in seq_list:
            ids = [next(chain_gen) for _ in range(count)]
            seqs2yaml.append({"protein": {"id": ids, "sequence": seq}})

        data = {"version": 1, "sequences": seqs2yaml}

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(yaml.dump(data, default_flow_style=None, sort_keys=False))
        return data

    @staticmethod
    def get_colabfold_input(
        seq_list: List[Tuple[str, str, int]],
        save_path: Optional[str] = None,
    ) -> str:
        """Build a ColabFold multimer sequence string and optionally save as FASTA."""
        parts = []
        for _, seq, count in seq_list:
            parts.extend([seq] * count)
        seq_str = ":".join(parts)
        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f">job\n{seq_str}\n")
        return seq_str
