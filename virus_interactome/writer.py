import csv
import json
import logging
import warnings
import yaml

from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from .proteome_manager import ProteomeManager
from .utils import check_sequence_validity

logger = logging.getLogger(__name__)


class PoolDesigner:
    """
    Designs protein pools for pooled-PPI prediction (Todor et al. 2026).

    Given a proteome, constructs pools where every unique protein pair appears
    in at least one pool, the total amino acids per pool stay within
    ``token_limit``, and the number of pools is minimised (greedy algorithm).

    ``token_limit`` is *not* a fixed architectural constant — it reflects the
    available GPU VRAM when running ColabFold locally.  A conservative starting
    point for a ~50 GB GPU is 4000 aa.

    Parameters
    ----------
    proteome : ProteomeManager
        Source proteome; all sequences must be validated.
    token_limit : int
        Maximum total amino acids per pool (no default — set based on VRAM).
    seed : int
        Random seed for reproducible pool construction. Defaults to 42.
    """

    def __init__(self, proteome: ProteomeManager, token_limit: int, seed: int = 42) -> None:
        self._proteome = proteome
        self._token_limit = token_limit
        self._seed = seed

    # ── Public API ────────────────────────────────────────────────────────────

    def design_pools(self) -> List[List[str]]:
        """Greedy pool construction that covers all coverable protein pairs.

        Returns
        -------
        List[List[str]]
            Ordered list of pools; each pool is an ordered list of protein IDs.
            The position within a pool determines the ColabFold chain letter
            (index 0 → chain A, index 1 → chain B, …).
        """
        rng = np.random.default_rng(self._seed)
        sequences: Dict[str, str] = self._proteome.sequences
        proteins: List[str] = list(sequences.keys())

        uncovered, uncoverable = self._partition_pairs(proteins, sequences)

        if uncoverable:
            logger.warning(
                f"PoolDesigner: {len(uncoverable)} pairs exceed token_limit "
                f"({self._token_limit} aa) and cannot be pooled."
            )

        pools: List[List[str]] = []

        while uncovered:
            pool, pool_aa = self._init_pool(proteins, sequences, uncovered, rng)
            if not pool:
                break

            pool, pool_aa = self._grow_pool(pool, pool_aa, proteins, sequences, uncovered, rng)

            # Mark pairs covered by this pool
            for i, p1 in enumerate(pool):
                for p2 in pool[i + 1:]:
                    uncovered.discard(frozenset([p1, p2]))

            pools.append(pool)

        return pools

    def coverage_report(self, pools: List[List[str]]) -> Dict[str, Any]:
        """Statistics for a designed pool set.

        Parameters
        ----------
        pools : List[List[str]]
            Output of :meth:`design_pools`.

        Returns
        -------
        dict
            Keys: ``n_pools``, ``n_pairs_total``, ``n_pairs_coverable``,
            ``n_pairs_covered``, ``n_pairs_uncoverable``, ``avg_pool_size``,
            ``max_pool_size``, ``min_pool_size``, ``avg_pool_aa``.
        """
        sequences = self._proteome.sequences
        proteins = list(sequences.keys())

        _, uncoverable = self._partition_pairs(proteins, sequences)
        n_total = len(proteins) * (len(proteins) - 1) // 2
        n_uncoverable = len(uncoverable)

        covered: Set[FrozenSet] = set()
        for pool in pools:
            for i, p1 in enumerate(pool):
                for p2 in pool[i + 1:]:
                    covered.add(frozenset([p1, p2]))

        pool_sizes = [len(p) for p in pools]
        pool_aas = [sum(len(sequences[pid]) for pid in p) for p in pools]

        return {
            "n_pools": len(pools),
            "n_pairs_total": n_total,
            "n_pairs_coverable": n_total - n_uncoverable,
            "n_pairs_covered": len(covered),
            "n_pairs_uncoverable": n_uncoverable,
            "avg_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "max_pool_size": max(pool_sizes) if pool_sizes else 0,
            "min_pool_size": min(pool_sizes) if pool_sizes else 0,
            "avg_pool_aa": float(np.mean(pool_aas)) if pool_aas else 0.0,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _partition_pairs(
        self,
        proteins: List[str],
        sequences: Dict[str, str],
    ) -> Tuple[Set[FrozenSet], Set[FrozenSet]]:
        """Split all unique pairs into coverable and uncoverable sets."""
        uncovered: Set[FrozenSet] = set()
        uncoverable: Set[FrozenSet] = set()
        for i, p1 in enumerate(proteins):
            for p2 in proteins[i + 1:]:
                pair: FrozenSet = frozenset([p1, p2])
                if len(sequences[p1]) + len(sequences[p2]) > self._token_limit:
                    uncoverable.add(pair)
                else:
                    uncovered.add(pair)
        return uncovered, uncoverable

    @staticmethod
    def _uncovered_count(
        protein: str,
        candidates: List[str],
        uncovered: Set[FrozenSet],
    ) -> int:
        return sum(1 for q in candidates if frozenset([protein, q]) in uncovered)

    def _init_pool(
        self,
        proteins: List[str],
        sequences: Dict[str, str],
        uncovered: Set[FrozenSet],
        rng: np.random.Generator,
    ) -> Tuple[List[str], int]:
        """Pick the starting protein for a new pool."""
        shuffled = proteins.copy()
        rng.shuffle(shuffled)

        # Sort by number of uncovered pairs (descending) — stable relative to shuffle
        shuffled.sort(
            key=lambda p: self._uncovered_count(p, proteins, uncovered),
            reverse=True,
        )

        for p in shuffled:
            if len(sequences[p]) <= self._token_limit and self._uncovered_count(p, proteins, uncovered) > 0:
                return [p], len(sequences[p])

        return [], 0

    def _grow_pool(
        self,
        pool: List[str],
        pool_aa: int,
        proteins: List[str],
        sequences: Dict[str, str],
        uncovered: Set[FrozenSet],
        rng: np.random.Generator,
    ) -> Tuple[List[str], int]:
        """Iteratively add the best protein to the pool until no more fit."""
        pool_set = set(pool)

        while True:
            best_p: Optional[str] = None
            best_gain: int = 0

            candidates = [p for p in proteins if p not in pool_set]
            rng.shuffle(candidates)

            for p in candidates:
                if pool_aa + len(sequences[p]) > self._token_limit:
                    continue
                gain = self._uncovered_count(p, pool, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p

            if best_p is None or best_gain == 0:
                break

            pool.append(best_p)
            pool_set.add(best_p)
            pool_aa += len(sequences[best_p])

        return pool, pool_aa


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

    def write_pooled_jobs(
        self,
        engine: str,
        output_dir: str,
        token_limit: int,
        *,
        seed: int = 42,
        pool_manifest_name: str = "pool_manifest.csv",
        csv_name: str = "colabfold_pooled_input.csv",
    ) -> List[Dict[str, Any]]:
        """Design protein pools and write the ColabFold input CSV + manifest.

        Uses :class:`PoolDesigner` to build pools that cover every protein pair
        at least once while keeping each pool within ``token_limit`` amino acids.

        ``token_limit`` must match the GPU VRAM available on the machine running
        ColabFold (e.g. 4000 aa for ~50 GB VRAM).

        Parameters
        ----------
        engine : str
            Must be ``'colabfold'`` (only supported engine for pooling).
        output_dir : str
            Directory where ``colabfold_pooled_input.csv`` and
            ``pool_manifest.csv`` are written.
        token_limit : int
            Maximum total amino acids per pool.
        seed : int
            Random seed forwarded to :class:`PoolDesigner`. Defaults to 42.
        pool_manifest_name : str
            Filename for the pool manifest. Defaults to ``'pool_manifest.csv'``.
        csv_name : str
            Filename for the ColabFold input CSV.
            Defaults to ``'colabfold_pooled_input.csv'``.

        Returns
        -------
        List[dict]
            One dict per pool with keys ``pool_id``, ``proteins``,
            ``n_proteins``, ``total_aa``, ``n_pairs``.
        """
        engine_lower = engine.lower()
        if engine_lower != "colabfold":
            raise ValueError(
                f"write_pooled_jobs currently supports only 'colabfold', got '{engine}'."
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        proteome = self.proteome_a
        sequences = proteome.sequences

        designer = PoolDesigner(proteome, token_limit=token_limit, seed=seed)
        pools = designer.design_pools()

        report = designer.coverage_report(pools)
        logger.info(
            f"PoolDesigner: {report['n_pools']} pools cover "
            f"{report['n_pairs_covered']}/{report['n_pairs_coverable']} pairs "
            f"(avg pool size {report['avg_pool_size']:.1f} proteins, "
            f"{report['avg_pool_aa']:.0f} aa)."
        )
        if report["n_pairs_uncoverable"] > 0:
            logger.warning(
                f"{report['n_pairs_uncoverable']} pairs exceed token_limit and are omitted."
            )

        cf_csv_path = out_path / csv_name
        manifest_path = out_path / pool_manifest_name
        metas: List[Dict[str, Any]] = []

        with open(cf_csv_path, "w", newline="", encoding="utf-8") as cf_fh, \
             open(manifest_path, "w", newline="", encoding="utf-8") as mf_fh:

            cf_writer = csv.writer(cf_fh)
            cf_writer.writerow(["id", "sequence"])

            mf_writer = csv.DictWriter(
                mf_fh,
                fieldnames=["pool_id", "proteins", "n_proteins", "total_aa", "n_pairs"],
            )
            mf_writer.writeheader()

            for idx, pool in enumerate(pools):
                pool_id = f"pool_{idx:04d}"
                seq_str = ":".join(sequences[pid] for pid in pool)
                cf_writer.writerow([pool_id, seq_str])

                total_aa = sum(len(sequences[pid]) for pid in pool)
                n_pairs = len(pool) * (len(pool) - 1) // 2
                meta = {
                    "pool_id": pool_id,
                    "proteins": ",".join(pool),
                    "n_proteins": len(pool),
                    "total_aa": total_aa,
                    "n_pairs": n_pairs,
                }
                mf_writer.writerow(meta)
                metas.append(meta)

        logger.info(f"[Pooled ColabFold] {len(pools)} pools → {cf_csv_path}")
        return metas

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
