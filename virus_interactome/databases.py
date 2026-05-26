from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseClient:
    """PPI reference database loaded from a flat file.

    Normalises pairs as frozenset({A, B}) so directionality is irrelevant.
    Use ``from_file`` to construct.
    """

    def __init__(self, ppis: Set[FrozenSet[str]], metadata: pd.DataFrame) -> None:
        self._ppis: Set[FrozenSet[str]] = ppis
        self._metadata: pd.DataFrame = metadata

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        col_a: str = "protein_A",
        col_b: str = "protein_B",
        filters: Optional[Dict[str, List]] = None,
        extra_cols: Optional[List[str]] = None,
    ) -> "DatabaseClient":
        """Load a PPI database from a CSV or TSV file.

        Parameters
        ----------
        path:
            Path to file. Delimiter auto-detected from extension (.tsv/.txt → tab, else comma).
        col_a, col_b:
            Column names for the two protein identifiers.
        filters:
            Row-level inclusion filter. E.g. ``{"confidence": ["high"]}`` keeps only rows
            where ``confidence`` is ``"high"``. Multiple keys are ANDed.
        extra_cols:
            Additional columns to retain in ``metadata``. Missing columns are skipped with
            a warning.
        """
        path = Path(path)
        sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, dtype=str)

        for col in (col_a, col_b):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in '{path.name}'. "
                    f"Available columns: {df.columns.tolist()}"
                )

        if filters:
            for col, values in filters.items():
                if col not in df.columns:
                    raise ValueError(
                        f"Filter column '{col}' not found in '{path.name}'. "
                        f"Available columns: {df.columns.tolist()}"
                    )
                df = df[df[col].isin([str(v) for v in values])]
            df = df.reset_index(drop=True)

        requested_extra = extra_cols or []
        missing = [c for c in requested_extra if c not in df.columns]
        if missing:
            logger.warning("extra_cols not found and will be skipped: %s", missing)
        keep = [col_a, col_b] + [c for c in requested_extra if c in df.columns]
        metadata = df[keep].copy().reset_index(drop=True)

        ppis: Set[FrozenSet[str]] = {
            frozenset({str(a), str(b)})
            for a, b in zip(metadata[col_a], metadata[col_b])
        }

        return cls(ppis=ppis, metadata=metadata)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def ppis(self) -> Set[FrozenSet[str]]:
        """Set of frozenset({protein_A, protein_B}) — order-independent."""
        return self._ppis

    @property
    def metadata(self) -> pd.DataFrame:
        """DataFrame with (col_a, col_b[, extra_cols]) for filtered rows."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._ppis)

    def __contains__(self, pair: Union[Tuple[str, str], FrozenSet[str]]) -> bool:
        """``(A, B) in client`` — order-independent."""
        return frozenset(pair) in self._ppis

    @property
    def proteins(self) -> Set[str]:
        """Set of all protein IDs referenced in any PPI."""
        return {p for pair in self._ppis for p in pair}

    def __repr__(self) -> str:
        return f"DatabaseClient({len(self._ppis)} PPIs)"
