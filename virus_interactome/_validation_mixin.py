import logging
import pandas as pd
from typing import Dict, Optional

from .databases import DatabaseClient

logger = logging.getLogger(__name__)


class _ValidationMixin:
    """Database cross-validation methods for InteractomeAnalyzer."""

    def validate_against_database(
        self,
        known_ppis: DatabaseClient,
        id_map: Optional[Dict[str, str]] = None,
        ppi_separator: str = "__",
        model_agg: str = "mean",
    ) -> pd.DataFrame:
        """Cross-validate predicted PPIs against a reference PPI database.

        Parameters
        ----------
        known_ppis:
            Reference database built with :meth:`DatabaseClient.from_file`.
        id_map:
            Maps interactome protein IDs to database IDs.
            E.g. ``{"ORF1": "gene1"}``.  IDs absent from the map are treated
            as unmappable and yield ``experimental_support = NaN``.
            If ``None``, IDs are assumed to match the database directly.
        ppi_separator:
            Separator used in the ``PPI`` column. Default ``"__"``.
        model_agg:
            Aggregation strategy passed to :meth:`_aggregate_per_ppi`
            (``"mean"``, ``"max"``, or ``"best"``).

        Returns
        -------
        pd.DataFrame
            One row per unique PPI with all numeric metrics aggregated and an
            ``experimental_support`` column:

            - ``True``  — pair found in the reference database.
            - ``False`` — both proteins are in the database but the pair is
                          not recorded.
            - ``NaN``   — at least one protein is absent from the database
                          (interaction is not assessable).

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        ppi_df = self._aggregate_per_ppi(model_agg=model_agg)
        db_proteins = known_ppis.proteins

        def _support(ppi_str: str) -> Optional[bool]:
            parts = ppi_str.split(ppi_separator, 1)
            if len(parts) != 2:
                return None
            id_a, id_b = parts
            if id_map is not None:
                id_a = id_map.get(id_a)
                id_b = id_map.get(id_b)
                if id_a is None or id_b is None:
                    return None
            if id_a not in db_proteins or id_b not in db_proteins:
                return None
            return (id_a, id_b) in known_ppis

        ppi_df["experimental_support"] = ppi_df["PPI"].map(_support)
        return ppi_df

    def validation_summary(
        self,
        validated_df: pd.DataFrame,
        group_by: str = "Tier",
        known_ppis: Optional[DatabaseClient] = None,
        ppi_separator: str = "__",
    ) -> pd.DataFrame:
        """Summarise cross-validation results as precision / recall / F1 per group.

        Parameters
        ----------
        validated_df:
            Output of :meth:`validate_against_database` — one row per PPI with
            an ``experimental_support`` column.
        group_by:
            Column in ``validated_df`` to group by (e.g. ``"Tier"``,
            ``"iLIS_Tier"``). Defaults to ``"Tier"``.
        known_ppis:
            The same :class:`DatabaseClient` used in
            :meth:`validate_against_database`. When provided, enables recall
            and F1 computation. Recall denominator = experimental PPIs where
            both proteins appear anywhere in the interactome data
            (reachable subset).
        ppi_separator:
            Separator in the ``PPI`` column. Default ``"__"``.

        Returns
        -------
        pd.DataFrame
            One row per group plus an ``"Overall"`` row. Columns:

            - ``<group_by>`` — group label.
            - ``n_predicted`` — total PPIs in the group.
            - ``n_assessable`` — PPIs where ``experimental_support`` is not NaN.
            - ``n_supported`` — ``experimental_support == True`` (TP).
            - ``n_unconfirmed`` — ``experimental_support == False`` (not in database,
              but assessable — may be a true interaction not yet documented).
            - ``precision`` — n_supported / (n_supported + n_unconfirmed).
            - ``n_reachable_experimental`` *(if known_ppis provided)* — reachable
              experimental PPIs used as recall denominator (same for all rows).
            - ``recall`` *(if known_ppis provided)* — TP / n_reachable_experimental.
            - ``f1`` *(if known_ppis provided)* — harmonic mean of precision and recall.

        Raises
        ------
        ValueError
            If ``group_by`` or ``experimental_support`` columns are missing.
        """
        if "experimental_support" not in validated_df.columns:
            raise ValueError(
                "validated_df must contain 'experimental_support'. "
                "Call validate_against_database first."
            )
        if group_by not in validated_df.columns:
            raise ValueError(
                f"Column '{group_by}' not found in validated_df. "
                f"Available columns: {validated_df.columns.tolist()}"
            )

        def _metrics(sub: pd.DataFrame) -> dict:
            n_predicted = len(sub)
            n_assessable = int(sub["experimental_support"].notna().sum())
            n_supported = int((sub["experimental_support"] == True).sum())
            n_unconfirmed = int((sub["experimental_support"] == False).sum())
            precision = n_supported / n_assessable if n_assessable > 0 else float("nan")
            return {
                "n_predicted": n_predicted,
                "n_assessable": n_assessable,
                "n_supported": n_supported,
                "n_unconfirmed": n_unconfirmed,
                "precision": precision,
            }

        rows = []
        for group_val, sub in validated_df.groupby(group_by, sort=False):
            row = {group_by: group_val}
            row.update(_metrics(sub))
            rows.append(row)

        overall = {group_by: "Overall"}
        overall.update(_metrics(validated_df))
        rows.append(overall)

        summary = pd.DataFrame(rows).reset_index(drop=True)

        if known_ppis is not None:
            interactome_proteins: set = set()
            for ppi_str in validated_df["PPI"]:
                parts = ppi_str.split(ppi_separator, 1)
                if len(parts) == 2:
                    interactome_proteins.update(parts)

            n_reachable = sum(
                1 for pair in known_ppis.ppis
                if pair.issubset(interactome_proteins)
            )

            def _recall_f1(tp: int, prec: float) -> tuple:
                recall = tp / n_reachable if n_reachable > 0 else float("nan")
                denom = prec + recall
                f1 = (2 * prec * recall / denom) if (denom > 0 and not (
                    pd.isna(prec) or pd.isna(recall))) else float("nan")
                return recall, f1

            summary["n_reachable_experimental"] = n_reachable
            summary["recall"] = summary.apply(
                lambda r: _recall_f1(r["n_supported"], r["precision"])[0], axis=1
            )
            summary["f1"] = summary.apply(
                lambda r: _recall_f1(r["n_supported"], r["precision"])[1], axis=1
            )

        return summary
