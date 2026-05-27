import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class _StatsMixin:
    """Generic statistical analysis and metric-based methods for InteractomeAnalyzer."""

    def _resolve_metric_col(self, preferred: str, fallback: str) -> Optional[str]:
        """Returns preferred column name if present, fallback if not, else None."""
        if self._interactome_data is None:
            return None
        if preferred in self._interactome_data.columns:
            return preferred
        if fallback in self._interactome_data.columns:
            return fallback
        return None

    def filter_by_metrics(self, criteria: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Filters the interactome by multiple metric ranges simultaneously.

        Parameters
        ----------
        criteria : Dict[str, Tuple[float, float]]
            Keys are column names, values are (min, max) inclusive ranges.
            Example: {"ipSAE_AB": (0.5, 1.0), "msa_depth": (20, 9999)}

        Returns
        -------
        pd.DataFrame
            Filtered subset of the interactome data.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()
        missing = [col for col in criteria if col not in df.columns]
        if missing:
            logger.warning(f"filter_by_metrics: columns not found and will be skipped: {missing}")

        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in criteria.items():
            if col in df.columns:
                mask &= df[col].between(lo, hi)

        result = df[mask].reset_index(drop=True)
        logger.info(f"filter_by_metrics: {len(result)}/{len(df)} rows passed the filter.")
        return result

    def get_top_interactions(
        self,
        metric: str = "ipSAE_AB",
        top_n: int = 10,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the top N interactions ranked by a given metric.

        Falls back to 'ipSAE' if the preferred column is absent.

        Parameters
        ----------
        metric : str
            Column name to rank by. Defaults to "ipSAE_AB".
        top_n : int
            Number of interactions to return.
        ascending : bool
            If True, returns the N lowest values instead.

        Returns
        -------
        pd.DataFrame
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data
        fallback = metric.replace("_AB", "") if "_AB" in metric else metric
        col = metric if metric in df.columns else (fallback if fallback in df.columns else None)

        if col is None:
            raise ValueError(f"Column '{metric}' (and fallback '{fallback}') not found in interactome data.")

        return (
            df.sort_values(col, ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )

    def summarize_by_protein(self, ppi_separator: str = "__") -> pd.DataFrame:
        """
        Generates a per-protein summary across all interactions.

        Parses the 'PPI' column to extract individual protein IDs, then aggregates:
        - degree: number of interaction partners
        - mean_ipSAE, mean_pDockQ2: average confidence metrics
        - best_partner: partner with the highest ipSAE value

        Parameters
        ----------
        ppi_separator : str
            Delimiter used in the PPI column to separate the two protein IDs.
            Defaults to "__".

        Returns
        -------
        pd.DataFrame
            One row per protein with aggregated statistics.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()
        ipsae_col = self._resolve_metric_col("ipSAE_AB", "ipSAE")
        pdockq2_col = self._resolve_metric_col("pDockQ2_AB", "pDockQ2")

        try:
            split = df["PPI"].str.split(ppi_separator, n=1, expand=True)
            df["_prot_a"] = split[0]
            df["_prot_b"] = split[1] if 1 in split.columns else split[0]
        except Exception as e:
            raise ValueError(f"Failed to parse PPI column with separator '{ppi_separator}': {e}")

        records: Dict[str, Any] = {}

        for _, row in df.iterrows():
            for prot, partner in [(row["_prot_a"], row["_prot_b"]), (row["_prot_b"], row["_prot_a"])]:
                if prot not in records:
                    records[prot] = {"ipsae_vals": [], "pdockq2_vals": [], "partners": {}}
                ipsae_val = row[ipsae_col] if ipsae_col else np.nan
                records[prot]["ipsae_vals"].append(ipsae_val)
                if pdockq2_col:
                    records[prot]["pdockq2_vals"].append(row[pdockq2_col])
                # Track best partner by ipSAE
                current_best = records[prot]["partners"].get(partner, -np.inf)
                records[prot]["partners"][partner] = max(current_best, ipsae_val if not np.isnan(ipsae_val) else -np.inf)

        summary_rows = []
        for prot, data in records.items():
            ipsae_arr = [v for v in data["ipsae_vals"] if not np.isnan(v)]
            pdockq2_arr = [v for v in data["pdockq2_vals"] if not np.isnan(v)]
            best_partner = max(data["partners"], key=data["partners"].get) if data["partners"] else None
            summary_rows.append({
                "protein": prot,
                "degree": len(data["partners"]),
                "mean_ipSAE": np.mean(ipsae_arr) if ipsae_arr else np.nan,
                "max_ipSAE": np.max(ipsae_arr) if ipsae_arr else np.nan,
                "mean_pDockQ2": np.mean(pdockq2_arr) if pdockq2_arr else np.nan,
                "best_partner": best_partner,
            })

        result = pd.DataFrame(summary_rows).sort_values("degree", ascending=False).reset_index(drop=True)
        logger.info(f"summarize_by_protein: {len(result)} unique proteins found.")
        return result

    def export_to_network(
        self,
        output_format: str = "cytoscape",
        ppi_separator: str = "__",
        extra_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Exports the interactome as an edge list for network visualization tools.

        Produces a DataFrame compatible with Cytoscape (edge list format) or Gephi
        (with 'Source'/'Target' headers). Edge attributes include ipSAE, pDockQ2,
        pLDDT_mean, msa_depth, and Tier if present.

        Parameters
        ----------
        output_format : str
            "cytoscape" or "gephi". Controls header naming convention.
        ppi_separator : str
            Delimiter to split the PPI column into source/target nodes.
        extra_cols : list of str, optional
            Additional interactome columns to include as edge attributes.

        Returns
        -------
        pd.DataFrame
            Edge list with node and attribute columns.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        try:
            split = df["PPI"].str.split(ppi_separator, n=1, expand=True)
            df["_source"] = split[0]
            df["_target"] = split[1] if 1 in split.columns else split[0]
        except Exception as e:
            raise ValueError(f"Failed to parse PPI column with separator '{ppi_separator}': {e}")

        src_col, tgt_col = ("Source", "Target") if output_format == "gephi" else ("source", "target")

        default_attrs = ["ipSAE_AB", "ipSAE", "pDockQ2_AB", "pDockQ2",
                         "pLDDT_mean", "msa_depth", "Tier"]
        attr_cols = [c for c in default_attrs if c in df.columns]
        if extra_cols:
            attr_cols += [c for c in extra_cols if c in df.columns and c not in attr_cols]

        edge_df = df[["_source", "_target"] + attr_cols].rename(
            columns={"_source": src_col, "_target": tgt_col}
        ).reset_index(drop=True)

        logger.info(f"export_to_network: exported {len(edge_df)} edges with {len(attr_cols)} attributes.")
        return edge_df

    def compare_engines(
        self,
        other_df: pd.DataFrame,
        suffix_self: str = "_a",
        suffix_other: str = "_b",
        on: str = "PPI",
    ) -> pd.DataFrame:
        """
        Compares metrics between two interactome runs (e.g., AF3 vs Boltz2).

        Merges the loaded interactome with a second DataFrame on the PPI key,
        computing the delta for each shared numeric metric.

        Parameters
        ----------
        other_df : pd.DataFrame
            Second interactome result DataFrame. Must contain a 'PPI' column.
        suffix_self : str
            Suffix appended to columns from the loaded (self) interactome.
        suffix_other : str
            Suffix appended to columns from `other_df`.
        on : str
            Join key column name. Defaults to "PPI".

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with per-metric delta columns (prefix 'delta_').
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")
        if on not in other_df.columns:
            raise ValueError(f"Column '{on}' not found in other_df.")

        merged = pd.merge(
            self._interactome_data,
            other_df,
            on=on,
            suffixes=(suffix_self, suffix_other),
            how="inner",
        )

        # Compute deltas for numeric columns that appear in both
        base_cols = set(self._interactome_data.columns) - {on}
        other_cols = set(other_df.columns) - {on}
        shared_numeric = [
            c for c in base_cols & other_cols
            if pd.api.types.is_numeric_dtype(self._interactome_data[c])
        ]

        for col in shared_numeric:
            try:
                merged[f"delta_{col}"] = merged[f"{col}{suffix_self}"] - merged[f"{col}{suffix_other}"]
            except KeyError:
                pass  # column may have been renamed or absent after merge

        logger.info(
            f"compare_engines: {len(merged)} PPIs in common, "
            f"{len(shared_numeric)} numeric metrics compared."
        )
        return merged

    def cluster_interactome_by_metrics(
        self,
        n_clusters: int = 4,
        metric_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Applies K-Means clustering to group interactions by their metric profile.

        Discovers non-obvious interaction patterns beyond single-threshold filtering.
        Uses StandardScaler normalization before clustering to handle metric heterogeneity.

        Parameters
        ----------
        n_clusters : int
            Number of K-Means clusters. Defaults to 4.
        metric_cols : list of str, optional
            Columns to use as feature vector. If None, auto-selects all available
            numeric columns among: ipSAE_AB, pDockQ2_AB, pLDDT_mean, msa_depth,
            ipTM, pTM, and their non-suffixed variants.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Interactome data with an added 'km_cluster' column.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for cluster_interactome_by_metrics.")

        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        df = self._interactome_data.copy()

        if metric_cols is None:
            candidates = [
                "ipSAE_AB", "ipSAE", "pDockQ2_AB", "pDockQ2",
                "pLDDT_mean", "msa_depth", "ipTM", "pTM",
                "ipSAE_d0_dom_AB", "ipSAE_d0dom_AB",
            ]
            metric_cols = [c for c in candidates if c in df.columns]
        else:
            metric_cols = [c for c in metric_cols if c in df.columns]

        if not metric_cols:
            raise ValueError("No valid metric columns found for clustering.")

        feature_matrix = df[metric_cols].copy()
        # Drop rows with all-NaN features; impute remaining NaNs with column median
        feature_matrix = feature_matrix.dropna(how="all")
        feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True))

        if len(feature_matrix) < n_clusters:
            raise ValueError(
                f"Not enough valid rows ({len(feature_matrix)}) for {n_clusters} clusters."
            )

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(X)

        result = df.loc[feature_matrix.index].copy()
        result["km_cluster"] = labels

        logger.info(
            f"cluster_interactome_by_metrics: {n_clusters} clusters on {len(feature_matrix)} rows "
            f"using features {metric_cols}."
        )
        logger.info(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")
        return result.reset_index(drop=True)
