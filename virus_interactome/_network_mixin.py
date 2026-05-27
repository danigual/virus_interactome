import logging
import pandas as pd
import numpy as np
from typing import Any, Optional, Union
from pathlib import Path

from .plotting import plot_network as _plot_network_figure

logger = logging.getLogger(__name__)


class _NetworkMixin:
    """Network topology analysis methods for InteractomeAnalyzer."""

    # Non-numeric columns excluded from rank aggregation
    _META_COLS: frozenset = frozenset({
        "PPI", "ORF_A", "ORF_B", "Folder", "Path",
        "LIR_AB", "cLIR_AB", "pool_id", "Tier", "LIS_Tier", "iLIS_Tier",
    })

    def _aggregate_per_ppi(
        self,
        model_agg: str = "mean",
        weight_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Collapse multiple model-rank rows into one row per PPI.

        Parameters
        ----------
        model_agg : str
            ``"mean"`` — average all numeric columns across ranks.
            ``"max"``  — take the per-column maximum across ranks.
            ``"best"`` — keep the single rank row with the highest ``weight_col``.
        weight_col : str, optional
            Required only for ``model_agg="best"``. The column used to pick the
            best rank.

        Returns
        -------
        pd.DataFrame
            One row per unique PPI with string meta columns preserved and
            numeric columns aggregated.

        Raises
        ------
        ValueError
            If ``model_agg`` is invalid, or ``weight_col`` is missing when
            ``model_agg="best"``.
        """
        df = self._interactome_data.copy()

        if model_agg not in ("mean", "max", "best"):
            raise ValueError(f"model_agg must be 'mean', 'max', or 'best'. Got: '{model_agg}'")
        if model_agg == "best" and (weight_col is None or weight_col not in df.columns):
            raise ValueError(
                f"model_agg='best' requires a valid weight_col. Got: '{weight_col}'"
            )

        num_cols = [
            c for c in df.columns
            if c not in self._META_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        meta_present = [c for c in df.columns if c != "PPI" and c not in num_cols]

        if model_agg == "best":
            return df.loc[df.groupby("PPI")[weight_col].idxmax()].reset_index(drop=True)

        meta = df.groupby("PPI")[meta_present].first().reset_index() if meta_present \
            else df.groupby("PPI")[[]].first().reset_index()
        agg_fn = df.groupby("PPI")[num_cols].mean() if model_agg == "mean" \
            else df.groupby("PPI")[num_cols].max()
        return meta.merge(agg_fn.reset_index(), on="PPI")

    def _build_ppi_graph(
        self,
        weight_col: str,
        min_weight: float,
        ppi_separator: str,
        model_agg: str,
    ) -> Any:
        """Build a weighted undirected NetworkX graph from _interactome_data."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        if weight_col not in self._interactome_data.columns:
            raise ValueError(f"Column '{weight_col}' not found in interactome data.")

        ppi_df = self._aggregate_per_ppi(model_agg=model_agg, weight_col=weight_col)

        G = nx.Graph()
        for _, row in ppi_df.iterrows():
            parts = str(row["PPI"]).split(ppi_separator, 1)
            if len(parts) != 2:
                logger.warning(f"Cannot parse PPI '{row['PPI']}' with separator '{ppi_separator}'. Skipping.")
                continue
            source, target = parts
            w = float(row[weight_col]) if not pd.isna(row[weight_col]) else 0.0
            if w > min_weight:
                G.add_edge(source, target, weight=w)

        return G

    def compute_network_properties(
        self,
        weight_col: str = "Best_iLIS",
        min_weight: float = 0.0,
        ppi_separator: str = "__",
        model_agg: str = "mean",
    ) -> pd.DataFrame:
        """Compute graph-theory metrics for every protein in the interactome.

        Aggregates multiple model-rank rows per PPI using ``model_agg``, builds a
        weighted undirected NetworkX graph, and returns per-protein centrality metrics.
        Hubs and bottlenecks are identified using adaptive thresholds (mean + 1σ of
        degree and betweenness distributions respectively), making the classification
        valid for any proteome size.

        Parameters
        ----------
        weight_col : str
            Numeric column to use as edge weight. Defaults to ``"Best_iLIS"``.
        min_weight : float
            Edges with weight ≤ min_weight are excluded. Defaults to ``0.0``
            (keeps all pairs with any detectable interaction signal).
        ppi_separator : str
            Separator used in the PPI column. Defaults to ``"__"``.
        model_agg : str
            Strategy to collapse multiple model-rank rows per PPI:
            ``"mean"`` (average across ranks), ``"max"`` (take maximum),
            or ``"best"`` (keep the rank with the highest weight_col value).
            Defaults to ``"mean"``.

        Returns
        -------
        pd.DataFrame
            One row per protein with columns: ``protein``, ``degree``,
            ``weighted_degree``, ``betweenness_centrality``,
            ``closeness_centrality``, ``eigenvector_centrality``,
            ``clustering_coefficient``, ``is_hub``, ``is_bottleneck``.
            Sorted descending by ``degree``.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        ValueError
            If ``weight_col`` is absent or ``model_agg`` is invalid.
        ImportError
            If ``networkx`` is not installed.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        G = self._build_ppi_graph(weight_col, min_weight, ppi_separator, model_agg)

        if G.number_of_nodes() == 0:
            logger.warning("compute_network_properties: graph has no nodes after filtering.")
            return pd.DataFrame(columns=[
                "protein", "degree", "weighted_degree",
                "betweenness_centrality", "closeness_centrality",
                "eigenvector_centrality", "clustering_coefficient",
                "is_hub", "is_bottleneck",
            ])

        import networkx as nx

        degrees = dict(G.degree())
        weighted_degrees = dict(G.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(G)

        try:
            eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge; values set to NaN.")
            eigenvector = {n: np.nan for n in G.nodes()}

        clustering = nx.clustering(G, weight="weight")

        deg_vals = np.array(list(degrees.values()), dtype=float)
        bet_vals = np.array(list(betweenness.values()), dtype=float)
        hub_threshold = float(deg_vals.mean() + deg_vals.std())
        bottleneck_threshold = float(bet_vals.mean() + bet_vals.std())

        rows = [
            {
                "protein": node,
                "degree": degrees[node],
                "weighted_degree": round(weighted_degrees[node], 4),
                "betweenness_centrality": round(betweenness[node], 4),
                "closeness_centrality": round(closeness[node], 4),
                "eigenvector_centrality": round(eigenvector[node], 4)
                    if not np.isnan(eigenvector[node]) else np.nan,
                "clustering_coefficient": round(clustering[node], 4),
                "is_hub": bool(degrees[node] > hub_threshold),
                "is_bottleneck": bool(betweenness[node] > bottleneck_threshold),
            }
            for node in G.nodes()
        ]

        result = (
            pd.DataFrame(rows)
            .sort_values("degree", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            f"compute_network_properties: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges. "
            f"Hubs: {result['is_hub'].sum()}, Bottlenecks: {result['is_bottleneck'].sum()}."
        )
        return result

    def plot_network(
        self,
        network_df: Optional[pd.DataFrame] = None,
        color_by: str = "betweenness_centrality",
        size_by: str = "degree",
        weight_col: str = "Best_iLIS",
        min_weight: float = 0.0,
        ppi_separator: str = "__",
        model_agg: str = "mean",
        label_top_n: int = 5,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Interactome Network",
    ) -> None:
        """Visualise the PPI network using a force-directed spring layout.

        Node size encodes ``size_by``, node colour encodes ``color_by`` (viridis
        colormap). Edge width scales with interaction weight. Only the top
        ``label_top_n`` nodes by ``size_by`` are labelled to avoid clutter.
        Hub and bottleneck nodes are outlined in red and blue respectively.

        Parameters
        ----------
        network_df : pd.DataFrame, optional
            Pre-computed output of :meth:`compute_network_properties`. If ``None``,
            it is computed using ``weight_col``, ``min_weight``, ``ppi_separator``,
            and ``model_agg``.
        color_by : str
            Node attribute column for colour mapping. Defaults to
            ``"betweenness_centrality"``.
        size_by : str
            Node attribute column for size scaling. Defaults to ``"degree"``.
        weight_col : str
            Edge weight column (used when rebuilding the graph). Defaults to
            ``"Best_iLIS"``.
        min_weight : float
            Edge weight threshold (used when rebuilding the graph).
        ppi_separator : str
            PPI column separator.
        model_agg : str
            Model-rank aggregation strategy.
        label_top_n : int
            Number of highest-``size_by`` nodes to label. Defaults to ``5``.
        output_path : str or Path, optional
            If provided, saves the figure to this path (300 dpi). Otherwise
            calls ``plt.show()``.
        title : str
            Figure title.

        Raises
        ------
        RuntimeError
            If interactome data is not loaded.
        ImportError
            If ``networkx`` or ``matplotlib`` is not installed.
        """
        if self._interactome_data is None:
            raise RuntimeError("Interactome data not loaded. Set interactome_path first.")

        if network_df is None:
            network_df = self.compute_network_properties(
                weight_col=weight_col,
                min_weight=min_weight,
                ppi_separator=ppi_separator,
                model_agg=model_agg,
            )

        G = self._build_ppi_graph(weight_col, min_weight, ppi_separator, model_agg)
        _plot_network_figure(
            G, network_df,
            color_by=color_by, size_by=size_by,
            label_top_n=label_top_n, output_path=output_path, title=title,
        )
