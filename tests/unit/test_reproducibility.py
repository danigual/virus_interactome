"""Phase 5 — Reproducibility tests for key methods."""

import pytest
import numpy as np
import pandas as pd
from virus_interactome.interactome import InteractomeAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(df: pd.DataFrame) -> InteractomeAnalyzer:
    analyzer = InteractomeAnalyzer()
    analyzer._interactome_data = df.copy()
    return analyzer


# ---------------------------------------------------------------------------
# filter_by_metrics — order independence
# ---------------------------------------------------------------------------

def test_filter_order_independence(dummy_interactome_df):
    shuffled = dummy_interactome_df.sample(frac=1, random_state=7).reset_index(drop=True)

    a = _make_analyzer(dummy_interactome_df)
    b = _make_analyzer(shuffled)

    criteria = {"ipSAE_AB": (0.3, 1.0), "pLDDT_mean": (50.0, 100.0)}
    result_a = set(a.filter_by_metrics(criteria)["PPI"])
    result_b = set(b.filter_by_metrics(criteria)["PPI"])

    assert result_a == result_b


# ---------------------------------------------------------------------------
# get_top_interactions — order independence
# ---------------------------------------------------------------------------

def test_top_interactions_order_independence(dummy_interactome_df):
    shuffled = dummy_interactome_df.sample(frac=1, random_state=13).reset_index(drop=True)

    a = _make_analyzer(dummy_interactome_df)
    b = _make_analyzer(shuffled)

    top_a = set(a.get_top_interactions(metric="ipSAE_AB", top_n=5)["PPI"])
    top_b = set(b.get_top_interactions(metric="ipSAE_AB", top_n=5)["PPI"])

    assert top_a == top_b


# ---------------------------------------------------------------------------
# get_confidence_tiers — order independence
# ---------------------------------------------------------------------------

def test_tier_order_independence(dummy_interactome_df):
    shuffled = dummy_interactome_df.sample(frac=1, random_state=99).reset_index(drop=True)

    a = _make_analyzer(dummy_interactome_df)
    b = _make_analyzer(shuffled)

    df_a = a.get_confidence_tiers()
    df_b = b.get_confidence_tiers()

    # Each PPI should land in the same tier regardless of input row order
    mapping_a = df_a.set_index("PPI")["Tier"].to_dict()
    mapping_b = df_b.set_index("PPI")["Tier"].to_dict()
    assert mapping_a == mapping_b


# ---------------------------------------------------------------------------
# cluster_interactome_by_metrics — determinism with fixed random_state
# ---------------------------------------------------------------------------

def test_cluster_determinism(dummy_interactome_df):
    a = _make_analyzer(dummy_interactome_df)
    b = _make_analyzer(dummy_interactome_df)

    result_a = a.cluster_interactome_by_metrics(n_clusters=3, random_state=42)
    result_b = b.cluster_interactome_by_metrics(n_clusters=3, random_state=42)

    pd.testing.assert_series_equal(
        result_a["km_cluster"].reset_index(drop=True),
        result_b["km_cluster"].reset_index(drop=True),
    )


def test_summarize_by_protein_order_independence(dummy_interactome_df):
    """summarize_by_protein aggregations must be identical regardless of row order."""
    shuffled = dummy_interactome_df.sample(frac=1, random_state=5).reset_index(drop=True)

    a = _make_analyzer(dummy_interactome_df)
    b = _make_analyzer(shuffled)

    summary_a = a.summarize_by_protein().sort_values("protein").reset_index(drop=True)
    summary_b = b.summarize_by_protein().sort_values("protein").reset_index(drop=True)

    pd.testing.assert_frame_equal(summary_a, summary_b)
