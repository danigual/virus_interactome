import pytest
import numpy as np
import pandas as pd
from virus_interactome.interactome import InteractomeAnalyzer


# ---------------------------------------------------------------------------
# Property setter/getter tests
# ---------------------------------------------------------------------------

class TestAnalyzerProperties:
    def test_cannot_set_interactome_data_directly(self, analyzer_with_data):
        with pytest.raises(AttributeError):
            analyzer_with_data.interactome_data = pd.DataFrame()

    def test_cannot_set_cluster_data_directly(self, analyzer_with_data):
        with pytest.raises(AttributeError):
            analyzer_with_data.cluster_data = pd.DataFrame()

    def test_cannot_set_bad_interactome_path(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.interactome_path = "/nonexistent/path.csv"

    def test_cannot_set_bad_cluster_path(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.cluster_path = "/nonexistent/path.csv"

    def test_cannot_set_bad_interactome_format(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col1,col2\n1,2\n")
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.interactome_path = str(bad_csv)

    def test_cannot_set_bad_cluster_format(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col1,col2\n1,2\n")
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.cluster_path = str(bad_csv)

    def test_empty_interactome_raises(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("PPI,Folder\n")
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.interactome_path = str(empty_csv)

    def test_cluster_with_foreign_ppi(self, dummy_interactome_csv, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        cluster_csv = tmp_path / "foreign_clusters.csv"
        cluster_csv.write_text("PPI,path,cluster_id\nFOREIGN__PPI,/fake/path,0\n")
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(dummy_interactome_csv)
        analyzer.cluster_path = str(cluster_csv)
        assert "NOT present in the interactome" in caplog.text

    def test_interactome_data_loaded(self, analyzer_with_data):
        assert analyzer_with_data.interactome_data is not None
        assert len(analyzer_with_data.interactome_data) == 10

    def test_cluster_data_loaded(self, analyzer_with_data):
        assert analyzer_with_data.cluster_data is not None
        assert len(analyzer_with_data.cluster_data) == 15

    def test_len(self, analyzer_with_data):
        assert len(analyzer_with_data) == 10

    def test_str(self, analyzer_with_data):
        s = str(analyzer_with_data)
        assert "InteractomeAnalyzer" in s


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestConfidenceTiers:
    def test_defaults(self, analyzer_with_data):
        result = analyzer_with_data.get_confidence_tiers()
        assert "Tier" in result.columns
        assert len(result) == 10
        valid_tiers = {
            "Tier 1 (High Confidence)",
            "Tier 2 (Specific/Novel)",
            "Tier 3 (Weak/Dynamic)",
            "Low Confidence",
        }
        assert set(result["Tier"].unique()).issubset(valid_tiers)

    def test_custom_thresholds(self, analyzer_with_data):
        result = analyzer_with_data.get_confidence_tiers(
            ipsae_threshold=0.0, pdockq2_threshold=0.0, msa_threshold=0
        )
        assert all(result["Tier"].str.startswith("Tier 1"))

    def test_strict_thresholds(self, analyzer_with_data):
        result = analyzer_with_data.get_confidence_tiers(
            ipsae_threshold=0.99, pdockq2_threshold=0.99, msa_threshold=9999
        )
        assert all(result["Tier"] == "Low Confidence")

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.get_confidence_tiers()


# ---------------------------------------------------------------------------
# filter_by_metrics
# ---------------------------------------------------------------------------

class TestFilterByMetrics:
    def test_single_range(self, analyzer_with_data):
        result = analyzer_with_data.filter_by_metrics({"ipSAE_AB": (0.5, 1.0)})
        assert all(result["ipSAE_AB"] >= 0.5)
        assert all(result["ipSAE_AB"] <= 1.0)

    def test_multi_range(self, analyzer_with_data):
        result = analyzer_with_data.filter_by_metrics({
            "ipSAE_AB": (0.3, 1.0),
            "pDockQ2_AB": (0.2, 1.0),
        })
        assert all(result["ipSAE_AB"] >= 0.3)
        assert all(result["pDockQ2_AB"] >= 0.2)

    def test_missing_column_warns(self, analyzer_with_data, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        result = analyzer_with_data.filter_by_metrics({"nonexistent_col": (0, 1)})
        assert "not found" in caplog.text
        assert len(result) == 10

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.filter_by_metrics({"ipSAE_AB": (0.5, 1.0)})

    def test_empty_result(self, analyzer_with_data):
        result = analyzer_with_data.filter_by_metrics({"ipSAE_AB": (99.0, 100.0)})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_top_interactions
# ---------------------------------------------------------------------------

class TestGetTopInteractions:
    def test_default(self, analyzer_with_data):
        result = analyzer_with_data.get_top_interactions(top_n=5)
        assert len(result) == 5
        assert list(result["ipSAE_AB"]) == sorted(result["ipSAE_AB"], reverse=True)

    def test_ascending(self, analyzer_with_data):
        result = analyzer_with_data.get_top_interactions(top_n=3, ascending=True)
        assert len(result) == 3
        assert list(result["ipSAE_AB"]) == sorted(result["ipSAE_AB"])

    def test_column_fallback(self, dummy_interactome_csv):
        df = pd.read_csv(dummy_interactome_csv)
        df = df.rename(columns={"ipSAE_AB": "ipSAE"})
        new_csv = dummy_interactome_csv.parent / "fallback.csv"
        df.to_csv(new_csv, index=False)

        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(new_csv)
        result = analyzer.get_top_interactions(metric="ipSAE_AB", top_n=3)
        assert len(result) == 3

    def test_missing_column_raises(self, analyzer_with_data):
        with pytest.raises(ValueError, match="not found"):
            analyzer_with_data.get_top_interactions(metric="totally_fake_column")

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.get_top_interactions()


# ---------------------------------------------------------------------------
# summarize_by_protein
# ---------------------------------------------------------------------------

class TestSummarizeByProtein:
    def test_basic(self, analyzer_with_data):
        result = analyzer_with_data.summarize_by_protein()
        expected_cols = {"protein", "degree", "mean_ipSAE", "max_ipSAE", "mean_pDockQ2", "best_partner"}
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == 6

    def test_degree_counts(self, analyzer_with_data):
        result = analyzer_with_data.summarize_by_protein()
        prot_a = result[result["protein"] == "ProtA"]
        assert prot_a["degree"].iloc[0] == 3

    def test_best_partner_not_null(self, analyzer_with_data):
        result = analyzer_with_data.summarize_by_protein()
        assert result["best_partner"].notna().all()

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.summarize_by_protein()


# ---------------------------------------------------------------------------
# export_to_network
# ---------------------------------------------------------------------------

class TestExportToNetwork:
    def test_cytoscape(self, analyzer_with_data):
        result = analyzer_with_data.export_to_network(output_format="cytoscape")
        assert "source" in result.columns
        assert "target" in result.columns
        assert len(result) == 10

    def test_gephi(self, analyzer_with_data):
        result = analyzer_with_data.export_to_network(output_format="gephi")
        assert "Source" in result.columns
        assert "Target" in result.columns

    def test_extra_cols(self, analyzer_with_data):
        result = analyzer_with_data.export_to_network(extra_cols=["ipTM", "pTM"])
        assert "ipTM" in result.columns
        assert "pTM" in result.columns

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.export_to_network()


# ---------------------------------------------------------------------------
# compare_engines
# ---------------------------------------------------------------------------

class TestCompareEngines:
    def test_basic_merge(self, analyzer_with_data, dummy_interactome_df):
        other_df = dummy_interactome_df.copy()
        other_df["ipSAE_AB"] = other_df["ipSAE_AB"] + 0.1
        result = analyzer_with_data.compare_engines(other_df)
        assert len(result) == 10
        delta_cols = [c for c in result.columns if c.startswith("delta_")]
        assert len(delta_cols) > 0

    def test_missing_on_column_raises(self, analyzer_with_data):
        bad_df = pd.DataFrame({"not_PPI": ["a", "b"]})
        with pytest.raises(ValueError, match="not found"):
            analyzer_with_data.compare_engines(bad_df)

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.compare_engines(pd.DataFrame({"PPI": ["a"]}))


# ---------------------------------------------------------------------------
# cluster_interactome_by_metrics
# ---------------------------------------------------------------------------

class TestClusterByMetrics:
    def test_basic(self, analyzer_with_data):
        result = analyzer_with_data.cluster_interactome_by_metrics(n_clusters=3)
        assert "km_cluster" in result.columns
        assert result["km_cluster"].nunique() == 3

    def test_deterministic(self, analyzer_with_data):
        r1 = analyzer_with_data.cluster_interactome_by_metrics(n_clusters=3, random_state=42)
        r2 = analyzer_with_data.cluster_interactome_by_metrics(n_clusters=3, random_state=42)
        assert list(r1["km_cluster"]) == list(r2["km_cluster"])

    def test_custom_cols(self, analyzer_with_data):
        result = analyzer_with_data.cluster_interactome_by_metrics(
            n_clusters=2, metric_cols=["ipSAE_AB", "pDockQ2_AB"]
        )
        assert "km_cluster" in result.columns
        assert result["km_cluster"].nunique() == 2

    def test_no_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="not loaded"):
            analyzer.cluster_interactome_by_metrics()
