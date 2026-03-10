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

    def test_no_valid_metric_cols_raises(self, analyzer_with_data):
        with pytest.raises(ValueError, match="No valid metric columns"):
            analyzer_with_data.cluster_interactome_by_metrics(metric_cols=["NONEXISTENT_COL"])

    def test_too_few_rows_raises(self, dummy_interactome_csv):
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(dummy_interactome_csv)
        with pytest.raises(ValueError, match="Not enough valid rows"):
            analyzer.cluster_interactome_by_metrics(n_clusters=999)


# ---------------------------------------------------------------------------
# _resolve_metric_col
# ---------------------------------------------------------------------------

class TestResolveMetricCol:
    def test_none_when_no_data(self):
        analyzer = InteractomeAnalyzer()
        assert analyzer._resolve_metric_col("ipSAE_AB", "ipSAE") is None

    def test_preferred_exists(self, analyzer_with_data):
        result = analyzer_with_data._resolve_metric_col("ipSAE_AB", "ipSAE")
        assert result == "ipSAE_AB"

    def test_fallback_when_preferred_missing(self, analyzer_with_data):
        result = analyzer_with_data._resolve_metric_col("NONEXISTENT", "ipSAE_AB")
        assert result == "ipSAE_AB"

    def test_none_when_both_missing(self, analyzer_with_data):
        result = analyzer_with_data._resolve_metric_col("XXX", "YYY")
        assert result is None


# ---------------------------------------------------------------------------
# Additional property / getter tests
# ---------------------------------------------------------------------------

class TestAnalyzerAdditionalProperties:
    def test_binder_data_default_none(self):
        analyzer = InteractomeAnalyzer()
        assert analyzer.binder_data is None

    def test_binder_data_setter(self):
        analyzer = InteractomeAnalyzer()
        df = pd.DataFrame({"col": [1, 2]})
        analyzer.binder_data = df
        pd.testing.assert_frame_equal(analyzer.binder_data, df)

    def test_interactome_path_default_none(self):
        analyzer = InteractomeAnalyzer()
        assert analyzer.interactome_path is None

    def test_cluster_path_default_none(self):
        analyzer = InteractomeAnalyzer()
        assert analyzer.cluster_path is None

    def test_models_path_default_none(self):
        analyzer = InteractomeAnalyzer()
        assert analyzer.models_path is None

    def test_interactome_path_getter(self, analyzer_with_data, dummy_interactome_csv):
        assert analyzer_with_data.interactome_path == dummy_interactome_csv

    def test_cluster_path_getter(self, analyzer_with_data, dummy_cluster_csv):
        assert analyzer_with_data.cluster_path == dummy_cluster_csv

    def test_len_returns_record_count(self, analyzer_with_data):
        assert len(analyzer_with_data) == 10

    def test_str_representation(self, analyzer_with_data):
        s = str(analyzer_with_data)
        assert "InteractomeAnalyzer" in s


# ---------------------------------------------------------------------------
# models_path setter — path relocation
# ---------------------------------------------------------------------------

class TestModelsPathSetter:
    def test_models_path_relocation(self, analyzer_with_data):
        old = analyzer_with_data.models_path or ""
        analyzer_with_data.models_path = "/new/root"
        assert analyzer_with_data.models_path == "/new/root"
        # Cluster paths updated
        for p in analyzer_with_data.cluster_data["path"]:
            assert "/new/root" in str(p) or old == ""

    def test_models_path_without_data_raises(self):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(RuntimeError, match="Data not loaded"):
            analyzer.models_path = "/new/root"


# ---------------------------------------------------------------------------
# get_confidence_tiers — edge cases
# ---------------------------------------------------------------------------

class TestConfidenceTiersEdgeCases:
    def test_unknown_tier_when_columns_missing(self, tmp_path):
        """Rows with missing metric columns get 'Unknown' tier."""
        df = pd.DataFrame({
            "PPI": ["A__B"],
            "Folder": ["/fake"],
            "other_col": [99],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        result = analyzer.get_confidence_tiers()
        assert result["Tier"].iloc[0] == "Unknown"


# ---------------------------------------------------------------------------
# run_full_pipeline — cluster_data not loaded early return
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    def test_no_cluster_data_warns(self, dummy_interactome_csv, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(dummy_interactome_csv)
        analyzer.run_full_pipeline()
        assert "Cluster data is missing" in caplog.text


# ---------------------------------------------------------------------------
# Empty interactome / cluster file edge cases
# ---------------------------------------------------------------------------

class TestEmptyFileEdgeCases:
    def test_empty_interactome_csv_raises(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("PPI,Folder\n")  # header only, no data
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.interactome_path = str(csv)

    def test_empty_cluster_csv_raises(self, tmp_path):
        csv = tmp_path / "empty_cluster.csv"
        csv.write_text("PPI,cluster_id\n")
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.cluster_path = str(csv)
