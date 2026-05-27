import io
import json
import tarfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from virus_interactome.interactome import InteractomeAnalyzer, InteractomeProcessor
from virus_interactome.foldseek import FoldseekClient
from virus_interactome.databases import DatabaseClient


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

    def test_lis_tier_high_confidence(self, tmp_path):
        """LIS_Tier == 'High Confidence' when Best_LIS >= 0.203 and Best_LIA >= 3432."""
        df = pd.DataFrame({
            "PPI": ["A__B"], "Folder": ["/fake"],
            "ipSAE_AB": [0.6], "pDockQ2_AB": [0.3], "msa_depth": [30],
            "Best_LIS": [0.25], "Best_LIA": [4000.0],
            "Best_iLIS": [0.30],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        result = analyzer.get_confidence_tiers()
        assert result["LIS_Tier"].iloc[0] == "High Confidence"
        assert result["iLIS_Tier"].iloc[0] == "High Confidence"

    def test_lis_tier_low_confidence(self, tmp_path):
        """LIS_Tier == 'Low Confidence' when Best_LIS < 0.203."""
        df = pd.DataFrame({
            "PPI": ["A__B"], "Folder": ["/fake"],
            "ipSAE_AB": [0.3], "pDockQ2_AB": [0.1], "msa_depth": [5],
            "Best_LIS": [0.05], "Best_LIA": [100.0],
            "Best_iLIS": [0.01],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        result = analyzer.get_confidence_tiers()
        assert result["LIS_Tier"].iloc[0] == "Low Confidence"
        assert result["iLIS_Tier"].iloc[0] == "Low Confidence"

    def test_lis_tier_low_lia(self, tmp_path):
        """LIS_Tier == 'Low LIA' when Best_LIS passes but Best_LIA does not."""
        df = pd.DataFrame({
            "PPI": ["A__B"], "Folder": ["/fake"],
            "ipSAE_AB": [0.6], "pDockQ2_AB": [0.3], "msa_depth": [30],
            "Best_LIS": [0.30], "Best_LIA": [100.0],
            "Best_iLIS": [0.10],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        result = analyzer.get_confidence_tiers()
        assert result["LIS_Tier"].iloc[0] == "Low LIA"

    def test_lis_tier_na_when_columns_absent(self, tmp_path, caplog):
        """LIS_Tier and iLIS_Tier are 'N/A' when Best_LIS/Best_iLIS columns absent."""
        import logging
        df = pd.DataFrame({
            "PPI": ["A__B"], "Folder": ["/fake"],
            "ipSAE_AB": [0.6], "pDockQ2_AB": [0.3], "msa_depth": [30],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        caplog.set_level(logging.WARNING)
        result = analyzer.get_confidence_tiers()
        assert result["LIS_Tier"].iloc[0] == "N/A"
        assert result["iLIS_Tier"].iloc[0] == "N/A"
        assert "Best_LIS" in caplog.text

    def test_tier_columns_independent(self, tmp_path):
        """Tier and LIS_Tier can disagree — they are independent classifications."""
        df = pd.DataFrame({
            "PPI": ["A__B"], "Folder": ["/fake"],
            # ipSAE says Low Confidence
            "ipSAE_AB": [0.3], "pDockQ2_AB": [0.1], "msa_depth": [5],
            # LIS says High Confidence
            "Best_LIS": [0.25], "Best_LIA": [5000.0],
            "Best_iLIS": [0.30],
        })
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        result = analyzer.get_confidence_tiers()
        assert result["Tier"].iloc[0] == "Low Confidence"
        assert result["LIS_Tier"].iloc[0] == "High Confidence"


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

class TestLenNoData:
    def test_len_returns_zero_when_no_data(self):
        """Line 2418: __len__ returns 0 when interactome_data is not loaded."""
        assert len(InteractomeAnalyzer()) == 0


class TestClusterPathMixedPaths:
    def test_mixed_abs_relative_paths_warns_and_clears_models_path(self, tmp_path, caplog):
        """Lines 2358-2360: os.path.commonpath raises ValueError for mixed paths → models_path=""."""
        import logging
        caplog.set_level(logging.WARNING)
        cluster_csv = tmp_path / "clusters.csv"
        cluster_csv.write_text(
            "PPI,path,cluster_id\n"
            "A__B,/abs/path/model_0.cif,0\n"
            "A__B,relative/path/model_1.cif,1\n"
        )
        analyzer = InteractomeAnalyzer()
        analyzer.cluster_path = str(cluster_csv)
        assert analyzer.models_path == ""
        assert "Could not automatically determine" in caplog.text


class TestRunFullPipelineIpsaeFilter:
    def test_ipsae_filter_logs_report(self, analyzer_with_data, caplog, monkeypatch):
        """Lines 2438-2457: ipsae_filter branch logs count of high-confidence PPIs."""
        import logging
        caplog.set_level(logging.INFO)
        # Bypass analyze_peptide_proteins_pairs (needs MoleculeKit + real CIF files)
        monkeypatch.setattr(analyzer_with_data, "analyze_peptide_proteins_pairs", lambda **kw: None)
        analyzer_with_data.run_full_pipeline(ipsae_filter=0.3)
        assert "PPIs are above" in caplog.text

    def test_ipsae_filter_col_not_found_skips_log(self, tmp_path, monkeypatch):
        """ipsae_col resolution skips logging when no ipSAE column present."""
        import logging
        df = pd.DataFrame({"PPI": ["A__B"], "Folder": ["/fake"], "other": [1.0]})
        csv = tmp_path / "i.csv"
        df.to_csv(csv, index=False)
        cluster_csv = tmp_path / "c.csv"
        cluster_csv.write_text("PPI,path,cluster_id\nA__B,/fake/m.cif,0\n")
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(csv)
        analyzer.cluster_path = str(cluster_csv)
        monkeypatch.setattr(analyzer, "analyze_peptide_proteins_pairs", lambda **kw: None)
        # Should complete without error even though no ipSAE column exists
        analyzer.run_full_pipeline(ipsae_filter=0.5)


class TestGetCandidateClusters:
    @pytest.fixture
    def analyzer_with_geometry_clusters(self, dummy_interactome_csv, tmp_path):
        """Analyzer with cluster data that has full geometry columns."""
        cluster_csv = tmp_path / "geo_clusters.csv"
        cluster_csv.write_text(
            "PPI,path,cluster_id,model_num,cluster_ratio,x_len,y_len,x_min,x_max,y_min,y_max\n"
            # High ratio, x > y → Binder=A=ProtA, Peptide=B=ProtB
            "ProtA__ProtB,/fake/m1.cif,0,1,10.0,50,5,0,50,100,105\n"
            # High ratio, y > x → Binder=B=ProtB, Peptide=A=ProtA
            "ProtA__ProtB,/fake/m2.cif,1,2,9.0,5,40,0,5,100,140\n"
            # Low ratio → excluded
            "ProtA__ProtC,/fake/m3.cif,0,1,2.0,10,10,0,10,0,10\n"
            # x_len < min_peptide_len (3 < 5) → excluded even with high ratio
            "ProtA__ProtC,/fake/m4.cif,1,2,8.0,3,40,0,3,0,40\n"
        )
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(dummy_interactome_csv)
        analyzer.cluster_path = str(cluster_csv)
        return analyzer

    def test_high_ratio_rows_included(self, analyzer_with_geometry_clusters):
        result = analyzer_with_geometry_clusters._get_candidate_clusters(cluster_ratio_threshold=7.0)
        # Only rows 1 and 2 have ratio > 7 AND x_len/y_len >= 5
        assert len(result) == 2

    def test_binder_assignment_x_longer(self, analyzer_with_geometry_clusters):
        """x_len > y_len → Binder=chain A, Peptide=chain B."""
        result = analyzer_with_geometry_clusters._get_candidate_clusters(cluster_ratio_threshold=7.0)
        row = result[result["cluster_id"] == 0].iloc[0]
        assert row["Binder_chain"] == "A"
        assert row["Peptide_chain"] == "B"
        assert row["Binder_name"] == "ProtA"
        assert row["Peptide_name"] == "ProtB"

    def test_binder_assignment_y_longer(self, analyzer_with_geometry_clusters):
        """y_len > x_len → Binder=chain B, Peptide=chain A."""
        result = analyzer_with_geometry_clusters._get_candidate_clusters(cluster_ratio_threshold=7.0)
        row = result[result["cluster_id"] == 1].iloc[0]
        assert row["Binder_chain"] == "B"
        assert row["Peptide_chain"] == "A"
        assert row["Binder_name"] == "ProtB"
        assert row["Peptide_name"] == "ProtA"

    def test_low_ratio_excluded(self, analyzer_with_geometry_clusters):
        result = analyzer_with_geometry_clusters._get_candidate_clusters(cluster_ratio_threshold=7.0)
        assert "ProtA__ProtC" not in result["PPI"].values

    def test_min_peptide_len_filter(self, analyzer_with_geometry_clusters):
        """x_len < min_peptide_len excluded regardless of cluster_ratio."""
        result = analyzer_with_geometry_clusters._get_candidate_clusters(
            cluster_ratio_threshold=7.0, min_peptide_len=5
        )
        # Row 4 has x_len=3 < 5 → excluded
        assert len(result) == 2

    def test_empty_cluster_data_returns_empty(self):
        """No cluster data loaded → returns empty DataFrame."""
        result = InteractomeAnalyzer()._get_candidate_clusters()
        assert result.empty

    def test_legacy_cluster_ratio_column(self, dummy_interactome_csv, tmp_path):
        """Legacy 'Cluster_ratio' column name is handled."""
        cluster_csv = tmp_path / "legacy.csv"
        cluster_csv.write_text(
            "PPI,path,cluster_id,Cluster_ratio,x_len,y_len,x_min,x_max,y_min,y_max\n"
            "ProtA__ProtB,/fake/m.cif,0,10.0,50,5,0,50,100,105\n"
        )
        analyzer = InteractomeAnalyzer()
        analyzer.interactome_path = str(dummy_interactome_csv)
        analyzer.cluster_path = str(cluster_csv)
        result = analyzer._get_candidate_clusters(cluster_ratio_threshold=7.0)
        assert len(result) == 1


class TestCompareEnginesMissingColSilenced:
    def test_delta_col_keyerror_silenced(self, analyzer_with_data):
        """Lines 3335-3337: KeyError caught silently when other_df has no metric columns."""
        other_df = pd.DataFrame({"PPI": ["ProtA__ProtB", "ProtA__ProtC"]})
        result = analyzer_with_data.compare_engines(other_df)
        delta_cols = [c for c in result.columns if c.startswith("delta_")]
        assert len(delta_cols) == 0


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


# ---------------------------------------------------------------------------
# Foldseek helpers — unit tests with mocked HTTP
# ---------------------------------------------------------------------------

def _make_tsv_targz(rows: list[str]) -> bytes:
    """Build an in-memory tar.gz containing a single results.tsv."""
    tsv_content = "\n".join(rows).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="results.tsv")
        info.size = len(tsv_content)
        tar.addfile(info, io.BytesIO(tsv_content))
    return buf.getvalue()


_FAKE_TSV_ROWS = [
    "protA\ttarget1\t0.9\t100\t5\t2\t1\t100\t1\t100\t1e-50\t250",
    "protA\ttarget2\t0.7\t80\t10\t3\t1\t80\t5\t85\t1e-10\t120",
    "protA\ttarget3\t0.5\t60\t15\t4\t1\t60\t10\t70\t0.5\t50",  # above e-value cutoff
]


class TestSubmitFoldseekJob:
    def _client(self):
        with patch("requests.post"), patch("requests.get"):
            return FoldseekClient()

    def test_returns_ticket_id(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc123"}
        with patch.object(client, "_requests") as mock_req:
            mock_req.post.return_value = mock_resp
            ticket = client._submit("CIF_CONTENT", ["pdb100"])
        assert ticket == "abc123"

    def test_raises_on_http_error(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch.object(client, "_requests") as mock_req:
            mock_req.post.return_value = mock_resp
            with pytest.raises(RuntimeError, match="500"):
                client._submit("CIF", ["pdb100"])

    def test_sends_all_databases(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "x1"}
        with patch.object(client, "_requests") as mock_req:
            mock_req.post.return_value = mock_resp
            client._submit("CIF", ["afdb-swissprot", "pdb100"])
        payload = mock_req.post.call_args.kwargs["data"]
        assert payload["database[]"] == ["afdb-swissprot", "pdb100"]


class TestPollFoldseekJob:
    def _client(self):
        with patch("requests.post"), patch("requests.get"):
            return FoldseekClient()

    def test_returns_on_complete(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "COMPLETE"}
        with patch.object(client, "_requests") as mock_req, patch("time.sleep"):
            mock_req.get.return_value = mock_resp
            client._poll("abc123")  # should not raise

    def test_raises_on_error_status(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ERROR"}
        with patch.object(client, "_requests") as mock_req, patch("time.sleep"):
            mock_req.get.return_value = mock_resp
            with pytest.raises(RuntimeError, match="ERROR"):
                client._poll("abc123")

    def test_raises_on_timeout(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "RUNNING"}
        with patch.object(client, "_requests") as mock_req, patch("time.sleep"):
            mock_req.get.return_value = mock_resp
            with pytest.raises(TimeoutError):
                client._poll("abc123", poll_interval=1, timeout=2)


class TestDownloadFoldseekResults:
    def _client(self):
        with patch("requests.post"), patch("requests.get"):
            return FoldseekClient()

    def test_writes_tsv_file(self, tmp_path):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _make_tsv_targz(_FAKE_TSV_ROWS[:2])
        with patch.object(client, "_requests") as mock_req:
            mock_req.get.return_value = mock_resp
            tsv = client._download("abc123", tmp_path, "protA")
        assert tsv.exists()
        assert tsv.name == "protA.tsv"

    def test_raises_on_http_error(self, tmp_path):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not found"
        with patch.object(client, "_requests") as mock_req:
            mock_req.get.return_value = mock_resp
            with pytest.raises(RuntimeError, match="404"):
                client._download("bad_id", tmp_path, "protA")

    def test_plain_tsv_fallback(self, tmp_path):
        """If server returns plain TSV instead of tar.gz, it is still saved."""
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = "\n".join(_FAKE_TSV_ROWS).encode()
        with patch.object(client, "_requests") as mock_req:
            mock_req.get.return_value = mock_resp
            tsv = client._download("abc123", tmp_path, "protA")
        assert tsv.read_text()


# ---------------------------------------------------------------------------
# Network topology — compute_network_properties + plot_network
# ---------------------------------------------------------------------------

def _make_analyzer(tmp_path, ppis, weights, weight_col="Best_iLIS"):
    """Helper: build an InteractomeAnalyzer from explicit PPI/weight lists."""
    df = pd.DataFrame({
        "PPI": ppis,
        "Folder": ["/fake"] * len(ppis),
        weight_col: weights,
    })
    csv = tmp_path / "interactome_data.csv"
    df.to_csv(csv, index=False)
    a = InteractomeAnalyzer()
    a.interactome_path = str(csv)
    return a


class TestComputeNetworkProperties:

    _EXPECTED_COLS = {
        "protein", "degree", "weighted_degree",
        "betweenness_centrality", "closeness_centrality",
        "eigenvector_centrality", "clustering_coefficient",
        "is_hub", "is_bottleneck",
    }

    def test_no_data_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            InteractomeAnalyzer().compute_network_properties()

    def test_missing_weight_col_raises(self, analyzer_with_data):
        with pytest.raises(ValueError, match="not found"):
            analyzer_with_data.compute_network_properties(weight_col="nonexistent_col")

    def test_invalid_model_agg_raises(self, analyzer_with_data):
        with pytest.raises(ValueError, match="model_agg"):
            analyzer_with_data.compute_network_properties(model_agg="bad_agg")

    def test_expected_columns_present(self, analyzer_with_data):
        result = analyzer_with_data.compute_network_properties()
        assert self._EXPECTED_COLS.issubset(result.columns)

    def test_one_row_per_protein(self, analyzer_with_data):
        # fixture has 6 proteins: A B C D E F
        result = analyzer_with_data.compute_network_properties()
        assert len(result) == 6
        assert result["protein"].nunique() == 6

    def test_degree_matches_expected(self, tmp_path):
        # Star: A connected to B, C, D, E → A degree=4, leaves degree=1
        ppis = ["A__B", "A__C", "A__D", "A__E"]
        a = _make_analyzer(tmp_path, ppis, [0.5] * 4)
        result = a.compute_network_properties().set_index("protein")
        assert result.at["A", "degree"] == 4
        for leaf in ("B", "C", "D", "E"):
            assert result.at[leaf, "degree"] == 1

    def test_hub_detected_in_star(self, tmp_path):
        ppis = ["A__B", "A__C", "A__D", "A__E"]
        a = _make_analyzer(tmp_path, ppis, [0.5] * 4)
        result = a.compute_network_properties().set_index("protein")
        assert result.at["A", "is_hub"]
        for leaf in ("B", "C", "D", "E"):
            assert not result.at[leaf, "is_hub"]

    def test_bottleneck_detected_in_star(self, tmp_path):
        # In a star all shortest paths between leaves pass through centre
        ppis = ["A__B", "A__C", "A__D", "A__E"]
        a = _make_analyzer(tmp_path, ppis, [0.5] * 4)
        result = a.compute_network_properties().set_index("protein")
        assert result.at["A", "is_bottleneck"]
        for leaf in ("B", "C", "D", "E"):
            assert not result.at[leaf, "is_bottleneck"]

    def test_hub_bottleneck_are_bool(self, analyzer_with_data):
        result = analyzer_with_data.compute_network_properties()
        assert result["is_hub"].dtype == bool
        assert result["is_bottleneck"].dtype == bool

    def test_triangle_clustering_coefficient(self, tmp_path):
        # Complete triangle → every node's neighbours are also connected → cc=1
        ppis = ["A__B", "B__C", "A__C"]
        a = _make_analyzer(tmp_path, ppis, [0.5] * 3)
        result = a.compute_network_properties().set_index("protein")
        for node in ("A", "B", "C"):
            assert result.at[node, "clustering_coefficient"] == pytest.approx(1.0)

    def test_min_weight_excludes_weak_edges(self, tmp_path):
        ppis = ["A__B", "B__C"]
        a = _make_analyzer(tmp_path, ppis, [0.3, 0.3])
        result = a.compute_network_properties(min_weight=0.5)
        assert result.empty

    def test_min_weight_keeps_strong_edges(self, tmp_path):
        ppis = ["A__B", "B__C", "A__C"]
        a = _make_analyzer(tmp_path, ppis, [0.8, 0.2, 0.8])
        result = a.compute_network_properties(min_weight=0.5)
        # only A__B and A__C survive; B__C is excluded
        assert set(result["protein"]) == {"A", "B", "C"}
        assert result.set_index("protein").at["B", "degree"] == 1

    def test_model_agg_mean_averages_ranks(self, tmp_path):
        # Two rows for same PPI with different weights
        df = pd.DataFrame({
            "PPI": ["X__Y", "X__Y"],
            "Folder": ["/f", "/f"],
            "Best_iLIS": [0.2, 0.8],
        })
        csv = tmp_path / "interactome_data.csv"
        df.to_csv(csv, index=False)
        a = InteractomeAnalyzer()
        a.interactome_path = str(csv)
        result = a.compute_network_properties(model_agg="mean")
        assert result.set_index("protein").at["X", "weighted_degree"] == pytest.approx(0.5, abs=1e-3)

    def test_model_agg_best_picks_max_rank(self, tmp_path):
        df = pd.DataFrame({
            "PPI": ["X__Y", "X__Y"],
            "Folder": ["/f", "/f"],
            "Best_iLIS": [0.2, 0.8],
        })
        csv = tmp_path / "interactome_data.csv"
        df.to_csv(csv, index=False)
        a = InteractomeAnalyzer()
        a.interactome_path = str(csv)
        result = a.compute_network_properties(model_agg="best")
        assert result.set_index("protein").at["X", "weighted_degree"] == pytest.approx(0.8, abs=1e-3)

    def test_sorted_by_degree_descending(self, analyzer_with_data):
        result = analyzer_with_data.compute_network_properties()
        degrees = result["degree"].tolist()
        assert degrees == sorted(degrees, reverse=True)

    def test_custom_weight_col(self, tmp_path):
        df = pd.DataFrame({
            "PPI": ["A__B", "A__C"],
            "Folder": ["/f", "/f"],
            "ipSAE_AB": [0.6, 0.7],
        })
        csv = tmp_path / "interactome_data.csv"
        df.to_csv(csv, index=False)
        a = InteractomeAnalyzer()
        a.interactome_path = str(csv)
        result = a.compute_network_properties(weight_col="ipSAE_AB")
        assert len(result) == 3


class TestPlotNetwork:

    def test_no_data_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            InteractomeAnalyzer().plot_network()

    def test_saves_file(self, tmp_path):
        ppis = ["A__B", "A__C", "B__C"]
        a = _make_analyzer(tmp_path, ppis, [0.5, 0.6, 0.4])
        out = tmp_path / "network.png"
        a.plot_network(output_path=out)
        assert out.exists()

    def test_accepts_precomputed_network_df(self, tmp_path):
        ppis = ["A__B", "A__C", "B__C"]
        a = _make_analyzer(tmp_path, ppis, [0.5, 0.6, 0.4])
        net_df = a.compute_network_properties()
        out = tmp_path / "network2.png"
        a.plot_network(network_df=net_df, output_path=out)
        assert out.exists()

    def test_empty_graph_does_not_raise(self, tmp_path, caplog):
        import logging
        ppis = ["A__B"]
        a = _make_analyzer(tmp_path, ppis, [0.1])
        # min_weight above all weights → empty graph
        with caplog.at_level(logging.WARNING):
            a.plot_network(min_weight=0.9, output_path=tmp_path / "net.png")
        assert "no nodes" in caplog.text



# ---------------------------------------------------------------------------
# Helpers shared by database cross-validation tests
# ---------------------------------------------------------------------------

def _make_validated_analyzer(tmp_path, ppis, tiers, ipsae_vals=None):
    """InteractomeAnalyzer loaded from CSV with Tier column."""
    df = pd.DataFrame({
        "PPI": ppis,
        "Folder": ["/fake"] * len(ppis),
        "ipSAE_AB": ipsae_vals if ipsae_vals is not None else [0.5] * len(ppis),
        "Tier": tiers,
    })
    csv = tmp_path / "interactome_data.csv"
    df.to_csv(csv, index=False)
    a = InteractomeAnalyzer()
    a.interactome_path = str(csv)
    return a


def _make_db_client(tmp_path, pairs, name="db.csv"):
    """DatabaseClient from an in-memory pair list."""
    rows = "protein_A,protein_B\n" + "".join(f"{a},{b}\n" for a, b in pairs)
    p = tmp_path / name
    p.write_text(rows)
    return DatabaseClient.from_file(p)


# ---------------------------------------------------------------------------
# TestValidateAgainstDatabase
# ---------------------------------------------------------------------------

class TestValidateAgainstDatabase:

    def test_no_data_raises(self, tmp_path):
        client = _make_db_client(tmp_path, [("A", "B")])
        with pytest.raises(RuntimeError, match="not loaded"):
            InteractomeAnalyzer().validate_against_database(client)

    def test_supported_true_for_known_pair(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["A__B"], ["Tier 1"])
        client = _make_db_client(tmp_path, [("A", "B")])
        result = a.validate_against_database(client)
        assert result.loc[result["PPI"] == "A__B", "experimental_support"].item() is True

    def test_unconfirmed_false_when_both_proteins_known(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["A__B", "A__C"], ["Tier 1", "Tier 1"])
        # DB knows A, B, C (via A-B and B-C) but does not record A-C → False
        client = _make_db_client(tmp_path, [("A", "B"), ("B", "C")])
        result = a.validate_against_database(client)
        assert result.loc[result["PPI"] == "A__C", "experimental_support"].item() is False

    def test_nan_when_protein_absent_from_db(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["A__X"], ["Tier 2"])
        client = _make_db_client(tmp_path, [("A", "B")])  # X not in DB
        result = a.validate_against_database(client)
        assert pd.isna(result.loc[result["PPI"] == "A__X", "experimental_support"].item())

    def test_order_independent_lookup(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["B__A"], ["Tier 1"])
        client = _make_db_client(tmp_path, [("A", "B")])
        result = a.validate_against_database(client)
        assert result.loc[result["PPI"] == "B__A", "experimental_support"].item() is True

    def test_id_map_applied(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["orf1__orf2"], ["Tier 1"])
        client = _make_db_client(tmp_path, [("geneA", "geneB")])
        result = a.validate_against_database(
            client, id_map={"orf1": "geneA", "orf2": "geneB"}
        )
        assert result["experimental_support"].item() is True

    def test_id_map_unmapped_gives_nan(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["orf1__orf3"], ["Tier 1"])
        client = _make_db_client(tmp_path, [("geneA", "geneB")])
        result = a.validate_against_database(
            client, id_map={"orf1": "geneA"}  # orf3 not in map
        )
        assert pd.isna(result["experimental_support"].item())

    def test_one_row_per_ppi(self, tmp_path):
        # Two model rows for same PPI → aggregated to one
        df = pd.DataFrame({
            "PPI": ["A__B", "A__B"],
            "Folder": ["/f", "/f"],
            "ipSAE_AB": [0.6, 0.8],
            "Tier": ["Tier 1", "Tier 1"],
        })
        csv = tmp_path / "interactome_data.csv"
        df.to_csv(csv, index=False)
        a = InteractomeAnalyzer()
        a.interactome_path = str(csv)
        client = _make_db_client(tmp_path, [("A", "B")])
        result = a.validate_against_database(client)
        assert len(result) == 1

    def test_tier_column_preserved(self, tmp_path):
        a = _make_validated_analyzer(tmp_path, ["A__B"], ["Tier 1"])
        client = _make_db_client(tmp_path, [("A", "B")])
        result = a.validate_against_database(client)
        assert "Tier" in result.columns


# ---------------------------------------------------------------------------
# TestValidationSummary
# ---------------------------------------------------------------------------

class TestValidationSummary:

    @pytest.fixture
    def validated_df(self):
        return pd.DataFrame({
            "PPI":                  ["A__B", "A__C", "B__C", "D__E"],
            "Tier":                 ["Tier 1", "Tier 1", "Tier 2", "Tier 2"],
            "experimental_support": [True,    False,    True,     None],
        })

    def test_missing_experimental_support_raises(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        bad = validated_df.drop(columns=["experimental_support"])
        with pytest.raises(ValueError, match="experimental_support"):
            a.validation_summary(bad)

    def test_missing_group_by_raises(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        with pytest.raises(ValueError, match="nonexistent"):
            a.validation_summary(validated_df, group_by="nonexistent")

    def test_overall_row_always_present(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        result = a.validation_summary(validated_df)
        assert "Overall" in result["Tier"].values

    def test_precision_per_group(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        result = a.validation_summary(validated_df)
        tier1 = result[result["Tier"] == "Tier 1"].iloc[0]
        assert tier1["n_supported"] == 1
        assert tier1["n_unconfirmed"] == 1
        assert tier1["precision"] == pytest.approx(0.5)

    def test_overall_precision(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        result = a.validation_summary(validated_df)
        overall = result[result["Tier"] == "Overall"].iloc[0]
        # assessable: A__B(T), A__C(F), B__C(T) → 3; supported: 2
        assert overall["n_assessable"] == 3
        assert overall["n_supported"] == 2
        assert overall["precision"] == pytest.approx(2 / 3)

    def test_no_recall_col_without_known_ppis(self, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        result = a.validation_summary(validated_df)
        assert "recall" not in result.columns
        assert "f1" not in result.columns

    def test_recall_added_with_known_ppis(self, tmp_path, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        client = _make_db_client(tmp_path, [("A", "B"), ("A", "C"), ("B", "C")])
        result = a.validation_summary(validated_df, known_ppis=client)
        assert "recall" in result.columns
        assert "f1" in result.columns

    def test_n_reachable_correct(self, tmp_path, validated_df):
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = validated_df
        # Proteins in interactome: A,B,C,D,E. DB has A-B and A-C → both reachable.
        client = _make_db_client(tmp_path, [("A", "B"), ("A", "C")])
        result = a.validation_summary(validated_df, known_ppis=client)
        assert result["n_reachable_experimental"].iloc[0] == 2

    def test_recall_zero_when_no_supported(self, tmp_path):
        df = pd.DataFrame({
            "PPI":  ["A__B"],
            "Tier": ["Tier 1"],
            "experimental_support": [False],
        })
        a = InteractomeAnalyzer.__new__(InteractomeAnalyzer)
        a._interactome_data = df
        client = _make_db_client(tmp_path, [("A", "B")])
        result = a.validation_summary(df, known_ppis=client)
        overall = result[result["Tier"] == "Overall"].iloc[0]
        assert overall["recall"] == pytest.approx(0.0)
        assert pd.isna(overall["f1"])


# ---------------------------------------------------------------------------
# FoldseekClient — init validation + search orchestration
# ---------------------------------------------------------------------------

class TestFoldseekClientInit:
    def test_negative_plddt_raises(self):
        with pytest.raises(ValueError, match="pLDDT"):
            FoldseekClient(plddt_threshold=-1)

    def test_over_100_plddt_raises(self):
        with pytest.raises(ValueError, match="pLDDT"):
            FoldseekClient(plddt_threshold=101)

    def test_valid_plddt_at_boundary(self):
        client = FoldseekClient(plddt_threshold=100)
        assert client.plddt_threshold == 100

    def test_import_requests_raises_when_missing(self):
        import sys
        with patch.dict(sys.modules, {"requests": None}):
            with pytest.raises(ImportError, match="requests"):
                FoldseekClient._import_requests()


class TestFoldseekSearch:
    def _client(self):
        return FoldseekClient()

    def test_missing_cif_raises(self, tmp_path):
        client = self._client()
        with pytest.raises(FileNotFoundError):
            client.search(tmp_path / "nonexistent.cif", ["pdb100"])

    def test_existing_tsv_reused_without_submit(self, tmp_path):
        cif = tmp_path / "prot.cif"
        cif.write_text("mock")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        existing_tsv = out_dir / "result.tsv"
        existing_tsv.write_text("col1\tcol2\nA\tB")

        client = self._client()
        with patch.object(client, "_submit") as mock_sub:
            result = client.search(cif, ["pdb100"], out_dir=out_dir)
        mock_sub.assert_not_called()
        assert result == existing_tsv

    def test_full_pipeline_orchestration(self, tmp_path):
        cif = tmp_path / "myprotein.cif"
        cif.write_text("CIF content here")
        out_dir = tmp_path / "out"
        fake_tsv = out_dir / "myprotein.tsv"

        client = self._client()
        with patch.object(client, "_submit", return_value="ticket42") as mock_sub, \
             patch.object(client, "_poll") as mock_poll, \
             patch.object(client, "_download", return_value=fake_tsv) as mock_dl:
            result = client.search(cif, ["pdb100"], out_dir=out_dir)

        mock_sub.assert_called_once()
        mock_poll.assert_called_once_with("ticket42")
        mock_dl.assert_called_once_with("ticket42", out_dir, "myprotein")
        assert result == fake_tsv

    def test_submit_none_warns_and_returns_none(self, tmp_path):
        cif = tmp_path / "prot.cif"
        cif.write_text("content")
        client = self._client()
        with patch.object(client, "_submit", return_value=None), \
             pytest.warns(UserWarning):
            result = client.search(cif, ["pdb100"], out_dir=tmp_path / "out")
        assert result is None

    def test_protein_id_defaults_to_stem(self, tmp_path):
        cif = tmp_path / "myprot.cif"
        cif.write_text("content")
        out_dir = tmp_path / "out"
        fake_tsv = out_dir / "myprot.tsv"

        client = self._client()
        with patch.object(client, "_submit", return_value="t1"), \
             patch.object(client, "_poll"), \
             patch.object(client, "_download", return_value=fake_tsv) as mock_dl:
            client.search(cif, ["pdb100"], out_dir=out_dir)
        assert mock_dl.call_args[0][2] == "myprot"

    def test_out_dir_created_if_missing(self, tmp_path):
        cif = tmp_path / "prot.cif"
        cif.write_text("content")
        out_dir = tmp_path / "nested" / "deep" / "out"

        client = self._client()
        with patch.object(client, "_submit", return_value="t1"), \
             patch.object(client, "_poll"), \
             patch.object(client, "_download", return_value=out_dir / "prot.tsv"):
            client.search(cif, ["pdb100"], out_dir=out_dir)
        assert out_dir.exists()


class TestSubmitMissingId:
    def _client(self):
        return FoldseekClient()

    def test_warns_and_returns_none_when_id_missing(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "PENDING"}
        with patch.object(client, "_requests") as mock_req, \
             pytest.warns(UserWarning):
            mock_req.post.return_value = mock_resp
            result = client._submit("CIF", ["pdb100"])
        assert result is None
