import pytest
import numpy as np
import pandas as pd
from virus_interactome.interactome import InteractomeProcessor


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestProcessorInit:
    def test_valid_af3(self):
        proc = InteractomeProcessor(["/fake/model.cif"], engine="af3")
        assert proc.engine == "af3"
        assert len(proc.model_paths) == 1

    def test_valid_boltz(self):
        proc = InteractomeProcessor(["/fake/model.cif"], engine="boltz")
        assert proc.engine == "boltz"

    def test_valid_colabfold(self):
        proc = InteractomeProcessor(["/fake/model.cif"], engine="colabfold")
        assert proc.engine == "colabfold"

    def test_case_insensitive(self):
        proc = InteractomeProcessor([], engine="AF3")
        assert proc.engine == "af3"

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Engine should be one of"):
            InteractomeProcessor([], engine="rosettafold")

    def test_empty_model_list(self):
        proc = InteractomeProcessor([], engine="af3")
        assert proc.model_paths == []
        assert proc.df_het is None
        assert proc.df_hom is None
        assert proc.cluster_data is None


# ---------------------------------------------------------------------------
# cluster_pae — static method
# ---------------------------------------------------------------------------

class TestClusterPAE:
    def test_basic_cluster(self):
        """Dense low-PAE block should produce at least one cluster."""
        pae = np.full((50, 50), 30.0)
        # Insert a 20x20 low-PAE block
        pae[5:25, 10:30] = 3.0
        coords, labels = InteractomeProcessor.cluster_pae(pae, threshold=15.0, eps=3.0, min_samples=5)
        assert coords.shape[0] > 0
        assert coords.shape[1] == 2
        # At least one real cluster (label >= 0)
        assert np.any(labels >= 0)

    def test_no_contacts(self):
        """All-high PAE matrix → no contacts, empty arrays."""
        pae = np.full((30, 30), 50.0)
        coords, labels = InteractomeProcessor.cluster_pae(pae, threshold=15.0)
        assert coords.shape == (0, 2)
        assert labels.shape == (0,)

    def test_all_below_threshold(self):
        """All-low PAE → everything is a contact, should cluster."""
        pae = np.full((20, 20), 2.0)
        coords, labels = InteractomeProcessor.cluster_pae(pae, threshold=15.0, eps=5.0, min_samples=3)
        assert coords.shape[0] == 20 * 20
        assert np.any(labels >= 0)

    def test_custom_threshold(self):
        pae = np.full((20, 20), 10.0)
        # With threshold=5, nothing passes
        coords_strict, labels_strict = InteractomeProcessor.cluster_pae(pae, threshold=5.0)
        assert coords_strict.shape[0] == 0
        # With threshold=15, everything passes
        coords_lax, labels_lax = InteractomeProcessor.cluster_pae(pae, threshold=15.0, eps=3.0, min_samples=3)
        assert coords_lax.shape[0] == 20 * 20

    def test_sparse_contacts_noise(self):
        """Scattered low-PAE points with strict DBSCAN → all noise (label -1)."""
        pae = np.full((100, 100), 30.0)
        # Place 5 isolated low-PAE points far apart
        for i in range(0, 100, 25):
            pae[i, i] = 3.0
        coords, labels = InteractomeProcessor.cluster_pae(pae, threshold=15.0, eps=2.0, min_samples=5)
        # Points exist but all should be noise
        assert coords.shape[0] > 0
        assert np.all(labels == -1)


# ---------------------------------------------------------------------------
# cluster_info — static method
# ---------------------------------------------------------------------------

class TestClusterInfo:
    def test_basic_structure(self):
        """Check output DataFrame has all expected columns."""
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = InteractomeProcessor.cluster_info(coords, labels)
        expected_cols = {
            "cluster_id", "num_points", "x_len", "y_len",
            "x_min", "x_max", "y_min", "y_max",
            "center_x", "center_y", "cluster_ratio",
        }
        assert expected_cols == set(result.columns)
        assert len(result) == 2

    def test_noise_ignored(self):
        """Points labeled -1 (noise) should not appear in output."""
        coords = np.array([[0, 0], [0, 1], [50, 50]])
        labels = np.array([0, 0, -1])
        result = InteractomeProcessor.cluster_info(coords, labels)
        assert len(result) == 1
        assert result.iloc[0]["cluster_id"] == 0
        assert result.iloc[0]["num_points"] == 2

    def test_empty_input(self):
        """Empty coords + labels → empty DataFrame with correct columns."""
        coords = np.empty((0, 2), dtype=int)
        labels = np.array([], dtype=int)
        result = InteractomeProcessor.cluster_info(coords, labels)
        assert len(result) == 0
        assert "cluster_id" in result.columns

    def test_bounding_box_values(self):
        """Verify bounding box geometry for a known cluster."""
        coords = np.array([[2, 5], [2, 10], [8, 5], [8, 10]])
        labels = np.array([0, 0, 0, 0])
        result = InteractomeProcessor.cluster_info(coords, labels)
        row = result.iloc[0]
        # Rows → Y, Cols → X
        assert row["y_min"] == 2
        assert row["y_max"] == 8
        assert row["x_min"] == 5
        assert row["x_max"] == 10
        assert row["y_len"] == 6
        assert row["x_len"] == 5

    def test_aspect_ratio_square(self):
        """Square cluster should have aspect_ratio = 1.0."""
        coords = np.array([[0, 0], [0, 10], [10, 0], [10, 10]])
        labels = np.array([0, 0, 0, 0])
        result = InteractomeProcessor.cluster_info(coords, labels)
        assert result.iloc[0]["cluster_ratio"] == pytest.approx(1.0)

    def test_aspect_ratio_elongated(self):
        """Elongated cluster should have high aspect_ratio."""
        coords = np.array([[0, 0], [0, 1], [100, 0], [100, 1]])
        labels = np.array([0, 0, 0, 0])
        result = InteractomeProcessor.cluster_info(coords, labels)
        assert result.iloc[0]["cluster_ratio"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Integration: cluster_pae → cluster_info pipeline
# ---------------------------------------------------------------------------

class TestClusterPipeline:
    def test_pae_to_info_pipeline(self):
        """Full pipeline: synthetic PAE → cluster_pae → cluster_info."""
        pae = np.full((60, 40), 30.0)
        # Block 1: rows 0-19, cols 0-9 (compact)
        pae[0:20, 0:10] = 2.0
        # Block 2: rows 30-55, cols 5-8 (elongated)
        pae[30:56, 5:9] = 2.0

        coords, labels = InteractomeProcessor.cluster_pae(pae, threshold=15.0, eps=3.0, min_samples=5)
        info = InteractomeProcessor.cluster_info(coords, labels)

        # Should find 2 clusters
        assert len(info) == 2
        # Both should have reasonable num_points
        assert all(info["num_points"] > 10)
        # The elongated block should have higher aspect_ratio
        ratios = info.sort_values("cluster_ratio", ascending=False)
        assert ratios.iloc[0]["cluster_ratio"] > ratios.iloc[1]["cluster_ratio"]
