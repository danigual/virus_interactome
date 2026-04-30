import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
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


# ---------------------------------------------------------------------------
# process_ppi — full static method test with real AF3 dummy data
# ---------------------------------------------------------------------------

import shutil

@pytest.fixture
def af3_model_in_ppi_dir(tmp_path, data_dir):
    """Copies AF3 dummy CIF + JSON files into a PPI-named directory for process_ppi."""
    ppi_dir = tmp_path / "ProtA__ProtB"
    ppi_dir.mkdir()
    src = data_dir / "af3_dummy_example"
    # Copy explicitly — iterdir() order can be unreliable
    for name in [
        "fold_adv5_pvi_protease_model_0.cif",
        "fold_adv5_pvi_protease_full_data_0.json",
        "fold_adv5_pvi_protease_summary_confidences_0.json",
    ]:
        shutil.copy2(src / name, ppi_dir / name)
    cif = ppi_dir / "fold_adv5_pvi_protease_model_0.cif"
    assert cif.exists(), f"CIF not found: {cif}"
    assert (ppi_dir / "fold_adv5_pvi_protease_full_data_0.json").exists()
    return cif


@pytest.mark.slow
class TestProcessPpi:
    def test_returns_tuple(self, af3_model_in_ppi_dir):
        result = InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_summary_dict_keys(self, af3_model_in_ppi_dir):
        summary, _ = InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        required = {"PPI", "ORF_A", "ORF_B", "Folder", "Model_num", "ipTM", "pTM"}
        assert required.issubset(set(summary.keys()))

    def test_ppi_parsed_from_dir_name(self, af3_model_in_ppi_dir):
        summary, _ = InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        assert summary["PPI"] == "ProtA__ProtB"
        assert summary["ORF_A"] == "ProtA"
        assert summary["ORF_B"] == "ProtB"

    def test_extracts_idx_zero(self, af3_model_in_ppi_dir):
        # NOTE: test name must NOT contain '_model_' — process_full_data_af3 uses
        # str.replace("_model_", ...) on the full path, which would corrupt
        # pytest's tmp_path if it contains that substring.
        summary, _ = InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        assert summary["Model_num"] == 0

    def test_cluster_data_non_empty_for_heteromer(self, af3_model_in_ppi_dir):
        _, clusters = InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        # Heteromer (2 chains) → should produce cluster data
        assert not clusters.empty
        assert "cluster_id" in clusters.columns
        assert "PPI" in clusters.columns
        assert "cluster_ratio" in clusters.columns

    def test_plots_created(self, af3_model_in_ppi_dir):
        InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="AF3")
        parent = af3_model_in_ppi_dir.parent
        stem = af3_model_in_ppi_dir.stem
        assert (parent / f"{stem}_plddt.png").exists()
        assert (parent / f"{stem}_pae.png").exists()
        assert (parent / f"{stem}_cluster.png").exists()

    def test_invalid_model_type_raises(self, af3_model_in_ppi_dir):
        with pytest.raises(ValueError, match="not supported"):
            InteractomeProcessor.process_ppi(str(af3_model_in_ppi_dir), model_type="InvalidEngine")

    def test_prefix_stripping(self, af3_model_in_ppi_dir):
        summary, _ = InteractomeProcessor.process_ppi(
            str(af3_model_in_ppi_dir), model_type="AF3", prefix="Prot"
        )
        # prefix="Prot" replaces "Prot" in dir name "ProtA__ProtB" → "A__B"
        assert summary["PPI"] == "A__B"


# ---------------------------------------------------------------------------
# process_ppi — Boltz engine (L1726)
# ---------------------------------------------------------------------------

@pytest.fixture
def boltz_model_in_ppi_dir(tmp_path, data_dir):
    """Copies Boltz2 dummy files into a PPI-named directory for process_ppi."""
    ppi_dir = tmp_path / "pvi__protease"
    ppi_dir.mkdir()
    src = data_dir / "boltz_dummy_example"
    for name in [
        "pvi__protease_model_0.cif",
        "confidence_pvi__protease_model_0.json",
        "pae_pvi__protease_model_0.npz",
        "plddt_pvi__protease_model_0.npz",
        "pde_pvi__protease_model_0.npz",
    ]:
        shutil.copy2(src / name, ppi_dir / name)
    return ppi_dir / "pvi__protease_model_0.cif"


@pytest.mark.slow
@pytest.mark.skip(reason="_load_boltz_metrics not yet implemented in Model")
class TestProcessPpiBoltz:
    def test_boltz_returns_tuple(self, boltz_model_in_ppi_dir):
        """L1726: process_ppi dispatches to process_full_data_boltz."""
        result = InteractomeProcessor.process_ppi(str(boltz_model_in_ppi_dir), model_type="boltz")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_boltz_summary_keys(self, boltz_model_in_ppi_dir):
        summary, _ = InteractomeProcessor.process_ppi(str(boltz_model_in_ppi_dir), model_type="boltz")
        assert {"PPI", "ORF_A", "ORF_B", "Model_num", "ipTM", "pTM"}.issubset(summary.keys())

    def test_boltz_ppi_parsed_from_dir_name(self, boltz_model_in_ppi_dir):
        summary, _ = InteractomeProcessor.process_ppi(str(boltz_model_in_ppi_dir), model_type="boltz")
        assert summary["PPI"] == "pvi__protease"
        assert summary["ORF_A"] == "pvi"
        assert summary["ORF_B"] == "protease"


# ---------------------------------------------------------------------------
# process_models — orchestration (L1836-1932)
# ---------------------------------------------------------------------------

class _SyncPool:
    """Drop-in replacement for ProcessPoolExecutor that runs synchronously."""
    def __init__(self, **kw):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def map(self, fn, iterable):
        return list(map(fn, iterable))


@pytest.mark.slow
class TestProcessModels:
    def test_creates_output_csvs(self, af3_model_in_ppi_dir, tmp_path):
        """L1836-1932: process_models produces interactome_data.csv and clusters_data.csv."""
        from unittest.mock import patch
        proc = InteractomeProcessor([str(af3_model_in_ppi_dir)], engine="af3")
        out_dir = tmp_path / "output"
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            proc.process_models(str(out_dir))
        assert (out_dir / "interactome_data.csv").exists()
        df = pd.read_csv(out_dir / "interactome_data.csv")
        assert len(df) == 1
        assert "PPI" in df.columns

    def test_resume_skips_already_processed(self, af3_model_in_ppi_dir, tmp_path):
        """L1850-1869: second call skips models already in existing CSV."""
        from unittest.mock import patch
        proc = InteractomeProcessor([str(af3_model_in_ppi_dir)], engine="af3")
        out_dir = tmp_path / "output"
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            proc.process_models(str(out_dir))
        # Second call: all models already processed → early return, no re-processing
        call_count = {"n": 0}
        original_ppi = InteractomeProcessor.process_ppi
        def counting_ppi(*args, **kwargs):
            call_count["n"] += 1
            return original_ppi(*args, **kwargs)
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            with patch.object(InteractomeProcessor, "process_ppi", staticmethod(counting_ppi)):
                proc.process_models(str(out_dir))
        assert call_count["n"] == 0  # no model was reprocessed


# ---------------------------------------------------------------------------
# _extract_monomer_plddt — static method
# ---------------------------------------------------------------------------

class TestExtractMonomerpLDDT:
    """Tests for InteractomeProcessor._extract_monomer_plddt."""

    def _mock_model(self, plddts: np.ndarray):
        mock = MagicMock()
        mock._metrics.ca_plddts = plddts
        return mock

    def test_returns_correct_keys(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome.Model", return_value=self._mock_model(np.full(10, 80.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "af3")
        assert set(result.keys()) == {"plddt_mean", "plddt_median", "n_residues"}

    def test_correct_plddt_values(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        plddts = np.array([70.0, 80.0, 90.0, 85.0, 75.0])
        with patch("virus_interactome.interactome.Model", return_value=self._mock_model(plddts)):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "af3")
        assert result["plddt_mean"] == pytest.approx(np.mean(plddts))
        assert result["plddt_median"] == pytest.approx(np.median(plddts))
        assert result["n_residues"] == 5

    def test_boltz_engine(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome.Model", return_value=self._mock_model(np.full(20, 75.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "boltz")
        assert result["n_residues"] == 20

    def test_boltz2_alias(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome.Model", return_value=self._mock_model(np.full(15, 85.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "boltz2")
        assert result["n_residues"] == 15

    def test_colabfold_engine(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome.Model", return_value=self._mock_model(np.full(8, 90.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "colabfold")
        assert result["plddt_mean"] == pytest.approx(90.0)

    def test_parse_failure_returns_nan(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome.Model", side_effect=RuntimeError("parse error")):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "af3")
        assert np.isnan(result["plddt_mean"])
        assert np.isnan(result["plddt_median"])
        assert np.isnan(result["n_residues"])

    def test_unsupported_engine_returns_nan(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        result = InteractomeProcessor._extract_monomer_plddt(cif, "rosettafold")
        assert np.isnan(result["plddt_mean"])


# ---------------------------------------------------------------------------
# process_monomers — instance method
# ---------------------------------------------------------------------------

class TestProcessMonomers:
    """Tests for InteractomeProcessor.process_monomers."""

    def _make_cif_dir(self, tmp_path: Path, proteins: list) -> tuple:
        """Create monomer folder structure and return (model_paths, cif_root)."""
        cif_paths = []
        for pid in proteins:
            d = tmp_path / pid
            d.mkdir(parents=True)
            cif = d / f"{pid}_model_0.cif"
            cif.write_text("")
            cif_paths.append(cif)
        return cif_paths, tmp_path

    def _fake_result(self, n: int = 10, val: float = 80.0) -> dict:
        return {"plddt_mean": val, "plddt_median": val, "n_residues": n}

    def _mock_extract(self, n: int = 10, val: float = 80.0):
        result = self._fake_result(n, val)
        return staticmethod(lambda *a, **k: result)

    def test_creates_monomer_csv(self, tmp_path):
        cif_paths, _ = self._make_cif_dir(tmp_path / "models", ["protA", "protB"])
        proc = InteractomeProcessor([str(p) for p in cif_paths], engine="af3")
        out_dir = tmp_path / "out"
        with patch.object(InteractomeProcessor, "_extract_monomer_plddt", self._mock_extract()), \
             patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            df = proc.process_monomers(str(out_dir))
        assert (out_dir / "monomer_data.csv").exists()
        assert len(df) == 2

    def test_output_columns(self, tmp_path):
        cif_paths, _ = self._make_cif_dir(tmp_path / "models", ["protA"])
        proc = InteractomeProcessor([str(p) for p in cif_paths], engine="af3")
        out_dir = tmp_path / "out"
        with patch.object(InteractomeProcessor, "_extract_monomer_plddt", self._mock_extract(12, 85.0)), \
             patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            df = proc.process_monomers(str(out_dir))
        assert set(df.columns) >= {"protein_id", "cif_path", "n_residues", "plddt_mean", "plddt_median"}

    def test_protein_id_from_folder_name(self, tmp_path):
        cif_paths, _ = self._make_cif_dir(tmp_path / "models", ["hexon"])
        proc = InteractomeProcessor([str(p) for p in cif_paths], engine="af3")
        out_dir = tmp_path / "out"
        with patch.object(InteractomeProcessor, "_extract_monomer_plddt", self._mock_extract()), \
             patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            df = proc.process_monomers(str(out_dir))
        assert df.iloc[0]["protein_id"] == "hexon"

    def test_resume_skips_existing(self, tmp_path):
        cif_paths, _ = self._make_cif_dir(tmp_path / "models", ["protA", "protB"])
        proc = InteractomeProcessor([str(p) for p in cif_paths], engine="af3")
        out_dir = tmp_path / "out"
        with patch.object(InteractomeProcessor, "_extract_monomer_plddt", self._mock_extract()), \
             patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            proc.process_monomers(str(out_dir))
        call_count = {"n": 0}
        def counting(*args, **kwargs):
            call_count["n"] += 1
            return self._fake_result()
        with patch.object(InteractomeProcessor, "_extract_monomer_plddt", staticmethod(counting)), \
             patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            proc.process_monomers(str(out_dir))
        assert call_count["n"] == 0

    def test_empty_model_list_returns_empty_df(self, tmp_path):
        proc = InteractomeProcessor([], engine="af3")
        out_dir = tmp_path / "out"
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            df = proc.process_monomers(str(out_dir))
        assert df.empty
