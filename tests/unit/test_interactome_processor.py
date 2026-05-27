import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from virus_interactome.interactome_processor import InteractomeProcessor


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

    def test_engine_enum_input(self):
        from virus_interactome.model import Engine
        proc = InteractomeProcessor([], engine=Engine.AF3)
        assert proc.engine == "af3"

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


@pytest.fixture
def af3_proc(af3_model_in_ppi_dir):
    return InteractomeProcessor([str(af3_model_in_ppi_dir)], engine="af3")


@pytest.mark.slow
class TestProcessPpi:
    def test_returns_tuple(self, af3_model_in_ppi_dir, af3_proc):
        result = af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_summary_dict_keys(self, af3_model_in_ppi_dir, af3_proc):
        summary, _ = af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        required = {"PPI", "ORF_A", "ORF_B", "Folder", "Model_num", "ipTM", "pTM"}
        assert required.issubset(set(summary.keys()))

    def test_ppi_parsed_from_dir_name(self, af3_model_in_ppi_dir, af3_proc):
        summary, _ = af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        assert summary["PPI"] == "ProtA__ProtB"
        assert summary["ORF_A"] == "ProtA"
        assert summary["ORF_B"] == "ProtB"

    def test_extracts_idx_zero(self, af3_model_in_ppi_dir, af3_proc):
        summary, _ = af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        assert summary["Model_num"] == 0

    def test_cluster_data_non_empty_for_heteromer(self, af3_model_in_ppi_dir, af3_proc):
        _, clusters = af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        assert not clusters.empty
        assert "cluster_id" in clusters.columns
        assert "PPI" in clusters.columns
        assert "cluster_ratio" in clusters.columns

    def test_plots_created(self, af3_model_in_ppi_dir, af3_proc):
        af3_proc.process_ppi(str(af3_model_in_ppi_dir))
        parent = af3_model_in_ppi_dir.parent
        stem = af3_model_in_ppi_dir.stem
        assert (parent / f"{stem}_plddt.png").exists()
        assert (parent / f"{stem}_pae.png").exists()
        assert (parent / f"{stem}_cluster.png").exists()

    def test_invalid_engine_raises(self, af3_model_in_ppi_dir):
        with pytest.raises(ValueError, match="Engine should be one of"):
            InteractomeProcessor([str(af3_model_in_ppi_dir)], engine="InvalidEngine")

    def test_prefix_stripping(self, af3_model_in_ppi_dir, af3_proc):
        summary, _ = af3_proc.process_ppi(str(af3_model_in_ppi_dir), prefix="Prot")
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


@pytest.fixture
def boltz_proc(boltz_model_in_ppi_dir):
    return InteractomeProcessor([str(boltz_model_in_ppi_dir)], engine="boltz")


@pytest.mark.slow
class TestProcessPpiBoltz:
    def test_boltz_returns_tuple(self, boltz_model_in_ppi_dir, boltz_proc):
        result = boltz_proc.process_ppi(str(boltz_model_in_ppi_dir))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_boltz_summary_keys(self, boltz_model_in_ppi_dir, boltz_proc):
        summary, _ = boltz_proc.process_ppi(str(boltz_model_in_ppi_dir))
        assert {"PPI", "ORF_A", "ORF_B", "Model_num", "ipTM", "pTM"}.issubset(summary.keys())

    def test_boltz_ppi_parsed_from_dir_name(self, boltz_model_in_ppi_dir, boltz_proc):
        summary, _ = boltz_proc.process_ppi(str(boltz_model_in_ppi_dir))
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
        """Second call skips models already in existing CSV."""
        from unittest.mock import patch
        proc = InteractomeProcessor([str(af3_model_in_ppi_dir)], engine="af3")
        out_dir = tmp_path / "output"
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            proc.process_models(str(out_dir))
        # Second call: all models already processed → early return, no re-processing
        call_count = {"n": 0}
        original_ppi = InteractomeProcessor.process_ppi
        def counting_ppi(self_inst, *args, **kwargs):
            call_count["n"] += 1
            return original_ppi(self_inst, *args, **kwargs)
        with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool):
            with patch.object(InteractomeProcessor, "process_ppi", counting_ppi):
                proc.process_models(str(out_dir))
        assert call_count["n"] == 0


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
        with patch("virus_interactome.interactome_processor.Model", return_value=self._mock_model(np.full(10, 80.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "af3")
        assert set(result.keys()) == {"plddt_mean", "plddt_median", "n_residues"}

    def test_correct_plddt_values(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        plddts = np.array([70.0, 80.0, 90.0, 85.0, 75.0])
        with patch("virus_interactome.interactome_processor.Model", return_value=self._mock_model(plddts)):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "af3")
        assert result["plddt_mean"] == pytest.approx(np.mean(plddts))
        assert result["plddt_median"] == pytest.approx(np.median(plddts))
        assert result["n_residues"] == 5

    def test_boltz_engine(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome_processor.Model", return_value=self._mock_model(np.full(20, 75.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "boltz")
        assert result["n_residues"] == 20

    def test_boltz2_alias(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome_processor.Model", return_value=self._mock_model(np.full(15, 85.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "boltz2")
        assert result["n_residues"] == 15

    def test_colabfold_engine(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome_processor.Model", return_value=self._mock_model(np.full(8, 90.0))):
            result = InteractomeProcessor._extract_monomer_plddt(cif, "colabfold")
        assert result["plddt_mean"] == pytest.approx(90.0)

    def test_parse_failure_returns_nan(self, tmp_path):
        cif = tmp_path / "mono_model_0.cif"
        cif.write_text("")
        with patch("virus_interactome.interactome_processor.Model", side_effect=RuntimeError("parse error")):
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


# ── InteractomeProcessor._size_correct_iptm ───────────────────────────────────

class TestSizeCorrectIptm:

    def test_known_value(self):
        import numpy as np
        from virus_interactome.interactome_processor import InteractomeProcessor
        iptm = 0.5
        len_a, len_b = 100, 200
        expected_correction = -0.036255571 + 0.004470512 * np.sqrt(300)
        result = InteractomeProcessor._size_correct_iptm(iptm, len_a, len_b)
        assert abs(result - (iptm - expected_correction)) < 1e-9

    def test_larger_proteins_get_more_correction(self):
        from virus_interactome.interactome_processor import InteractomeProcessor
        sc_small = InteractomeProcessor._size_correct_iptm(0.5, 50, 50)
        sc_large = InteractomeProcessor._size_correct_iptm(0.5, 500, 500)
        # Larger proteins → larger expected_iptm → smaller corrected value
        assert sc_large < sc_small

    def test_returns_float(self):
        from virus_interactome.interactome_processor import InteractomeProcessor
        result = InteractomeProcessor._size_correct_iptm(0.3, 100, 100)
        assert isinstance(result, float)

    def test_high_iptm_stays_positive_for_typical_proteins(self):
        from virus_interactome.interactome_processor import InteractomeProcessor
        # For typical viral proteins (~200-500 aa), a confident ipTM of 0.8
        # should remain positive after correction
        result = InteractomeProcessor._size_correct_iptm(0.8, 200, 300)
        assert result > 0


# ---------------------------------------------------------------------------
# Helpers shared by pooled tests
# ---------------------------------------------------------------------------

_CHAINS = ["A", "B", "C"]
_N = {"A": 3, "B": 2, "C": 2}  # residues per chain
_CHAIN_TO_PROTEIN = {"A": "prot_A", "B": "prot_B", "C": "prot_C"}

# iptm_chain_pair[i, j] for chains sorted alphabetically (A=0, B=1, C=2)
_IPTM_MATRIX = np.array([
    [0.90, 0.70, 0.30],
    [0.70, 0.80, 0.40],
    [0.30, 0.40, 0.85],
])

_N_RES = sum(_N.values())  # 7


def _make_chain_ids():
    return np.array(["A"] * _N["A"] + ["B"] * _N["B"] + ["C"] * _N["C"])


def _make_mock_model():
    """Return a mock Model whose .metrics and .model_data mimic a 3-chain ColabFold output."""
    chain_ids = _make_chain_ids()

    mock_metrics = MagicMock()
    mock_metrics.pae = np.zeros((_N_RES, _N_RES))
    mock_metrics.cb_plddts = np.array(
        [80.0] * _N["A"] + [70.0] * _N["B"] + [60.0] * _N["C"]
    )
    mock_metrics.iptm_chain_pair = _IPTM_MATRIX.copy()

    mock_data = MagicMock()
    mock_data.token_chain_ids = chain_ids

    mock_model = MagicMock()
    mock_model.metrics = mock_metrics
    mock_model.model_data = mock_data
    return mock_model


def _make_ipsae_df(chains=_CHAINS):
    from itertools import permutations
    return pd.DataFrame([
        {"chain1": c1, "chain2": c2,
         "ipSAE": 0.5, "ipSAE_d0chn": 0.4, "ipSAE_d0dom": 0.6}
        for c1, c2 in permutations(chains, 2)
    ])


def _make_lis_df(chains=_CHAINS):
    from itertools import permutations
    return pd.DataFrame([
        {"chain1": c1, "chain2": c2,
         "LIS": 0.3, "LIA": 10.0, "cLIS": 0.2, "cLIA": 5.0,
         "iLIS": 0.245, "iLIA": 7.07, "LIR": "0,1", "cLIR": "0"}
        for c1, c2 in permutations(chains, 2)
    ])


def _make_pdockq2_df(chains=_CHAINS):
    from itertools import permutations
    return pd.DataFrame([
        {"chain1": c1, "chain2": c2, "pDockQ2": 0.15}
        for c1, c2 in permutations(chains, 2)
    ])


# ---------------------------------------------------------------------------
# _process_pool_model
# ---------------------------------------------------------------------------

REQUIRED_COLS = {
    "PPI", "ORF_A", "ORF_B", "Path", "ipTM", "size_corrected_ipTM",
    "pTM_chain_A", "pTM_chain_B",
    "pLDDT_mean", "pLDDT_mean_A", "pLDDT_mean_B",
    "pae_mean", "pae_mean_AB",
    "pDockQ2_AB", "pDockQ2_BA",
    "LIS_AB", "LIS_BA", "LIA_AB", "LIA_BA",
    "iLIS_AB", "iLIS_BA", "Best_LIS", "Best_iLIS",
    "ipSAE_AB", "ipSAE_BA", "max_ipSAE",
    "ipSAE_d0dom_AB", "ipSAE_d0dom_BA",
}


@pytest.fixture
def pool_model_patches():
    """Patch Model + metric functions so _process_pool_model runs without CIF files."""
    with patch("virus_interactome.interactome_processor.Model", return_value=_make_mock_model()), \
         patch("virus_interactome.interactome_processor.calculate_ipsae", return_value=_make_ipsae_df()), \
         patch("virus_interactome.interactome_processor.calculate_LIS_family", return_value=_make_lis_df()), \
         patch("virus_interactome.interactome_processor.calculate_pdockq2", return_value=_make_pdockq2_df()):
        yield


class TestProcessPoolModel:

    def test_returns_n_choose_2_pairs(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        assert len(results) == 3  # C(3,2)

    def test_ppi_names_use_separator(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD, ppi_separator="__"
        )
        ppis = {r["PPI"] for r in results}
        assert ppis == {"prot_A__prot_B", "prot_A__prot_C", "prot_B__prot_C"}

    def test_iptm_from_chain_pair_matrix(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        ab = next(r for r in results if r["PPI"] == "prot_A__prot_B")
        ac = next(r for r in results if r["PPI"] == "prot_A__prot_C")
        bc = next(r for r in results if r["PPI"] == "prot_B__prot_C")
        assert ab["ipTM"] == pytest.approx(_IPTM_MATRIX[0, 1])
        assert ac["ipTM"] == pytest.approx(_IPTM_MATRIX[0, 2])
        assert bc["ipTM"] == pytest.approx(_IPTM_MATRIX[1, 2])

    def test_ptm_chain_a_and_b_from_diagonal(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        ab = next(r for r in results if r["PPI"] == "prot_A__prot_B")
        assert ab["pTM_chain_A"] == pytest.approx(_IPTM_MATRIX[0, 0])  # A diagonal
        assert ab["pTM_chain_B"] == pytest.approx(_IPTM_MATRIX[1, 1])  # B diagonal

    def test_size_correction_differs_from_raw(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        for r in results:
            assert r["size_corrected_ipTM"] != pytest.approx(r["ipTM"])

    def test_plddt_mean_a_from_chain_a_residues_only(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        # Chain A has 3 residues all with pLDDT=80.0
        ab = next(r for r in results if r["PPI"] == "prot_A__prot_B")
        assert ab["pLDDT_mean_A"] == pytest.approx(80.0)
        assert ab["pLDDT_mean_B"] == pytest.approx(70.0)

    def test_required_columns_present(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        for r in results:
            assert REQUIRED_COLS <= set(r.keys()), f"Missing: {REQUIRED_COLS - set(r.keys())}"

    def test_orf_a_b_match_proteins(self, pool_model_patches, tmp_path):
        fake_cif = tmp_path / "pool_0000_rank_001.cif"
        from virus_interactome.model import Engine
        results = InteractomeProcessor._process_pool_model(
            fake_cif, _CHAIN_TO_PROTEIN, Engine.COLABFOLD
        )
        ab = next(r for r in results if r["PPI"] == "prot_A__prot_B")
        assert ab["ORF_A"] == "prot_A"
        assert ab["ORF_B"] == "prot_B"


# ---------------------------------------------------------------------------
# process_pooled
# ---------------------------------------------------------------------------

def _make_pool_record(ppi, orf_a, orf_b, iptm, pool_id):
    """Minimal per-model record returned by _process_pool_model."""
    return {
        "PPI": ppi, "ORF_A": orf_a, "ORF_B": orf_b,
        "pool_id": pool_id, "Path": f"/fake/{pool_id}/rank_001.cif",
        "ipTM": iptm, "size_corrected_ipTM": iptm - 0.05,
        "pLDDT_mean": 75.0, "pLDDT_mean_A": 80.0, "pLDDT_mean_B": 70.0,
        "pLDDT_median_A": 80.0, "pLDDT_median_B": 70.0,
        "pae_mean": 10.0, "pae_mean_A": 5.0, "pae_mean_B": 8.0, "pae_mean_AB": 12.0,
        "pDockQ2_AB": 0.15, "pDockQ2_BA": 0.12,
        "LIS_AB": 0.3, "LIS_BA": 0.25, "LIA_AB": 10.0, "LIA_BA": 8.0,
        "cLIS_AB": 0.2, "cLIS_BA": 0.18, "cLIA_AB": 5.0, "cLIA_BA": 4.0,
        "iLIS_AB": 0.245, "iLIS_BA": 0.21, "iLIA_AB": 7.0, "iLIA_BA": 5.7,
        "Best_LIS": 0.3, "Best_LIA": 10.0, "Best_iLIS": 0.245, "Best_iLIA": 7.0,
        "LIR_AB": "0,1", "cLIR_AB": "0",
        "ipSAE_AB": 0.5, "ipSAE_BA": 0.45, "max_ipSAE": 0.5,
        "ipSAE_d0chn_AB": 0.4, "ipSAE_d0chn_BA": 0.38,
        "ipSAE_d0dom_AB": 0.6, "ipSAE_d0dom_BA": 0.55,
        "pTM_chain_A": 0.9, "pTM_chain_B": 0.8,
    }


@pytest.fixture
def pooled_env(tmp_path):
    """
    Filesystem + manifest for two pools:
      pool_0000: prot_A, prot_B, prot_C  → pairs A__B (ipTM=0.6), A__C (0.4), B__C (0.3)
      pool_0001: prot_A, prot_B, prot_D  → pairs A__B (ipTM=0.8), A__D (0.5), B__D (0.2)
    prot_A__prot_B appears in both pools → n_pools=2, mean ipTM=0.7
    """
    import csv

    manifest_rows = [
        {"pool_id": "pool_0000", "proteins": "prot_A,prot_B,prot_C", "n_proteins": 3, "total_aa": 90, "n_pairs": 3},
        {"pool_id": "pool_0001", "proteins": "prot_A,prot_B,prot_D", "n_proteins": 3, "total_aa": 85, "n_pairs": 3},
    ]
    manifest_path = tmp_path / "pool_manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # Create pool directories with a dummy .cif file each
    for pid in ("pool_0000", "pool_0001"):
        d = tmp_path / "cf_output" / pid
        d.mkdir(parents=True)
        (d / f"{pid}_rank_001.cif").touch()

    # Side-effect function for _process_pool_model
    pool_data = {
        "pool_0000": [
            _make_pool_record("prot_A__prot_B", "prot_A", "prot_B", 0.6, "pool_0000"),
            _make_pool_record("prot_A__prot_C", "prot_A", "prot_C", 0.4, "pool_0000"),
            _make_pool_record("prot_B__prot_C", "prot_B", "prot_C", 0.3, "pool_0000"),
        ],
        "pool_0001": [
            _make_pool_record("prot_A__prot_B", "prot_A", "prot_B", 0.8, "pool_0001"),
            _make_pool_record("prot_A__prot_D", "prot_A", "prot_D", 0.5, "pool_0001"),
            _make_pool_record("prot_B__prot_D", "prot_B", "prot_D", 0.2, "pool_0001"),
        ],
    }

    def _fake_process(cif_path, chain_to_protein, engine, ppi_separator="__"):
        pool_id = cif_path.parent.name
        return pool_data[pool_id]

    return {
        "manifest": manifest_path,
        "cf_dir": tmp_path / "cf_output",
        "out_dir": tmp_path / "results",
        "fake_process": _fake_process,
    }


class TestProcessPooled:

    def test_creates_output_csv(self, pooled_env):
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            InteractomeProcessor.process_pooled(
                pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
            )
        assert (pooled_env["out_dir"] / "interactome_data.csv").exists()

    def test_one_row_per_unique_ppi(self, pooled_env):
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            df = InteractomeProcessor.process_pooled(
                pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
            )
        assert len(df) == 5  # A__B, A__C, B__C, A__D, B__D

    def test_n_pools_counted_correctly(self, pooled_env):
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            df = InteractomeProcessor.process_pooled(
                pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
            )
        ab = df.loc[df["PPI"] == "prot_A__prot_B", "n_pools"].values[0]
        assert ab == 2
        ac = df.loc[df["PPI"] == "prot_A__prot_C", "n_pools"].values[0]
        assert ac == 1

    def test_iptm_averaged_across_pools(self, pooled_env):
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            df = InteractomeProcessor.process_pooled(
                pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
            )
        ab_iptm = df.loc[df["PPI"] == "prot_A__prot_B", "ipTM"].values[0]
        assert ab_iptm == pytest.approx((0.6 + 0.8) / 2, abs=1e-4)

    def test_single_pool_iptm_unchanged(self, pooled_env):
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            df = InteractomeProcessor.process_pooled(
                pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
            )
        ac_iptm = df.loc[df["PPI"] == "prot_A__prot_C", "ipTM"].values[0]
        assert ac_iptm == pytest.approx(0.4, abs=1e-4)

    def test_missing_pool_dir_skipped(self, pooled_env, caplog):
        import logging
        import shutil
        shutil.rmtree(pooled_env["cf_dir"] / "pool_0001")
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            with caplog.at_level(logging.WARNING, logger="virus_interactome.interactome"):
                df = InteractomeProcessor.process_pooled(
                    pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
                )
        # Only pool_0000 processed → 3 PPIs
        assert len(df) == 3
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_empty_pool_dir_skipped(self, pooled_env, caplog):
        import logging
        # Remove the CIF from pool_0001 so glob returns nothing
        for f in (pooled_env["cf_dir"] / "pool_0001").glob("*.cif"):
            f.unlink()
        with patch.object(InteractomeProcessor, "_process_pool_model",
                          side_effect=pooled_env["fake_process"]):
            with caplog.at_level(logging.WARNING, logger="virus_interactome.interactome"):
                df = InteractomeProcessor.process_pooled(
                    pooled_env["manifest"], pooled_env["cf_dir"], pooled_env["out_dir"]
                )
        assert len(df) == 3
        assert any("no cif" in r.message.lower() for r in caplog.records)
