"""Tests for _PeptidePipelineMixin using a real two-chain ColabFold PDB fixture."""
import logging
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from moleculekit.molecule import Molecule

from virus_interactome.interactome_analyzer import InteractomeAnalyzer

DATA = Path(__file__).parent.parent / "data"
COMPLEX_PDB = (
    DATA / "colabfold_dummy_example"
    / "pVI__protease_unrelaxed_rank_001_alphafold2_multimer_v3_model_3_seed_000.pdb"
)


@pytest.fixture
def complex_pdb(tmp_path):
    """Copy of the real pVI(chain A, 250 res)/protease(chain B, 204 res) complex."""
    dest = tmp_path / "complex.pdb"
    shutil.copy(COMPLEX_PDB, dest)
    return dest


# ---------------------------------------------------------------------------
# _get_candidate_clusters — malformed PPI fallback (no "__" separator)
# ---------------------------------------------------------------------------

class TestGetCandidateClustersMalformedPPI:
    def test_ppi_without_separator_fallback(self):
        analyzer = InteractomeAnalyzer()
        analyzer._cluster_data = pd.DataFrame({
            "PPI": ["SingleProtein"],
            "path": ["/fake/m.cif"],
            "cluster_id": [0],
            "cluster_ratio": [10.0],
            "x_len": [50], "y_len": [5],
            "x_min": [0], "x_max": [50],
            "y_min": [0], "y_max": [5],
        })
        result = analyzer._get_candidate_clusters(cluster_ratio_threshold=7.0, min_peptide_len=5)
        assert len(result) == 1
        assert result.iloc[0]["Binder_name"] == "SingleProtein"
        assert result.iloc[0]["Peptide_name"] == ""


# ---------------------------------------------------------------------------
# _curate_protein_peptide_models
# ---------------------------------------------------------------------------

class TestCurateProteinPeptideModels:
    def test_no_swap_keeps_chain_assignment(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        row = pd.Series({"path": str(complex_pdb), "Peptide_chain": "B",
                          "Peptide_start": 10, "Peptide_end": 50})
        mol = analyzer._curate_protein_peptide_models(row)
        assert set(np.unique(mol.chain)) == {"A", "B"}
        assert mol.resid[mol.chain == "A"].max() == 250
        assert mol.resid[mol.chain == "B"].min() >= 10
        assert mol.resid[mol.chain == "B"].max() <= 50

    def test_swap_when_peptide_is_chain_a(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        row = pd.Series({"path": str(complex_pdb), "Peptide_chain": "A",
                          "Peptide_start": 10, "Peptide_end": 50})
        mol = analyzer._curate_protein_peptide_models(row)
        # Original chain B (204 res) becomes the binder (new chain A), kept whole
        assert mol.resid[mol.chain == "A"].max() == 204
        # Original chain A (peptide) becomes chain B, filtered to the interface range
        assert mol.resid[mol.chain == "B"].min() >= 10
        assert mol.resid[mol.chain == "B"].max() <= 50


# ---------------------------------------------------------------------------
# _get_reference_structure_for_binder
# ---------------------------------------------------------------------------

class TestGetReferenceStructureForBinder:
    def test_selects_highest_plddt_and_filters_high_confidence(self, complex_pdb, tmp_path):
        analyzer = InteractomeAnalyzer()
        # struct2 has chain A pLDDT boosted to 90 -> highest median, all residues > 70
        mol2 = Molecule(str(complex_pdb))
        mol2.set("beta", 90.0, "chain A")
        struct2 = tmp_path / "struct2.pdb"
        mol2.write(str(struct2))

        ref = analyzer._get_reference_structure_for_binder([str(complex_pdb), str(struct2)])
        assert set(np.unique(ref.chain)) == {"A"}
        assert len(np.unique(ref.resid)) == 250

    def test_no_high_confidence_residues_returns_full_chain(self, complex_pdb, tmp_path, caplog):
        analyzer = InteractomeAnalyzer()
        mol = Molecule(str(complex_pdb))
        mol.set("beta", 50.0, "chain A")
        struct = tmp_path / "lowconf.pdb"
        mol.write(str(struct))

        caplog.set_level(logging.WARNING)
        ref = analyzer._get_reference_structure_for_binder([str(struct)])
        assert "No high-confidence residues" in caplog.text
        assert set(np.unique(ref.chain)) == {"A"}
        assert len(np.unique(ref.resid)) == 250


# ---------------------------------------------------------------------------
# _create_binder_alignments
# ---------------------------------------------------------------------------

class TestCreateBinderAlignments:
    def test_aligns_to_reference_chain_a(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        ref_mol = Molecule(str(complex_pdb))
        ref_mol.filter("chain A")

        aligned = analyzer._create_binder_alignments(str(complex_pdb), ref_mol)
        assert set(np.unique(aligned.chain)) == {"A", "B"}
        assert len(np.unique(aligned.resid[aligned.chain == "A"])) == 250

    def test_reference_as_path_string(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        ref_path = str(complex_pdb)
        aligned = analyzer._create_binder_alignments(str(complex_pdb), ref_path)
        assert set(np.unique(aligned.chain)) == {"A", "B"}

    def test_invalid_reference_type_raises(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="reference_model"):
            analyzer._create_binder_alignments(str(complex_pdb), 12345)


# ---------------------------------------------------------------------------
# cluster_protein_peptides
# ---------------------------------------------------------------------------

class TestClusterProteinPeptides:
    def test_no_chain_b_atoms_returns_empty(self, complex_pdb, tmp_path, caplog):
        analyzer = InteractomeAnalyzer()
        mol = Molecule(str(complex_pdb))
        mol.filter("chain A")
        chain_a_only = tmp_path / "chainA.pdb"
        mol.write(str(chain_a_only))

        caplog.set_level(logging.WARNING)
        df, info = analyzer.cluster_protein_peptides([str(chain_a_only)], str(chain_a_only))
        assert df.empty
        assert info["cluster_labels"].size == 0
        assert info["peptide_centers"].size == 0
        assert "no Chain B CA atoms" in caplog.text

    def test_clusters_two_identical_models(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        ref_mol = Molecule(str(complex_pdb))
        ref_mol.filter("chain A")

        df, info = analyzer.cluster_protein_peptides(
            [str(complex_pdb), str(complex_pdb)], ref_mol, eps=50, min_samples=1
        )
        assert len(df) == 1
        assert df.iloc[0]["Cluster_label"] == 0
        assert info["cluster_labels"].tolist() == [0, 0]
        assert info["peptide_centers"].shape == (2, 3)

    def test_invalid_reference_type_raises(self, complex_pdb):
        analyzer = InteractomeAnalyzer()
        with pytest.raises(ValueError, match="reference_model"):
            analyzer.cluster_protein_peptides([str(complex_pdb)], 12345)


# ---------------------------------------------------------------------------
# _create_chimera_session
# ---------------------------------------------------------------------------

class TestCreateChimeraSession:
    @pytest.fixture
    def ppi_and_cluster_info(self, tmp_path):
        binder_dir = tmp_path / "prot_peptide" / "pVI"
        (binder_dir / "filtered").mkdir(parents=True)
        ppi_data = pd.DataFrame({
            "Binder_name": ["pVI", "pVI"],
            "filtered_path": [
                str(binder_dir / "filtered" / "m0.pdb"),
                str(binder_dir / "filtered" / "m1.pdb"),
            ],
            "PPI": ["pVI__protease_0_0", "pVI__protease_1_0"],
            "Cluster_info": [0, -1],
            "Center_X": [1.0, 2.0],
            "Center_Y": [1.0, 2.0],
            "Center_Z": [1.0, 2.0],
        })
        cluster_info = pd.DataFrame({
            "Cluster_label": [0, -1],
            "Center_X": [1.0, 0.0],
            "Center_Y": [1.0, 0.0],
            "Center_Z": [1.0, 0.0],
            "Residues": [np.array([1, 2, 3]), np.array([])],
        })
        return binder_dir, ppi_data, cluster_info

    def test_writes_script_and_handles_missing_chimerax(self, tmp_path, complex_pdb, caplog,
                                                          ppi_and_cluster_info):
        binder_dir, ppi_data, cluster_info = ppi_and_cluster_info
        analyzer = InteractomeAnalyzer(output_path=tmp_path)

        caplog.set_level(logging.ERROR)
        with patch("virus_interactome._peptide_mixin.subprocess.run", side_effect=FileNotFoundError):
            analyzer._create_chimera_session(ppi_data, str(complex_pdb), cluster_info)

        script = binder_dir / "pVI_peptide_binding.cxc"
        content = script.read_text()
        assert "open \"" in content
        assert "Cluster_1" in content
        assert "Unclassified" in content
        assert "ChimeraX executable not found" in caplog.text

    def test_called_process_error_logged(self, tmp_path, complex_pdb, caplog,
                                           ppi_and_cluster_info):
        binder_dir, ppi_data, cluster_info = ppi_and_cluster_info
        analyzer = InteractomeAnalyzer(output_path=tmp_path)

        caplog.set_level(logging.ERROR)
        with patch("virus_interactome._peptide_mixin.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, "chimerax")):
            analyzer._create_chimera_session(ppi_data, str(complex_pdb), cluster_info)

        assert "ChimeraX execution failed" in caplog.text


# ---------------------------------------------------------------------------
# analyze_peptide_proteins_pairs (full pipeline, end-to-end with real structures)
# ---------------------------------------------------------------------------

class TestAnalyzePeptideProteinsPairs:
    @pytest.fixture
    def two_model_cluster_data(self, complex_pdb):
        """x_len > y_len -> Binder=pVI(chain A), Peptide=protease(chain B), interface 10-50."""
        return pd.DataFrame({
            "PPI": ["pVI__protease", "pVI__protease"],
            "path": [str(complex_pdb), str(complex_pdb)],
            "model_num": [0, 1],
            "cluster_id": [0, 0],
            "cluster_ratio": [10.0, 10.0],
            "x_len": [50, 50], "y_len": [5, 5],
            "x_min": [0, 0], "x_max": [249, 249],
            "y_min": [10, 10], "y_max": [50, 50],
        })

    def test_full_pipeline_creates_outputs(self, tmp_path, two_model_cluster_data):
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = two_model_cluster_data

        with patch("virus_interactome._peptide_mixin.subprocess.run") as mock_run:
            analyzer.analyze_peptide_proteins_pairs(
                eps=50, min_samples=1, cluster_ratio_threshold=7.0, min_peptide_len=5
            )
        mock_run.assert_called_once()

        out = tmp_path / "prot_peptide" / "pVI"
        assert (out / "reference_pVI.pdb").exists()
        assert (out / "filtered" / "pVI__protease_0_0.pdb").exists()
        assert (out / "aligned" / "pVI__protease_0_0.pdb").exists()
        assert (out / "pVI_peptide_binding.cxc").exists()

        binder_csv = tmp_path / "peptide_binder_info.csv"
        assert binder_csv.exists()
        binder_df = pd.read_csv(binder_csv)
        assert (binder_df["Binder"] == "pVI").all()

    def test_second_run_reuses_existing_files(self, tmp_path, two_model_cluster_data, caplog):
        """Filtered/aligned/reference files are not regenerated if already present."""
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = two_model_cluster_data

        with patch("virus_interactome._peptide_mixin.subprocess.run"):
            analyzer.analyze_peptide_proteins_pairs(
                eps=50, min_samples=1, cluster_ratio_threshold=7.0, min_peptide_len=5
            )

            caplog.set_level(logging.INFO)
            analyzer.analyze_peptide_proteins_pairs(
                eps=50, min_samples=1, cluster_ratio_threshold=7.0, min_peptide_len=5
            )

        assert "already filtered" in caplog.text
        assert "Skipping reference generation" in caplog.text

    def test_no_candidates_logs_warning_and_returns(self, tmp_path, caplog):
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = pd.DataFrame({
            "PPI": ["A__B"], "path": ["/fake/m.cif"], "cluster_id": [0],
            "cluster_ratio": [1.0], "x_len": [50], "y_len": [5],
            "x_min": [0], "x_max": [50], "y_min": [0], "y_max": [5],
        })
        caplog.set_level(logging.WARNING)
        analyzer.analyze_peptide_proteins_pairs()
        assert "No candidate peptide-protein clusters found" in caplog.text

    def test_no_spatial_clusters_skips_chimera(self, tmp_path, two_model_cluster_data, caplog,
                                                monkeypatch):
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = two_model_cluster_data
        monkeypatch.setattr(
            analyzer, "cluster_protein_peptides",
            lambda *a, **k: (pd.DataFrame(), {"cluster_labels": np.array([])}),
        )

        caplog.set_level(logging.WARNING)
        with patch("virus_interactome._peptide_mixin.subprocess.run") as mock_run:
            analyzer.analyze_peptide_proteins_pairs(cluster_ratio_threshold=7.0, min_peptide_len=5)

        assert "No spatial clusters found" in caplog.text
        mock_run.assert_not_called()

    def test_shape_mismatch_logs_error(self, tmp_path, two_model_cluster_data, caplog, monkeypatch):
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = two_model_cluster_data

        # Fake cluster result with only 1 label for 2 models -> shape mismatch
        tmp_df = pd.DataFrame({
            "Cluster_label": [0], "Center_X": [0.0], "Center_Y": [0.0], "Center_Z": [0.0],
            "Residues": [np.array([1])],
        })
        cluster_info = {"cluster_labels": np.array([0]), "peptide_centers": np.array([[0.0, 0.0, 0.0]])}
        monkeypatch.setattr(analyzer, "cluster_protein_peptides", lambda *a, **k: (tmp_df, cluster_info))

        caplog.set_level(logging.ERROR)
        with patch("virus_interactome._peptide_mixin.subprocess.run") as mock_run:
            analyzer.analyze_peptide_proteins_pairs(cluster_ratio_threshold=7.0, min_peptide_len=5)

        assert "Shape mismatch" in caplog.text
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# run_full_pipeline -> analyze_peptide_proteins_pairs (real structural pipeline)
# ---------------------------------------------------------------------------

class TestRunFullPipelineWithRealStructures:
    def test_run_full_pipeline_executes_structural_analysis(self, tmp_path, complex_pdb):
        cluster_data = pd.DataFrame({
            "PPI": ["pVI__protease", "pVI__protease"],
            "path": [str(complex_pdb), str(complex_pdb)],
            "model_num": [0, 1],
            "cluster_id": [0, 0],
            "cluster_ratio": [10.0, 10.0],
            "x_len": [50, 50], "y_len": [5, 5],
            "x_min": [0, 0], "x_max": [249, 249],
            "y_min": [10, 10], "y_max": [50, 50],
        })
        analyzer = InteractomeAnalyzer(output_path=tmp_path)
        analyzer._cluster_data = cluster_data

        with patch("virus_interactome._peptide_mixin.subprocess.run"):
            analyzer.run_full_pipeline(eps=50, min_samples=1)

        assert (tmp_path / "peptide_binder_info.csv").exists()
