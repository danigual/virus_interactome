import pytest
import logging
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
from virus_interactome.proteome_manager import ProteomeManager


# --- Loading tests ---

def test_load_proteome_missing_file():
    with pytest.raises(FileNotFoundError):
        ProteomeManager("non_existent_file.fasta")


def test_load_proteome_empty_file(tmp_path):
    pm = ProteomeManager()
    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("")
    result = pm.load_proteome(str(empty_file))
    assert result == {}


def test_load_proteome_basic(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    expected = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected
    assert pm.file_path == str(dummy_fasta_path)


def test_load_proteome_with_repeated_protein_id(dummy_fasta_with_repeated_protein_id_path, caplog):
    caplog.set_level(logging.WARNING)
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_fasta_with_repeated_protein_id_path))
    assert "Duplicate protein ID 'Protein1_isoformB' found" in caplog.text
    expected = {
        "Protein1_isoformB": "RQDILDMKTAYIAKQRQISFVKSHFS",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected


def test_load_non_fasta_file(dummy_non_fasta_proteome):
    pm = ProteomeManager(str(dummy_non_fasta_proteome))
    expected = {
        "Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILD",
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert isinstance(pm.sequences, dict)
    assert pm.sequences == expected


def test_mixed_sequences(dummy_proteome_with_invalid_sequences, caplog):
    caplog.set_level(logging.WARNING)
    pm = ProteomeManager()
    pm.load_proteome(str(dummy_proteome_with_invalid_sequences))
    assert "Invalid amino acids in Protein1_isoformB. Sequence skipped." in caplog.text
    expected = {
        "Protein2": "GVALSKGEEAVRLFK",
        "Protein3_variantA": "LLKSDGQVLKAV",
        "Protein4_isoformC": "MKQLQKDLGK",
    }
    assert pm.sequences == expected
    assert {"Protein1_isoformB": "MKTAYIAKQRQISFVKSHFSRQDILDZZZ"} == pm.invalid_sequences


# --- Identity tests ---

def test_compute_identity_empty_proteome():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_identity()


def test_compute_identity_single_protein():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "MKTAYIAKQR"}
    df = pm.compute_identity(n_jobs=1)
    assert df.empty


def test_compute_identity_two_proteins():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT"}
    df = pm.compute_identity(n_jobs=1)
    assert len(df) == 1
    assert set(df.columns) == {"ORF1", "ORF2", "Identity"}
    assert 0.75 <= df.iloc[0]["Identity"] <= 1.0


@pytest.mark.slow
def test_compute_identity_multiprocessing():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity(n_jobs=2)
    assert len(df) == 3
    assert set(df.columns) == {"ORF1", "ORF2", "Identity"}

    def _get(a, b):
        row = df[((df.ORF1 == a) & (df.ORF2 == b)) | ((df.ORF1 == b) & (df.ORF2 == a))]
        return row.iloc[0]["Identity"]

    assert _get("Protein1", "Protein3") == pytest.approx(0.0)
    assert _get("Protein1", "Protein2") == pytest.approx(0.75)
    assert _get("Protein2", "Protein3") == pytest.approx(0.25)


@pytest.mark.slow
def test_compute_identity_with_similarity_threshold():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity(n_jobs=2, similarity_threshold=0.7)
    assert len(df) == 3
    high_sim_pairs = pm.high_similarity_pairs
    assert ("Protein1", "Protein2", 0.75) in high_sim_pairs


# --- ids property ---

def test_ids_insertion_order():
    pm = ProteomeManager()
    pm.sequences = {"C": "AAA", "A": "GGG", "B": "TTT"}
    assert pm.ids == ("C", "A", "B")


def test_ids_cached():
    pm = ProteomeManager()
    pm.sequences = {"P1": "AAA", "P2": "GGG"}
    first = pm.ids
    second = pm.ids
    assert first is second


# --- filter_by_regex ---

def test_filter_by_regex_ids_only(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"isoform")
    assert isinstance(result, list)
    assert "Protein1_isoformB" in result
    assert "Protein4_isoformC" in result
    assert "Protein2" not in result


def test_filter_by_regex_return_sequences(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"variant", return_sequences=True)
    assert isinstance(result, dict)
    assert "Protein3_variantA" in result
    assert result["Protein3_variantA"] == "LLKSDGQVLKAV"


def test_filter_by_regex_no_matches(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.filter_by_regex(r"NOMATCH_XYZ_999")
    assert result == []


# --- compute_properties ---

def test_compute_properties_empty_raises():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_properties()


def test_compute_properties_columns(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    df = pm.compute_properties()
    expected_cols = {"length", "molecular_weight", "isoelectric_point",
                     "instability_index", "gravy", "aromaticity"}
    assert expected_cols.issubset(set(df.columns))


def test_compute_properties_values(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    df = pm.compute_properties()
    assert df.index.name == "id"
    assert set(df.index) == set(pm.sequences.keys())
    # Sanity: lengths match sequence lengths
    for pid, seq in pm.sequences.items():
        assert df.loc[pid, "length"] == len(seq)
    # MW and pI are positive numbers
    assert (df["molecular_weight"] > 0).all()
    assert (df["isoelectric_point"] > 0).all()


# --- summary ---

def test_summary_dict(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    result = pm.summary()
    assert isinstance(result, dict)
    expected_keys = {"total_sequences", "total_residues", "average_length",
                     "min_length", "max_length", "invalid_sequences", "high_similarity_pairs"}
    assert expected_keys == set(result.keys())
    assert result["total_sequences"] == 4
    assert result["min_length"] <= result["average_length"] <= result["max_length"]


def test_summary_empty_proteome():
    pm = ProteomeManager()
    result = pm.summary()
    assert result["total_sequences"] == 0
    assert result["total_residues"] == 0
    assert result["min_length"] == 0


# --- get_sequence ---

def test_get_sequence_found(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    seq = pm.seq_from_id("Protein2")
    assert seq == "GVALSKGEEAVRLFK"


def test_get_sequence_not_found(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    with pytest.raises(KeyError):
        pm.seq_from_id("NONEXISTENT_PROTEIN")


# --- __str__ ---

def test_str_representation(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    s = str(pm)
    assert "ProteomeManager Summary" in s
    assert "Total sequences" in s


# --- sorted ids mode ---

def test_ids_sorted_order():
    pm = ProteomeManager()
    pm.sequences = {"C": "AAA", "A": "GGG", "B": "TTT"}
    pm._order_mode = "sorted"
    pm._ids_cache = None  # force recompute
    assert pm.ids == ("A", "B", "C")


# --- __len__ ---

def test_len_returns_sequence_count(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    assert len(pm) == 4


def test_len_empty_proteome():
    pm = ProteomeManager()
    assert len(pm) == 0


# --- get_ids ---

def test_get_ids(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    ids = pm.ids
    assert isinstance(ids, tuple)
    assert set(ids) == {"Protein1_isoformB", "Protein2", "Protein3_variantA", "Protein4_isoformC"}


# --- _compute_identity static method ---

def test_compute_identity_identical():
    i, j, score = ProteomeManager._compute_identity((0, 1, "AAAA", "AAAA"))
    assert i == 0
    assert j == 1
    assert score == pytest.approx(1.0)


def test_compute_identity_different():
    i, j, score = ProteomeManager._compute_identity((0, 1, "AAAA", "TTTT"))
    assert score == pytest.approx(0.0)


# --- normalize_fasta_headers ---

_NCBI_FASTA = """\
>lcl|NC_001.1 [protein=Hexon fiber/major] [gene=HEX]
MKTAYIAKQR
QISFVKSHFS
>lcl|NC_002.1 [protein=DNA polymerase] [gene=POL]
GVALSKGEEA
"""

def test_normalize_fasta_headers_default_parser(tmp_path):
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(_NCBI_FASTA)

    ProteomeManager.normalize_fasta_headers(str(infile), str(outfile))

    lines = outfile.read_text().splitlines()
    assert lines[0] == ">Hexon_fiber_major|lcl|NC_001.1 [protein=Hexon fiber/major] [gene=HEX]"
    assert lines[1] == "MKTAYIAKQR"
    assert lines[2] == "QISFVKSHFS"
    assert lines[3] == ">DNA_polymerase|lcl|NC_002.1 [protein=DNA polymerase] [gene=POL]"
    assert lines[4] == "GVALSKGEEA"


def test_normalize_fasta_headers_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProteomeManager.normalize_fasta_headers(
            str(tmp_path / "missing.fasta"), str(tmp_path / "out.fasta")
        )


def test_normalize_fasta_headers_custom_parser(tmp_path):
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(">gi|12345|ref|NP_001.1| some description\nMKTAY\n")

    # Extract the gi number as ID
    ProteomeManager.normalize_fasta_headers(
        str(infile), str(outfile),
        header_parser=lambda h: h.split("|")[1],
    )

    lines = outfile.read_text().splitlines()
    assert lines[0] == ">12345|gi|12345|ref|NP_001.1| some description"
    assert lines[1] == "MKTAY"


def test_normalize_fasta_headers_no_protein_field_falls_back(tmp_path):
    """Headers without protein= field fall back to first whitespace-delimited token."""
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(">MyProtein some description\nAAAA\n")

    ProteomeManager.normalize_fasta_headers(str(infile), str(outfile))

    lines = outfile.read_text().splitlines()
    assert lines[0] == ">MyProtein|MyProtein some description"


def test_normalize_fasta_headers_empty_id_raises(tmp_path):
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(">valid header\nAAAA\n")

    with pytest.raises(ValueError, match="empty ID"):
        ProteomeManager.normalize_fasta_headers(
            str(infile), str(outfile),
            header_parser=lambda h: "",
        )


def test_normalize_fasta_headers_special_chars_replaced(tmp_path):
    """Spaces, dots and slashes in protein= value are replaced with underscores."""
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(">x [protein=A.B/C D] y\nGGGG\n")

    ProteomeManager.normalize_fasta_headers(str(infile), str(outfile))

    first_line = outfile.read_text().splitlines()[0]
    new_id = first_line.split("|")[0][1:]  # strip leading >
    assert new_id == "A_B_C_D"


def test_normalize_fasta_headers_output_loadable(tmp_path):
    """Output FASTA can be loaded by ProteomeManager after normalisation."""
    infile = tmp_path / "raw.fasta"
    outfile = tmp_path / "clean.fasta"
    infile.write_text(_NCBI_FASTA)

    ProteomeManager.normalize_fasta_headers(str(infile), str(outfile))
    pm = ProteomeManager(str(outfile))

    assert "Hexon_fiber_major" in pm.sequences
    assert "DNA_polymerase" in pm.sequences


# =============================================================================
# _get_orf_id_from_path
# =============================================================================

def test_get_orf_id_af3_returns_parent_dir_name(tmp_path):
    model_file = tmp_path / "E1A" / "model_0.cif"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "af3") == "E1A"


def test_get_orf_id_boltz_returns_parent_dir_name(tmp_path):
    model_file = tmp_path / "pVII" / "model_0.cif"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "boltz") == "pVII"


def test_get_orf_id_colabfold_orf_token_in_basename(tmp_path):
    model_file = tmp_path / "E1A" / "prefix_ORF3_model_1_rank_001.pdb"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "colabfold") == "ORF3"


def test_get_orf_id_colabfold_fallback_model_split(tmp_path):
    model_file = tmp_path / "E1A" / "E1A_model_1_rank_001.pdb"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "colabfold") == "E1A"


def test_get_orf_id_colabfold_no_marker_returns_full_basename(tmp_path):
    model_file = tmp_path / "E1A" / "myfile.pdb"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "colabfold") == "myfile.pdb"


def test_get_orf_id_unknown_engine_returns_none(tmp_path):
    model_file = tmp_path / "E1A" / "model_0.cif"
    assert ProteomeManager._get_orf_id_from_path(str(model_file), "unknown_engine") is None


# =============================================================================
# load_model_info helpers
# =============================================================================

def _make_model_dir(tmp_path, orfs, file_ext="cif", n_models=2):
    """Create a fake engine output directory: tmp_path/{orf}/model_{i}.{ext}"""
    for orf in orfs:
        orf_dir = tmp_path / orf
        orf_dir.mkdir(exist_ok=True)
        for i in range(n_models):
            (orf_dir / f"model_{i}.{file_ext}").touch()
    return tmp_path


def _mock_monomer(path, engine):
    """Synchronous stand-in for _load_model_info_monomer. Returns pd.Series matching Model.summary()."""
    import pandas as pd
    from pathlib import Path
    orf_id = Path(path).parent.name  # AF3/Boltz convention: parent dir is the ORF id
    eng_val = engine.value if hasattr(engine, "value") else str(engine)
    return pd.Series({
        "id": orf_id,
        "model_num": 0,
        "engine": eng_val,
        "ptm": 0.8,
        "iptm": 0.7,
        "mean_plddt": 75.0,
        "mean_pae": 5.0,
        "path": path,
    })


class _SyncPool:
    """Drop-in for ProcessPoolExecutor that runs work in-process."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, fn, iterable): return list(map(fn, iterable))


# =============================================================================
# load_model_info tests
# =============================================================================

def test_load_model_info_all_match_no_warnings(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["E1A", "pVII"], n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY", "pVII": "GVALS"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert len(df) == 4  # 2 ORFs × 2 models
    assert set(df["id"]) == {"E1A", "pVII"}
    assert "skipping" not in caplog.text


def test_load_model_info_missing_structure_warns(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["E1A"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY", "pX": "GVALS"}  # pX has no models

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        pm.load_model_info(str(model_dir), engine="AF3")

    assert "pX" in caplog.text
    assert "no model files found" in caplog.text


def test_load_model_info_ignores_models_without_sequence(tmp_path, caplog):
    """Directories with models but no matching sequence are silently skipped."""
    model_dir = _make_model_dir(tmp_path, ["E1A", "pVII"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # pVII has models but no sequence

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert len(df) == 1
    assert "pVII" not in caplog.text


def test_load_model_info_no_sequences_returns_empty(tmp_path, caplog):
    """If no sequences are loaded, load_model_info has nothing to iterate and returns empty."""
    model_dir = _make_model_dir(tmp_path, ["E1A", "pVII"], n_models=2)
    pm = ProteomeManager()  # no FASTA loaded → sequences = {}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert df.empty


def test_load_model_info_no_valid_files_returns_empty_df(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["pVII"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # E1A dir does not exist → no cif files

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert df.empty
    assert "No model files found" in caplog.text


def test_load_model_info_exact_match_required(tmp_path, caplog):
    """Directory name must match sequence ID exactly; case mismatch means no models found."""
    model_dir = _make_model_dir(tmp_path, ["e1a"], n_models=1)  # lowercase dir
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # uppercase key

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert df.empty
    assert "E1A" in caplog.text  # warns about missing models for E1A


def test_load_model_info_colabfold_uses_pdb_ext(tmp_path):
    """ColabFold engine auto-selects .pdb files."""
    model_dir = _make_model_dir(tmp_path, ["E1A"], file_ext="pdb", n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        df = pm.load_model_info(str(model_dir), engine="ColabFold")

    assert len(df) == 2


def test_load_model_info_af3_ignores_pdb_files(tmp_path):
    """AF3 engine auto-selects .cif; .pdb files in the same dir are not picked up."""
    model_dir = _make_model_dir(tmp_path, ["E1A"], file_ext="pdb", n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert df.empty


def test_load_model_info_sets_model_info_attributes(tmp_path):
    model_dir = _make_model_dir(tmp_path, ["E1A"], n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "_load_model_info_monomer", side_effect=_mock_monomer):
        pm.load_model_info(str(model_dir), engine="AF3")

    assert pm.model_info_by_model is not None
    assert pm.model_info_by_orf is not None
    assert "id" in pm.model_info_by_orf.columns
    assert len(pm.model_info_by_orf) == 1  # 1 ORF, averaged across models


# ---------------------------------------------------------------------------
# file_path setter / load_proteome edge cases
# ---------------------------------------------------------------------------

def test_file_path_setter_loads_proteome(dummy_fasta_path):
    pm = ProteomeManager()
    pm.file_path = str(dummy_fasta_path)
    assert len(pm) == 4


def test_load_proteome_none_returns_without_error():
    pm = ProteomeManager()
    result = pm.load_proteome(None)
    assert result is None
    assert len(pm) == 0


# ---------------------------------------------------------------------------
# align_sequences
# ---------------------------------------------------------------------------

class TestAlignSequences:

    def test_identical_sequences_give_full_identity(self):
        seq = "MKTAYIAKQRQ"
        result = ProteomeManager.align_sequences(seq, seq)
        assert result["identity"] == pytest.approx(100.0)

    def test_returns_all_keys(self):
        result = ProteomeManager.align_sequences("MKTAY", "MKTAY")
        assert {"score", "identity", "coverage", "gaps"} == set(result.keys())

    def test_different_sequences_give_partial_identity(self):
        result = ProteomeManager.align_sequences("MKTAYIAKQRQ", "AAAAAAAAA")
        assert result["identity"] < 100.0

    def test_coverage_full_for_identical(self):
        seq = "MKTAYIAKQRQ"
        result = ProteomeManager.align_sequences(seq, seq)
        assert result["coverage"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# filter_proteome
# ---------------------------------------------------------------------------

class TestFilterProteome:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"A": "MKTAY", "B": "GVALSK", "C": "LLKSD"}
        return pm

    def test_keeps_specified_orfs(self):
        pm = self._make_pm()
        pm.filter_proteome(["A", "C"])
        assert set(pm.ids) == {"A", "C"}

    def test_missing_orf_warns(self, caplog):
        pm = self._make_pm()
        with caplog.at_level(logging.WARNING):
            pm.filter_proteome(["A", "MISSING"])
        assert "MISSING" in caplog.text

    def test_empty_proteome_warns(self, caplog):
        pm = ProteomeManager()
        with caplog.at_level(logging.WARNING):
            pm.filter_proteome(["A"])
        assert caplog.text  # any warning emitted

    def test_no_valid_orfs_results_in_empty(self, caplog):
        pm = self._make_pm()
        with caplog.at_level(logging.WARNING):
            pm.filter_proteome(["X", "Y"])
        assert len(pm) == 0


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_roundtrip_preserves_sequences(self, tmp_path, dummy_fasta_path):
        pm = ProteomeManager(str(dummy_fasta_path))
        pkl = tmp_path / "pm.pkl"
        pm.save(str(pkl))
        pm2 = ProteomeManager.load(str(pkl))
        assert pm2.ids == pm.ids
        assert dict(pm2.sequences) == dict(pm.sequences)

    def test_load_wrong_type_raises(self, tmp_path):
        import pickle
        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"not": "a ProteomeManager"}, f)
        with pytest.raises(TypeError):
            ProteomeManager.load(str(bad))


# ---------------------------------------------------------------------------
# filter (returns new ProteomeManager)
# ---------------------------------------------------------------------------

class TestFilter:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"A": "MKTAY", "B": "GVALSK", "CCC": "LLKSDGQVLKAV"}
        return pm

    def test_filter_by_ids(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A", "CCC"])
        assert set(result.ids) == {"A", "CCC"}

    def test_filter_by_min_length(self):
        pm = self._make_pm()
        result = pm.filter(min_length=6)
        assert all(len(result.sequences[i]) >= 6 for i in result.ids)

    def test_filter_by_max_length(self):
        pm = self._make_pm()
        result = pm.filter(max_length=5)
        assert all(len(result.sequences[i]) <= 5 for i in result.ids)

    def test_filter_by_regex(self):
        pm = self._make_pm()
        result = pm.filter(regex=r"^C")
        assert list(result.ids) == ["CCC"]

    def test_original_unchanged(self):
        pm = self._make_pm()
        pm.filter(ids=["A"])
        assert len(pm) == 3

    def test_missing_ids_warns(self, caplog):
        pm = self._make_pm()
        with caplog.at_level(logging.WARNING):
            result = pm.filter(ids=["A", "MISSING"])
        assert "MISSING" in caplog.text
        assert set(result.ids) == {"A"}

    def test_empty_result_warns(self, caplog):
        pm = self._make_pm()
        with caplog.at_level(logging.WARNING):
            result = pm.filter(min_length=9999)
        assert len(result) == 0


class TestSearchPdbSequence:

    def test_returns_hits_dataframe(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "result_set": [
                {"identifier": "4HHB_A", "score": 0.9},
                {"identifier": "1ABC_B", "score": 0.5},
            ]
        }
        with patch("requests.post", return_value=mock_response):
            df = ProteomeManager.search_pdb_sequence("MKTAYIAK", protein_name="P1")
        assert list(df["PDB_code"]) == ["4HHB", "1ABC"]
        assert list(df["PDB_chain"]) == ["A", "B"]
        assert list(df["protein_name"]) == ["P1", "P1"]

    def test_empty_result_set_returns_empty_dataframe(self):
        mock_response = Mock()
        mock_response.json.return_value = {}
        with patch("requests.post", return_value=mock_response):
            df = ProteomeManager.search_pdb_sequence("MKTAYIAK")
        assert df.empty

    def test_network_error_returns_empty_dataframe(self, caplog):
        with patch("requests.post", side_effect=ConnectionError("boom")):
            with caplog.at_level(logging.ERROR):
                df = ProteomeManager.search_pdb_sequence("MKTAYIAK")
        assert df.empty
        assert "PDB Search failed" in caplog.text


class TestScreenProteomeAgainstPdb:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"P1": "MKTAYIAKQR"}
        return pm

    def test_aggregates_hits_with_alignment_columns(self):
        pm = self._make_pm()
        hit_df = pd.DataFrame({
            "protein_name": ["P1"],
            "PDB_ID": ["4HHB_A"],
            "PDB_code": ["4HHB"],
            "PDB_chain": ["A"],
            "score": [0.9],
        })
        mock_mol = Mock()
        mock_mol.sequence.return_value = {"A": "MKTAYIAKQR"}
        with patch.object(ProteomeManager, "search_pdb_sequence", return_value=hit_df), \
             patch("virus_interactome.proteome_manager.Molecule", return_value=mock_mol):
            result = pm.screen_proteome_against_pdb()
        assert not result.empty
        assert {"alignment_score", "coverage", "identity", "gaps"} <= set(result.columns)
        assert list(result["protein_name"]) == ["P1"]

    def test_no_hits_returns_empty_dataframe(self, caplog):
        pm = self._make_pm()
        with patch.object(ProteomeManager, "search_pdb_sequence", return_value=pd.DataFrame()):
            with caplog.at_level(logging.WARNING):
                result = pm.screen_proteome_against_pdb()
        assert result.empty
        assert "No PDB matches" in caplog.text

    def test_score_cutoff_filters_low_scoring_hits(self):
        pm = self._make_pm()
        hit_df = pd.DataFrame({
            "protein_name": ["P1", "P1"],
            "PDB_ID": ["4HHB_A", "1ABC_B"],
            "PDB_code": ["4HHB", "1ABC"],
            "PDB_chain": ["A", "B"],
            "score": [0.9, 0.05],
        })
        mock_mol = Mock()
        mock_mol.sequence.return_value = {"A": "MKTAYIAKQR", "B": "GVALSKMNQT"}
        with patch.object(ProteomeManager, "search_pdb_sequence", return_value=hit_df), \
             patch("virus_interactome.proteome_manager.Molecule", return_value=mock_mol):
            result = pm.screen_proteome_against_pdb(score_cutoff=0.15)
        assert list(result["PDB_code"]) == ["4HHB"]


class TestDescribeOrf:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"P1": "MKTAYIAKQRSTVWYGHILMNPQ", "P2": "GVALSKMNQTYWFHDE"}
        pm.identity_table = pd.DataFrame({
            "ORF1": ["P1"],
            "ORF2": ["P2"],
            "Identity": [0.42],
        })
        return pm

    def test_unknown_orf_warns(self, capsys):
        pm = self._make_pm()
        pm.describe_orf("MISSING")
        out = capsys.readouterr().out
        assert "not found in proteome" in out

    def test_prints_sequence_section(self, capsys):
        pm = self._make_pm()
        pm.describe_orf("P1")
        out = capsys.readouterr().out
        assert "[SEQUENCE]" in out
        assert "Length      : 23 aa" in out

    def test_prints_no_model_data_message(self, capsys):
        pm = self._make_pm()
        pm.describe_orf("P1")
        out = capsys.readouterr().out
        assert "No model data" in out

    def test_prints_model_section_with_data(self, capsys):
        pm = self._make_pm()
        pm.model_info_by_model = pd.DataFrame({
            "id": ["P1", "P1"],
            "pTM": [0.8, 0.82],
            "mean_plddt": [85.0, 86.0],
        })
        pm._model_engine = "af3"
        pm.describe_orf("P1")
        out = capsys.readouterr().out
        assert "Engine      : af3" in out
        assert "N models    : 2" in out

    def test_accepts_list_of_orf_ids(self, capsys):
        pm = self._make_pm()
        pm.describe_orf(["P1", "P2"])
        out = capsys.readouterr().out
        assert "ORF: P1" in out
        assert "ORF: P2" in out

    def test_with_interactome_df_skip_via_n(self, capsys, monkeypatch):
        pm = self._make_pm()
        interactome_df = pd.DataFrame({
            "PPI": ["P1__P2"],
            "ipTM": [0.7],
            "Tier": ["Tier 1"],
        })
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")
        pm.describe_orf("P1", interactome_df=interactome_df)
        out = capsys.readouterr().out
        assert "[PPIs]" in out
        assert "Total PPIs  : 1" in out

    def test_with_interactome_df_custom_threshold(self, capsys, monkeypatch):
        pm = self._make_pm()
        interactome_df = pd.DataFrame({
            "PPI": ["P1__P2"],
            "ipTM": [0.7],
            "Tier": ["Tier 1"],
        })
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "0.5")
        pm.describe_orf("P1", interactome_df=interactome_df)
        out = capsys.readouterr().out
        assert "P1__P2" in out

    def test_with_foldseek_df_hits(self, capsys):
        pm = self._make_pm()
        foldseek_df = pd.DataFrame({
            "protein_id": ["P1"],
            "rank": [1],
            "target": ["4HHB_A"],
            "fident": [0.5],
        })
        pm.describe_orf("P1", foldseek_df=foldseek_df)
        out = capsys.readouterr().out
        assert "[FOLDSEEK]" in out
        assert "4HHB_A" in out

    def test_with_foldseek_df_no_hits(self, capsys):
        pm = self._make_pm()
        foldseek_df = pd.DataFrame({"protein_id": ["OTHER"], "rank": [1]})
        pm.describe_orf("P1", foldseek_df=foldseek_df)
        out = capsys.readouterr().out
        assert "No hits found." in out


class TestViewErrors:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"P1": "MKTAYIAKQR"}
        return pm

    def test_no_orf_models_attribute_raises(self):
        pm = self._make_pm()
        with pytest.raises(ValueError, match="Run load_model_info"):
            pm.view("P1")

    def test_orf_id_not_in_orf_models_raises(self):
        pm = self._make_pm()
        pm.orf_models = {"OTHER": ["/path/to/model.cif"]}
        with pytest.raises(ValueError, match="Run load_model_info"):
            pm.view("P1")

    def test_empty_model_list_raises(self):
        pm = self._make_pm()
        pm.orf_models = {"P1": []}
        with pytest.raises(ValueError, match="is empty"):
            pm.view("P1")


class TestFilterWithDerivedData:

    def _make_pm(self):
        pm = ProteomeManager()
        pm.sequences = {"A": "MKTAY", "B": "GVALSK", "CCC": "LLKSDGQVLKAV"}
        pm.identity_table = pd.DataFrame({
            "ORF1": ["A", "A", "B"],
            "ORF2": ["B", "CCC", "CCC"],
            "Identity": [0.5, 0.3, 0.6],
        })
        pm.high_similarity_pairs = [("A", "B", 0.5), ("B", "CCC", 0.6)]
        pm.model_info_by_orf = pd.DataFrame({"id": ["A", "B", "CCC"], "pTM": [0.7, 0.8, 0.6]})
        pm.model_info_by_model = pd.DataFrame({"id": ["A", "A", "B", "CCC"], "pTM": [0.7, 0.71, 0.8, 0.6]})
        pm.orf_models = {"A": ["a1.cif"], "B": ["b1.cif"], "CCC": ["c1.cif"]}
        pm.sequence_properties = pd.DataFrame({"mw": [500, 600, 1200]}, index=["A", "B", "CCC"])
        pm.invalid_sequences = {"X": "bad"}
        return pm

    def test_identity_table_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A", "B"])
        assert set(result.identity_table["ORF1"]) | set(result.identity_table["ORF2"]) <= {"A", "B"}
        assert len(result.identity_table) == 1

    def test_high_similarity_pairs_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A", "B"])
        assert result.high_similarity_pairs == [("A", "B", 0.5)]

    def test_model_info_by_orf_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A"])
        assert list(result.model_info_by_orf["id"]) == ["A"]

    def test_model_info_by_model_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A"])
        assert set(result.model_info_by_model["id"]) == {"A"}

    def test_orf_models_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A", "B"])
        assert set(result.orf_models.keys()) == {"A", "B"}

    def test_sequence_properties_subset(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A"])
        assert list(result.sequence_properties.index) == ["A"]

    def test_invalid_sequences_carried_over(self):
        pm = self._make_pm()
        result = pm.filter(ids=["A"])
        assert result.invalid_sequences == {"X": "bad"}
