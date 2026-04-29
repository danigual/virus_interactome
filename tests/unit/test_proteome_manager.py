import pytest
import logging
from pathlib import Path
from unittest.mock import patch
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


# --- Identity matrix tests ---

def test_compute_identity_matrix_empty_proteome():
    pm = ProteomeManager()
    with pytest.raises(ValueError, match="Proteome is empty"):
        pm.compute_identity_matrix()


def test_compute_identity_matrix_single_protein():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "MKTAYIAKQR"}
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == pytest.approx(1.0)


def test_compute_identity_matrix_two_proteins():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT"}
    df = pm.compute_identity_matrix(n_jobs=1)
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == pytest.approx(1.0)
    assert df.iloc[1, 1] == pytest.approx(1.0)
    assert df.iloc[0, 1] == pytest.approx(df.iloc[1, 0])
    assert 0.75 <= df.iloc[0, 1] <= 1.0


@pytest.mark.slow
def test_compute_identity_matrix_multiprocessing():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity_matrix(n_jobs=2)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == pytest.approx(1.0) for i in range(3))
    assert df.loc["Protein1", "Protein3"] == pytest.approx(0.0)
    assert df.loc["Protein3", "Protein1"] == pytest.approx(0.0)
    assert df.loc["Protein1", "Protein2"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein1"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein3"] == pytest.approx(0.25)
    assert df.loc["Protein3", "Protein2"] == pytest.approx(0.25)


@pytest.mark.slow
def test_compute_identity_matrix_with_similarity_threshold():
    pm = ProteomeManager()
    pm.sequences = {"Protein1": "AAAA", "Protein2": "AAAT", "Protein3": "TTTT"}
    df = pm.compute_identity_matrix(n_jobs=2, similarity_threshold=0.7)
    assert df.shape == (3, 3)
    assert all(df.iloc[i, i] == pytest.approx(1.0) for i in range(3))
    assert df.loc["Protein1", "Protein3"] == pytest.approx(0.0)
    assert df.loc["Protein1", "Protein2"] == pytest.approx(0.75)
    assert df.loc["Protein2", "Protein3"] == pytest.approx(0.25)
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
    seq = pm.get_sequence("Protein2")
    assert seq == "GVALSKGEEAVRLFK"


def test_get_sequence_not_found(dummy_fasta_path):
    pm = ProteomeManager(str(dummy_fasta_path))
    with pytest.raises(KeyError):
        pm.get_sequence("NONEXISTENT_PROTEIN")


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
    ids = pm.get_ids()
    assert isinstance(ids, list)
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
    """Synchronous stand-in for load_model_info_monomer."""
    orf_id = ProteomeManager._get_orf_id_from_path(path, engine)
    return {"ORF": orf_id, "Model_num": None, "ipTM": 0.7, "pTM": 0.8,
            "mean_plddt": 75.0, "mean_pae": 5.0}


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
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert len(df) == 4  # 2 ORFs × 2 models
    assert set(df["ORF"]) == {"E1A", "pVII"}
    assert "skipping" not in caplog.text


def test_load_model_info_missing_structure_warns(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["E1A"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY", "pX": "GVALS"}  # pX has no models

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        pm.load_model_info(str(model_dir), engine="AF3")

    assert "pX" in caplog.text
    assert "sequence loaded but no model files found" in caplog.text


def test_load_model_info_missing_sequence_warns(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["E1A", "pVII"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # pVII has models but no sequence

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        pm.load_model_info(str(model_dir), engine="AF3")

    assert "pVII" in caplog.text
    assert "model files found but no sequence" in caplog.text


def test_load_model_info_no_sequences_processes_all(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["E1A", "pVII"], n_models=2)
    pm = ProteomeManager()  # no FASTA loaded → sequences = {}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.INFO)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert len(df) == 4
    assert "No sequences loaded" in caplog.text


def test_load_model_info_no_valid_files_returns_empty_df(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["pVII"], n_models=1)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # no intersection

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert df.empty
    assert "No valid model files" in caplog.text


def test_load_model_info_case_insensitive_match_resolves(tmp_path, caplog):
    model_dir = _make_model_dir(tmp_path, ["e1a"], n_models=1)  # lowercase dir
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}  # uppercase key

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        caplog.set_level(logging.WARNING)
        df = pm.load_model_info(str(model_dir), engine="AF3")

    assert len(df) == 1
    assert "case mismatch" in caplog.text
    assert df.iloc[0]["ORF"] == "e1a"  # mock returns raw path parent name


def test_load_model_info_custom_file_ext_pdb(tmp_path):
    model_dir = _make_model_dir(tmp_path, ["E1A"], file_ext="pdb", n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        df = pm.load_model_info(str(model_dir), engine="AF3", file_ext="pdb")

    assert len(df) == 2


def test_load_model_info_default_ext_ignores_pdb_files(tmp_path):
    """With default file_ext='cif', .pdb files in the same dir are not picked up."""
    model_dir = _make_model_dir(tmp_path, ["E1A"], file_ext="pdb", n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        df = pm.load_model_info(str(model_dir), engine="AF3")  # file_ext="cif" by default

    assert df.empty


def test_load_model_info_sets_model_info_attributes(tmp_path):
    model_dir = _make_model_dir(tmp_path, ["E1A"], n_models=2)
    pm = ProteomeManager()
    pm.sequences = {"E1A": "MKTAY"}

    with patch("concurrent.futures.ProcessPoolExecutor", _SyncPool), \
         patch.object(ProteomeManager, "load_model_info_monomer", side_effect=_mock_monomer):
        pm.load_model_info(str(model_dir), engine="AF3")

    assert pm.model_info is not None
    assert pm.model_info_extended is not None
    assert "ORF" in pm.model_info_extended.columns
    assert len(pm.model_info_extended) == 1  # 1 ORF, averaged across models
