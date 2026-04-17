import pytest
from virus_interactome.interactome import InteractomeWriter
import json, yaml
import warnings

from virus_interactome.utils import load_boltz_input, load_json, load_yaml

AF3_THRESH = 5000
BOLTZ_THRESH = 1600

def test_af3_rejects_empty_list():
    with pytest.raises(ValueError):
        InteractomeWriter.get_af3_input([])

def test_boltz_rejects_empty_list():
    with pytest.raises(ValueError):
        InteractomeWriter.get_boltz2_input([])

def test_rejects_count_less_than_one():
    bad = [("A", "ACD", 0)]
    with pytest.raises(ValueError):
        InteractomeWriter.get_af3_input(bad)
    with pytest.raises(ValueError):
        InteractomeWriter.get_boltz2_input(bad)

def test_rejects_invalid_sequence_characters():
    bad = [("A", "ACD*E", 1)]  # '*' invalid
    with pytest.raises(ValueError):
        InteractomeWriter.get_af3_input(bad)
    with pytest.raises(ValueError):
        InteractomeWriter.get_boltz2_input(bad)

def test_af3_writes_json(tmp_path):
    seqs = [("A", "ACDE", 1), ("B", "GGGG", 2)]
    path = tmp_path / "job_af3.json"
    out = InteractomeWriter.get_af3_input(seqs, job_name="demo", save_path=str(path))
    assert path.exists()
    with open(path, "r", encoding="utf-8") as fh:
        parsed = json.load(fh)
    # basic shape checks
    assert parsed["name"] == "demo"
    assert len(parsed["sequences"]) == 2

def test_af3_threshold_warning():
    # Build a payload that exceeds 5000 residues
    long_seq = "A" * (AF3_THRESH // 2 + 10)  # > 2500
    seqs = [("A", long_seq, 2)]  # total > 5000
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = InteractomeWriter.get_af3_input(seqs)
        assert any("exceed" in str(x.message).lower() for x in w), "No warning emitted"

def test_boltz_threshold_warning():
    long_seq = "A" * (BOLTZ_THRESH // 2 + 10)
    seqs = [("A", long_seq, 2)]  # total > 1600
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = InteractomeWriter.get_boltz2_input(seqs)
        assert any("exceed" in str(x.message).lower() for x in w)

def test_af3_writes_and_loads_json(tmp_path):
    seqs = [("A", "ACDE", 1), ("B", "GGGG", 2)]
    save_file = tmp_path / "case_af3.json"

    out = InteractomeWriter.get_af3_input(seqs, job_name="demo_af3", save_path=str(save_file))
    assert save_file.exists(), "AF3 JSON file was not written."

    # Use user's load_json
    parsed = load_json(str(save_file))

    # Basic shape checks
    assert parsed["name"] == "demo_af3"
    assert isinstance(parsed["sequences"], list) and len(parsed["sequences"]) == 2

    # Content checks
    s0 = parsed["sequences"][0]["proteinChain"]
    s1 = parsed["sequences"][1]["proteinChain"]
    assert s0["sequence"] == "ACDE" and s0["count"] == 1
    assert s1["sequence"] == "GGGG" and s1["count"] == 2

    # Warnings mirrored
    assert parsed.get("_warnings", []) == out.get("_warnings", [])


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_boltz_writes_and_loads_yaml(tmp_path):
    # Second entry has count=3, so multiple_chains should exist
    seqs = [("A", "ACDE", 1), ("B", "GGGG", 3)]
    save_file = tmp_path / "case_boltz.yaml"

    out = InteractomeWriter.get_boltz2_input(seqs, save_path=str(save_file))
    assert save_file.exists(), "Boltz YAML file was not written."
    # Use user's load_yaml
    parsed_yaml = load_yaml(str(save_file))
    assert parsed_yaml["version"] == 1
    assert isinstance(parsed_yaml["sequences"], list) and len(parsed_yaml["sequences"]) == 2

    # Entry shape & content
    e0 = parsed_yaml["sequences"][0]
    e1 = parsed_yaml["sequences"][1]
    assert e0["protein"]["id"] == ["A"]
    assert e0["protein"]["sequence"] == "ACDE"
    assert "multiple_chains" not in e0

    assert e1["protein"]["id"] == ["B", "C", "D"]
    assert e1["protein"]["sequence"] == "GGGG"


    # Now use user's load_boltz_input which maps to AF3-like schema (list of one job)
    mapped = load_boltz_input(str(save_file), job_name="mapped_boltz")
    assert isinstance(mapped, list) and len(mapped) == 1
    job = mapped[0]
    assert job["name"] == "mapped_boltz"
    assert len(job["sequences"]) == 2
    assert job["sequences"][0]["proteinChain"]["sequence"] == "ACDE"
    assert job["sequences"][0]["proteinChain"]["count"] == 1
    assert job["sequences"][1]["proteinChain"]["sequence"] == "GGGG"
    assert job["sequences"][1]["proteinChain"]["count"] == 3


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestWriterInit:
    def test_init_with_fasta(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        assert w.mode == "intra"
        assert w.proteome_a is not None
        assert w.proteome_b is None

    def test_init_inter_mode(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path), str(dummy_fasta_path))
        assert w.mode == "inter"
        assert w.proteome_b is not None

    def test_init_with_proteome_manager(self, dummy_fasta_path):
        from virus_interactome.proteome_manager import ProteomeManager
        pm = ProteomeManager(str(dummy_fasta_path))
        w = InteractomeWriter(pm)
        assert w.proteome_a is pm

    def test_init_invalid_type_raises(self):
        with pytest.raises(ValueError, match="proteome_a"):
            InteractomeWriter(12345)

    def test_init_inter_with_proteome_manager(self, dummy_fasta_path):
        from virus_interactome.proteome_manager import ProteomeManager
        pm = ProteomeManager(str(dummy_fasta_path))
        w = InteractomeWriter(str(dummy_fasta_path), pm)
        assert w.mode == "inter"
        assert w.proteome_b is pm

    def test_init_invalid_proteome_b_raises(self, dummy_fasta_path):
        with pytest.raises(ValueError, match="proteome_b"):
            InteractomeWriter(str(dummy_fasta_path), 12345)


# ---------------------------------------------------------------------------
# generate_intra_pairs
# ---------------------------------------------------------------------------

class TestGenerateIntraPairs:
    def test_basic(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        pairs = list(w.generate_intra_pairs())
        # 4 proteins → C(4,2) = 6 pairs
        assert len(pairs) == 6
        # Each pair is a tuple of 2 distinct IDs
        for a, b in pairs:
            assert a != b

    def test_no_duplicates(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        pairs = list(w.generate_intra_pairs())
        pair_set = set(pairs)
        assert len(pair_set) == len(pairs)
        # No reverse pair should exist
        for a, b in pairs:
            assert (b, a) not in pair_set


# ---------------------------------------------------------------------------
# generate_inter_pairs
# ---------------------------------------------------------------------------

class TestGenerateInterPairs:
    def test_basic(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path), str(dummy_fasta_path))
        pairs = list(w.generate_inter_pairs())
        # 4 × 4 = 16 (cartesian product, includes self-pairs)
        assert len(pairs) == 16

    def test_intra_mode_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="inter"):
            list(w.generate_inter_pairs())


# ---------------------------------------------------------------------------
# generate_homo_mers
# ---------------------------------------------------------------------------

class TestGenerateHomomers:
    def test_defaults(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        homos = list(w.generate_homo_mers())
        # 4 proteins × (6-2+1) = 4 × 5 = 20
        assert len(homos) == 20
        for pid, copies in homos:
            assert 2 <= copies <= 6

    def test_custom_range(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        homos = list(w.generate_homo_mers(nmin=2, nmax=3))
        # 4 proteins × 2 = 8
        assert len(homos) == 8

    def test_nmin_less_than_2_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="nmin"):
            list(w.generate_homo_mers(nmin=1))

    def test_nmax_less_than_nmin_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="nmax"):
            list(w.generate_homo_mers(nmin=4, nmax=2))

    def test_inter_mode_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path), str(dummy_fasta_path))
        with pytest.raises(ValueError, match="intra"):
            list(w.generate_homo_mers())


# ---------------------------------------------------------------------------
# generate_single_run
# ---------------------------------------------------------------------------

class TestGenerateSingleRun:
    def test_source_a(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        singles = list(w.generate_single_run(source="a"))
        assert len(singles) == 4
        for pid, seq, cnt in singles:
            assert cnt == 1
            assert len(seq) > 0

    def test_source_b_intra_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="inter"):
            list(w.generate_single_run(source="b"))

    def test_source_both_inter(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path), str(dummy_fasta_path))
        singles = list(w.generate_single_run(source="both"))
        # 4 from A + 4 from B = 8
        assert len(singles) == 8

    def test_invalid_source_raises(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="source"):
            list(w.generate_single_run(source="c"))

    def test_filter_ids_a(self, dummy_fasta_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        singles = list(w.generate_single_run(source="a", ids_a=["Protein1_isoformB"]))
        assert len(singles) == 1
        assert singles[0][0] == "Protein1_isoformB"


# ---------------------------------------------------------------------------
# write_interactome_jobs — additional coverage
# ---------------------------------------------------------------------------

class TestWriteInteractomeJobsExtended:
    def test_inter_pairs_mode(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path), str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="inter_pairs")
        # 4 × 4 = 16 inter pairs
        assert len(metas) == 16

    def test_intra_pairs_ids_filter(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        subset = ["Protein1_isoformB", "Protein2"]
        metas = w.write_interactome_jobs(
            "af3", str(tmp_path), mode="intra_pairs", ids_a=subset,
        )
        # C(2,2) = 1
        assert len(metas) == 1

    def test_boltz_homomers(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("boltz2", str(tmp_path), mode="homomers", nmin=2, nmax=2)
        assert len(metas) == 4  # 4 proteins × 1 stoichiometry

    def test_skip_over_threshold(self, dummy_fasta_path, tmp_path):
        """Jobs exceeding residue threshold get warning in metadata."""
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs(
            "af3", str(tmp_path), mode="intra_pairs",
            af3_threshold=10, skip_over_threshold=True,
        )
        # All pairs with total_residues > 10 should have warning
        for m in metas:
            if m["total_residues"] > 10:
                assert "Skipped" in m["warnings"] or "exceed" in m["warnings"]

    def test_generate_intra_pairs_no_proteome_raises(self):
        """generate_intra_pairs raises when proteome_a is None."""
        w = InteractomeWriter.__new__(InteractomeWriter)
        w.proteome_a = None
        w.proteome_b = None
        w.mode = "intra"
        with pytest.raises(ValueError, match="proteome_a"):
            list(w.generate_intra_pairs())

    def test_intra_pairs_id_swap(self, dummy_fasta_path, tmp_path):
        """Intra pairs with reversed alphabetical order get idA/idB swapped."""
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="intra_pairs")
        for m in metas:
            # For intra mode, idA should always be <= idB alphabetically
            assert m["idA"] <= m["idB"], f"Expected {m['idA']} <= {m['idB']}"

    def test_counts_map(self, dummy_fasta_path, tmp_path):
        """write_interactome_jobs with counts_map sets correct stoichiometry."""
        w = InteractomeWriter(str(dummy_fasta_path))
        counts = {"Protein1_isoformB": 2, "Protein2": 3}
        metas = w.write_interactome_jobs(
            "af3", str(tmp_path), mode="intra_pairs", counts_map=counts,
        )
        for m in metas:
            if m["idA"] == "Protein1_isoformB":
                assert m["countA"] == 2
            if m["idB"] == "Protein2":
                assert m["countB"] == 3

    def test_intra_pairs_swap_actually_triggers(self, tmp_path):
        """Lines 273-274: swap branch activates when FASTA is in reverse-alpha order."""
        fasta = tmp_path / "reverse.fasta"
        fasta.write_text(">ZProtein\nMKTAY\n>AProtein\nGVALS\n")
        out = tmp_path / "out"
        w = InteractomeWriter(str(fasta))
        metas = w.write_interactome_jobs("af3", str(out), mode="intra_pairs")
        assert len(metas) == 1
        # Swap must have occurred: idA="AProtein" < idB="ZProtein"
        assert metas[0]["idA"] == "AProtein"
        assert metas[0]["idB"] == "ZProtein"

    def test_colabfold_engine_intra_pairs(self, dummy_fasta_path, tmp_path):
        """Lines 393-394, 440-442: ColabFold engine writes one .fasta per pair."""
        from pathlib import Path
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("colabfold", str(tmp_path), mode="intra_pairs")
        assert len(metas) == 6  # C(4,2)
        fasta_files = list(tmp_path.glob("*.fasta"))
        assert len(fasta_files) == 6
        # Verify content: file_path points to existing file with correct format
        first_file = Path(metas[0]["file_path"])
        content = first_file.read_text()
        assert content.startswith(f">{metas[0]['name']}\n")
        assert ":" in content.split("\n")[1]

    def test_boltz2_chain_id_double_letters(self):
        """Lines 625-627: chain_id_generator yields AA, AB... when copies exceed 26."""
        seqs = [("ProtA", "ACDE", 27)]
        result = InteractomeWriter.get_boltz2_input(seqs)
        ids = result["sequences"][0]["protein"]["id"]
        assert len(ids) == 27
        assert ids[25] == "Z"   # 26th chain (0-indexed)
        assert ids[26] == "AA"  # 27th chain — double-letter branch


# ---------------------------------------------------------------------------
# write_interactome_jobs
# ---------------------------------------------------------------------------

class TestWriteInteractomeJobs:
    def test_af3_intra_pairs(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="intra_pairs")
        assert len(metas) == 6  # C(4,2)
        # Check index.csv was created
        assert (tmp_path / "index.csv").exists()
        # Check JSON files were created
        json_files = list(tmp_path.glob("*.json"))
        # index.csv is not JSON, so only job files
        assert len(json_files) == 6

    def test_boltz2_intra_pairs(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("boltz2", str(tmp_path), mode="intra_pairs")
        assert len(metas) == 6
        yaml_files = list(tmp_path.glob("*.yaml"))
        assert len(yaml_files) == 6

    def test_homomers_mode(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="homomers", nmin=2, nmax=3)
        # 4 proteins × 2 stoichiometries = 8
        assert len(metas) == 8

    def test_single_mode(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="single")
        assert len(metas) == 4

    def test_invalid_engine_raises(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="Unsupported engine"):
            w.write_interactome_jobs("rosetta", str(tmp_path))

    def test_invalid_mode_raises(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        with pytest.raises(ValueError, match="Unknown mode"):
            w.write_interactome_jobs("af3", str(tmp_path), mode="fantasy")

    def test_meta_fields(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="single")
        required_keys = {"engine", "mode", "name", "idA", "idB", "countA", "countB", "total_residues", "warnings", "file_path"}
        for m in metas:
            assert required_keys.issubset(set(m.keys()))

    def test_single_mode_idb_empty(self, dummy_fasta_path, tmp_path):
        """mode='single' produces monomer jobs: idB must be empty, countB must be empty."""
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_interactome_jobs("af3", str(tmp_path), mode="single")
        for m in metas:
            assert m["idB"] == ""
            assert m["countB"] == ""


# ---------------------------------------------------------------------------
# _build_colabfold_seq_str (static method)
# ---------------------------------------------------------------------------

class TestBuildColabfoldSeqStr:
    def test_single_chain(self):
        seq_list = [("A", "MKTAY", 1)]
        result = InteractomeWriter._build_colabfold_seq_str(seq_list)
        assert result == "MKTAY"

    def test_two_chains(self):
        seq_list = [("A", "AAAA", 1), ("B", "GGGG", 1)]
        result = InteractomeWriter._build_colabfold_seq_str(seq_list)
        assert result == "AAAA:GGGG"

    def test_stoichiometry_expansion(self):
        seq_list = [("A", "AAAA", 2), ("B", "GGGG", 1)]
        result = InteractomeWriter._build_colabfold_seq_str(seq_list)
        assert result == "AAAA:AAAA:GGGG"

    def test_homomer_stoichiometry(self):
        seq_list = [("A", "MKVL", 3)]
        result = InteractomeWriter._build_colabfold_seq_str(seq_list)
        assert result == "MKVL:MKVL:MKVL"


# ---------------------------------------------------------------------------
# write_colabfold_fastas
# ---------------------------------------------------------------------------

class TestWriteColabfoldFastas:
    def test_intra_pairs_creates_fastas(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_fastas(str(tmp_path), mode="intra_pairs")
        assert len(metas) == 6  # C(4,2)
        fasta_files = list(tmp_path.glob("*.fasta"))
        assert len(fasta_files) == 6

    def test_fasta_content_format(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_fastas(str(tmp_path), mode="intra_pairs")
        # Check one FASTA file
        first = metas[0]
        fasta_path = tmp_path / f"{first['name']}.fasta"
        assert fasta_path.exists()
        content = fasta_path.read_text()
        assert content.startswith(f">{first['name']}\n")
        # Sequence line should contain colons for multimer
        seq_line = content.strip().split("\n")[1]
        assert ":" in seq_line

    def test_index_csv_created(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        w.write_colabfold_fastas(str(tmp_path), mode="intra_pairs")
        index = tmp_path / "colabfold_index.csv"
        assert index.exists()
        import pandas as pd
        df = pd.read_csv(index)
        assert len(df) == 6
        assert set(df.columns) >= {"name", "idA", "idB", "total_residues", "file_path"}

    def test_homomers_mode(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_fastas(str(tmp_path), mode="homomers", nmin=2, nmax=3)
        assert len(metas) == 8  # 4 proteins × 2 stoichiometries

    def test_meta_fields(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_fastas(str(tmp_path), mode="single")
        for m in metas:
            assert {"name", "mode", "idA", "total_residues", "file_path"}.issubset(m.keys())


# ---------------------------------------------------------------------------
# write_colabfold_csv
# ---------------------------------------------------------------------------

class TestWriteColabfoldCsv:
    def test_intra_pairs_creates_csv(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_csv(str(tmp_path), mode="intra_pairs")
        assert len(metas) == 6
        csv_path = tmp_path / "colabfold_input.csv"
        assert csv_path.exists()

    def test_csv_content_format(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        w.write_colabfold_csv(str(tmp_path), mode="intra_pairs")
        import pandas as pd
        df = pd.read_csv(tmp_path / "colabfold_input.csv")
        assert set(df.columns) == {"id", "sequence"}
        assert len(df) == 6
        # Every sequence should contain ':' (multimer format)
        assert df["sequence"].str.contains(":").all()

    def test_index_csv_created(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        w.write_colabfold_csv(str(tmp_path), mode="intra_pairs")
        index = tmp_path / "colabfold_index.csv"
        assert index.exists()
        import pandas as pd
        df = pd.read_csv(index)
        assert len(df) == 6
        assert set(df.columns) >= {"name", "idA", "idB", "total_residues"}

    def test_custom_filenames(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        w.write_colabfold_csv(
            str(tmp_path), mode="intra_pairs",
            csv_name="custom_input.csv", index_name="custom_index.csv",
        )
        assert (tmp_path / "custom_input.csv").exists()
        assert (tmp_path / "custom_index.csv").exists()

    def test_single_mode(self, dummy_fasta_path, tmp_path):
        w = InteractomeWriter(str(dummy_fasta_path))
        metas = w.write_colabfold_csv(str(tmp_path), mode="single")
        assert len(metas) == 4
