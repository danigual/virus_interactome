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
    # import pdb;pdb.set_trace()


    # Use user's load_yaml
    parsed_yaml = load_yaml(str(save_file))
    assert parsed_yaml["version"] == 1
    assert isinstance(parsed_yaml["sequences"], list) and len(parsed_yaml["sequences"]) == 2

    # Entry shape & content
    e0 = parsed_yaml["sequences"][0]
    e1 = parsed_yaml["sequences"][1]
    assert e0["protein"]["id"] == "A"
    assert e0["protein"]["sequence"] == "ACDE"
    assert "multiple_chains" not in e0

    assert e1["protein"]["id"] == "B"
    assert e1["protein"]["sequence"] == "GGGG"
    assert e1["protein"]["multiple_chains"] == "C,D"

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
