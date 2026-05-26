import pytest
import pandas as pd
from pathlib import Path
from virus_interactome.databases import DatabaseClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str, suffix: str = ".csv") -> Path:
    p = tmp_path / f"db{suffix}"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------

class TestDatabaseClientFromFile:

    def test_loads_basic_csv(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b\nA,B\nC,D\n")
        client = DatabaseClient.from_file(p, col_a="prot_a", col_b="prot_b")
        assert len(client) == 2

    def test_tsv_auto_detected(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a\tprot_b\nA\tB\n", suffix=".tsv")
        client = DatabaseClient.from_file(p, col_a="prot_a", col_b="prot_b")
        assert len(client) == 1

    def test_default_col_names(self, tmp_path):
        p = _write_csv(tmp_path, "protein_A,protein_B\nA,B\n")
        client = DatabaseClient.from_file(p)
        assert frozenset({"A", "B"}) in client.ppis

    def test_filter_keeps_matching_rows(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b,conf\nA,B,high\nC,D,low\n")
        client = DatabaseClient.from_file(
            p, col_a="prot_a", col_b="prot_b", filters={"conf": ["high"]}
        )
        assert len(client) == 1
        assert frozenset({"A", "B"}) in client.ppis

    def test_filter_all_rows_removed(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b,conf\nA,B,low\n")
        client = DatabaseClient.from_file(
            p, col_a="prot_a", col_b="prot_b", filters={"conf": ["high"]}
        )
        assert len(client) == 0

    def test_extra_cols_retained_in_metadata(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b,conf\nA,B,high\n")
        client = DatabaseClient.from_file(
            p, col_a="prot_a", col_b="prot_b", extra_cols=["conf"]
        )
        assert "conf" in client.metadata.columns

    def test_missing_extra_cols_skipped(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b\nA,B\n")
        client = DatabaseClient.from_file(
            p, col_a="prot_a", col_b="prot_b", extra_cols=["nonexistent"]
        )
        assert "nonexistent" not in client.metadata.columns

    def test_missing_col_a_raises(self, tmp_path):
        p = _write_csv(tmp_path, "x,prot_b\nA,B\n")
        with pytest.raises(ValueError, match="prot_a"):
            DatabaseClient.from_file(p, col_a="prot_a", col_b="prot_b")

    def test_missing_filter_col_raises(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b\nA,B\n")
        with pytest.raises(ValueError, match="conf"):
            DatabaseClient.from_file(
                p, col_a="prot_a", col_b="prot_b", filters={"conf": ["high"]}
            )


# ---------------------------------------------------------------------------
# Interface: __contains__, __len__, proteins, __repr__
# ---------------------------------------------------------------------------

class TestDatabaseClientInterface:

    @pytest.fixture
    def client(self, tmp_path):
        p = _write_csv(tmp_path, "prot_a,prot_b\nA,B\nB,C\n")
        return DatabaseClient.from_file(p, col_a="prot_a", col_b="prot_b")

    def test_len(self, client):
        assert len(client) == 2

    def test_contains_tuple_ordered(self, client):
        assert ("A", "B") in client

    def test_contains_tuple_reversed(self, client):
        assert ("B", "A") in client

    def test_contains_frozenset(self, client):
        assert frozenset({"A", "B"}) in client

    def test_not_contains(self, client):
        assert ("A", "C") not in client

    def test_proteins_covers_all_ids(self, client):
        assert client.proteins == {"A", "B", "C"}

    def test_repr(self, client):
        assert "2" in repr(client)
        assert "DatabaseClient" in repr(client)
