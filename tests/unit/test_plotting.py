"""Smoke tests for virus_interactome.plotting — verify functions run without error."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from virus_interactome.plotting import (
    plot_paes,
    plot_plddt,
    plot_pae_clusters,
    plot_boxplots,
    plot_iptm_vs_ptm,
    plot_confidence_landscape,
    plot_interactive_landscape,
    plot_network,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pae_matrix():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 25, (20, 20))


@pytest.fixture
def chain_boundaries():
    return {"A": (0, 9), "B": (10, 19)}


@pytest.fixture
def plddts():
    rng = np.random.default_rng(0)
    return rng.uniform(40, 100, 20)


@pytest.fixture
def interactome_df():
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "PPI":        [f"P{i}__Q{i}" for i in range(n)],
        "ORF_A":      [f"P{i}" for i in range(n)],
        "ORF_B":      [f"Q{i}" for i in range(n)],
        "ipSAE_AB":   rng.uniform(0, 1, n),
        "pDockQ2_AB": rng.uniform(0, 1, n),
        "pLDDT_mean": rng.uniform(40, 100, n),
        "msa_depth":  rng.integers(1, 200, n).astype(float),
        "ipTM":       rng.uniform(0, 1, n),
        "pTM":        rng.uniform(0, 1, n),
    })


@pytest.fixture
def network_df():
    return pd.DataFrame({
        "protein":                ["A", "B", "C", "D"],
        "degree":                 [3, 2, 1, 1],
        "weighted_degree":        [1.2, 0.8, 0.3, 0.3],
        "betweenness_centrality": [0.5, 0.3, 0.1, 0.0],
        "closeness_centrality":   [0.6, 0.5, 0.3, 0.3],
        "eigenvector_centrality": [0.7, 0.5, 0.2, 0.2],
        "clustering_coefficient": [0.3, 0.2, 0.0, 0.0],
        "is_hub":                 [True, False, False, False],
        "is_bottleneck":          [False, True, False, False],
    })


# ── plot_paes ─────────────────────────────────────────────────────────────────

class TestPlotPaes:
    def test_saves_file(self, tmp_path, pae_matrix, chain_boundaries):
        out = tmp_path / "pae.png"
        plot_paes(pae_matrix, chain_boundaries=chain_boundaries, save_name=str(out))
        assert out.exists()

    def test_no_chain_boundaries(self, tmp_path, pae_matrix):
        out = tmp_path / "pae.png"
        plot_paes(pae_matrix, save_name=str(out))
        assert out.exists()

    def test_with_title(self, tmp_path, pae_matrix):
        out = tmp_path / "pae.png"
        plot_paes(pae_matrix, title="Test PAE", save_name=str(out))
        assert out.exists()


# ── plot_plddt ────────────────────────────────────────────────────────────────

class TestPlotPlddt:
    def test_saves_file(self, tmp_path, plddts, chain_boundaries):
        out = tmp_path / "plddt.png"
        plot_plddt(plddts, chain_boundaries=chain_boundaries, save_name=str(out))
        assert out.exists()

    def test_no_chain_boundaries(self, tmp_path, plddts):
        out = tmp_path / "plddt.png"
        plot_plddt(plddts, save_name=str(out))
        assert out.exists()


# ── plot_pae_clusters ─────────────────────────────────────────────────────────

class TestPlotPaeClusters:
    def test_saves_file(self, tmp_path, pae_matrix):
        rng = np.random.default_rng(0)
        coords = rng.integers(0, 20, (15, 2))
        labels = rng.integers(-1, 3, 15)
        out = tmp_path / "clusters.png"
        plot_pae_clusters(pae_matrix, coords, labels, save_name=str(out))
        assert out.exists()

    def test_empty_coords(self, tmp_path, pae_matrix):
        out = tmp_path / "clusters_empty.png"
        plot_pae_clusters(pae_matrix, np.empty((0, 2), dtype=int), np.array([]), save_name=str(out))
        assert out.exists()


# ── plot_boxplots ─────────────────────────────────────────────────────────────

class TestPlotBoxplots:
    def test_plddt(self, tmp_path):
        rng = np.random.default_rng(0)
        values = [rng.uniform(50, 100, 20) for _ in range(5)]
        labels = np.array([f"ORF{i}" for i in range(5)])
        plot_boxplots("pLDDT", np.array(values, dtype=object), labels, output_path=str(tmp_path))
        assert (tmp_path / "pLDDT_boxplot.png").exists()

    def test_pae(self, tmp_path):
        rng = np.random.default_rng(0)
        values = [rng.uniform(0, 25, 20) for _ in range(5)]
        labels = np.array([f"ORF{i}" for i in range(5)])
        plot_boxplots("PAE", np.array(values, dtype=object), labels, output_path=str(tmp_path))
        assert (tmp_path / "PAE_boxplot.png").exists()


# ── plot_iptm_vs_ptm ──────────────────────────────────────────────────────────

class TestPlotIptmVsPtm:
    def test_saves_file(self, tmp_path, interactome_df):
        plot_iptm_vs_ptm(interactome_df, output_path=str(tmp_path))
        assert (tmp_path / "_scatterplot.png").exists()


# ── plot_confidence_landscape ─────────────────────────────────────────────────

class TestPlotConfidenceLandscape:
    def test_saves_file(self, tmp_path, interactome_df):
        out = tmp_path / "landscape.png"
        plot_confidence_landscape(interactome_df, output_path=str(out))
        assert out.exists()

    def test_missing_columns_logs_error(self, tmp_path):
        df = pd.DataFrame({"PPI": ["A__B"], "ipTM": [0.5]})
        plot_confidence_landscape(df, output_path=str(tmp_path / "out.png"))

    def test_no_msa_depth(self, tmp_path, interactome_df):
        df = interactome_df.drop(columns=["msa_depth"])
        out = tmp_path / "landscape.png"
        plot_confidence_landscape(df, output_path=str(out))
        assert out.exists()

    def test_no_plddt(self, tmp_path, interactome_df):
        df = interactome_df.drop(columns=["pLDDT_mean"])
        out = tmp_path / "landscape.png"
        plot_confidence_landscape(df, output_path=str(out))
        assert out.exists()


# ── plot_interactive_landscape ────────────────────────────────────────────────

class TestPlotInteractiveLandscape:
    def test_saves_html(self, tmp_path, interactome_df):
        out = tmp_path / "landscape.html"
        plot_interactive_landscape(interactome_df, output_path=str(out))
        assert out.exists()

    def test_missing_columns_logs_error(self, tmp_path):
        df = pd.DataFrame({"PPI": ["A__B"], "ipTM": [0.5]})
        plot_interactive_landscape(df, output_path=str(tmp_path / "out.html"))


# ── plot_network ──────────────────────────────────────────────────────────────

class TestPlotNetwork:
    def _make_graph(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edge("A", "B", weight=0.8)
        G.add_edge("A", "C", weight=0.5)
        G.add_edge("A", "D", weight=0.3)
        G.add_edge("B", "C", weight=0.4)
        return G

    def test_saves_file(self, tmp_path, network_df):
        G = self._make_graph()
        out = tmp_path / "network.png"
        plot_network(G, network_df, output_path=str(out))
        assert out.exists()

    def test_empty_df_no_crash(self, tmp_path):
        import networkx as nx
        G = nx.Graph()
        plot_network(G, pd.DataFrame(), output_path=str(tmp_path / "net.png"))

    def test_color_by_degree(self, tmp_path, network_df):
        G = self._make_graph()
        out = tmp_path / "network.png"
        plot_network(G, network_df, color_by="degree", size_by="weighted_degree", output_path=str(out))
        assert out.exists()
