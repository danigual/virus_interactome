
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .utils import process_full_data_colabfold

logger = logging.getLogger(__name__)

def plot_paes(
    pae_matrix: np.ndarray,
    chain_boundaries: list = None,
    chain_ids: list = None,
    title: str = None,
    save_name: str = None,
    ax=None
):
    """Render a PAE heatmap (green colormap, 0–25 Å) with optional chain boundary overlays.

    Parameters
    ----------
    pae_matrix : np.ndarray
        Square PAE matrix (N_res × N_res).
    chain_boundaries : dict, optional
        ``{chain_id: (start_idx, end_idx)}`` mapping used to draw boundary lines and axis labels.
    chain_ids : list, optional
        Unused when *chain_boundaries* is provided (labels are derived from its keys).
    title : str, optional
        Plot title.
    save_name : str, optional
        Output file path. If None the plot is shown interactively.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is not None:
        plt.sca(ax)
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the matrix
    im = ax.imshow(pae_matrix, cmap='Greens_r', origin='upper', vmin=0, vmax=25)

    # Draw chain boundaries
    if chain_boundaries is not None:
        chain_ids = []
        midpoints = []
        for chain_id, (start_idx, end_idx) in chain_boundaries.items():
            ax.axhline(end_idx + 0.5, color='black', linewidth=0.75, linestyle="dashed")
            ax.axvline(end_idx + 0.5, color='black', linewidth=0.75, linestyle="dashed")
            chain_ids.append(chain_id)
            midpoints.append((start_idx + end_idx) / 2)

        ax.set_xticks(midpoints)
        ax.set_yticks(midpoints)
        ax.set_xticklabels(chain_ids)
        ax.set_yticklabels(chain_ids)

    # Colorbar
    axins = inset_axes(ax, width="100%", height="2.5%", loc="lower center", borderpad=-5)
    cbar = fig.colorbar(im, cax=axins, orientation="horizontal")
    cbar.set_label("Expected position error (Ångströms)")

    if title is not None:
        ax.set_title(title, loc="center")
    
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_name
    else:
        plt.show()
        plt.close(fig)

    return ax

def plot_plddt(plddts: np.ndarray, 
               chain_boundaries: list = None,
               chain_ids: list = None,
               save_name: str = None, ax=None):
    """Render a per-residue pLDDT confidence plot with AlphaFold colour bands.

    Colour bands follow the AlphaFold convention:
    very high (>90, blue), high (70–90, cyan), low (50–70, orange), very low (<50, yellow).

    Parameters
    ----------
    plddts : np.ndarray
        Per-residue pLDDT values (0–100).
    chain_boundaries : dict, optional
        ``{chain_id: (start_idx, end_idx)}`` mapping used to draw chain-boundary lines.
    chain_ids : list, optional
        Unused when *chain_boundaries* is provided (labels are derived from its keys).
    save_name : str, optional
        Output file path. If None the plot is shown interactively.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """

    if ax is not None:
        plt.sca(ax)
    else:
        fig, ax = plt.subplots(figsize=(20, 5))

    max_length = len(plddts)
    position = [n + 1 for n in range(len(plddts))]

    ax.add_patch(Rectangle((0, 90), max_length, 10, color="#024fcc"))
    ax.add_patch(Rectangle((0, 70), max_length, 20, color="#60c2e8"))
    ax.add_patch(Rectangle((0, 50), max_length, 20, color="#f37842"))
    ax.add_patch(Rectangle((0, 0), max_length, 50, color="#f9d613"))

    ax.plot(
        position,
        plddts,
        color="black",
        linewidth=0.5,
        linestyle="-"
            )
    ax.set_xlabel("Position")
    
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(position) + 10)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    if chain_boundaries is not None:
        chain_ids = []
        midpoints = []
        for chain_id, (start, end) in chain_boundaries.items():
            if end == max_length: break
            ax.axvline(x=end + 1, color="red", linestyle="dashed", linewidth=1)
            chain_ids.append(chain_id)
            midpoints.append((start + end) / 2)

        ax.set_xticks(midpoints)
        ax.set_xticklabels(chain_ids)

    plddt_legend = {
        "Very high (pLDDT > 90)": "#024fcc86",
        "High (90 > pLDDT > 70)": "#60c2e886",
        "Low (70 > pLDDT > 50)": "#f3784286",
        "Very low (pLDDT < 50)": "#f9d61386",
    }

    ax.legend(plddt_legend, title="pLDDT Confidence", prop={'size': 10},
              bbox_to_anchor=(0.85, 1.15),
              frameon=False,
              ncol=4)

    ax.set_ylabel("pLDDT")

    if save_name is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return ax

def plot_pae_clusters(submatrix, low_pae_coords: list, cluster_labels_pae: list, save_name:str = None):

    """
    Visualizes a PAE submatrix and clusters of low PAE values using DBSCAN.

    This function displays a heatmap of a PAE submatrix and overlays clustering results
    on low PAE regions. It is useful for identifying structurally consistent regions
    between chains in protein complexes.

    Parameters
    ----------
    submatrix : np.ndarray
        A 2D array representing a subset of the PAE matrix.
    low_pae_coords : list
        Coordinates of low PAE values used for clustering.
    cluster_labels_pae : list
        Cluster labels assigned to each low PAE coordinate by DBSCAN.
    save_name : str, optional
        If provided, the plot will be saved to this filename. If None, the plot is shown interactively.

    Returns
    -------
    str or None
        The filename where the plot was saved, or None if the plot was displayed.

    Raises
    ------
    ValueError
        If the input matrix or clustering data is malformed.
    """

    #Visualize the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    #Original data heatmap
    ax[0].imshow(submatrix, vmin=0, vmax=35, aspect="auto", cmap="Greens_r")
    ax[0].set_title('PAE Submatrix')
    ax[0].set_xlabel('Chain B Residue Index')
    ax[0].set_ylabel('Chain A Residue Index')


    #DBSCAN clusters
    if len(low_pae_coords)>0:
        scatter = ax[1].scatter(low_pae_coords[:, 1], 
                                low_pae_coords[:, 0], 
                                alpha=0.6, 
                                c=cluster_labels_pae, 
                                cmap='tab10', 
                                s=30, marker="o", edgecolor='none')    
        plt.colorbar(scatter, ax=ax[1], label='Cluster ID')
   
    ax[1].set_title('DBSCAN Clusters of Low PAE')
    ax[1].set_xlabel('Chain B Residue Index')
    ax[1].set_ylabel('Chain A Residue Index')
    ax[1].set_xlim(0, submatrix.shape[1])
    ax[1].set_ylim(submatrix.shape[0], 0)

    if save_name is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return save_name

def plot_boxplots (data_type:str, value_matrix:np.array, orf_labels:np.array, output_path:str = None):
    
    """
    Generates a boxplot visualization for pLDDT or PAE values grouped by ORFs.

    This function creates a boxplot for each ORF based on the provided value matrix.
    It includes reference lines for confidence thresholds depending on the data type
    and saves the resulting plot to the specified output path.

    Parameters
    ----------
    data_type : str
        Type of data to plot ('pLDDT' or 'PAE').
    value_matrix : np.ndarray
        Array of lists, each containing values for an ORF, ordered by mean value.
    orf_labels : np.ndarray
        Array of ORF names corresponding to the value_matrix.
    output_path : str, optional
        Directory path where the plot will be saved.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If data_type is not 'pLDDT' or 'PAE'.
    FileNotFoundError
        If the output path does not exist.
    """
    plt.figure(figsize=(12, 6))
    plt.boxplot(list(value_matrix),
                tick_labels=list(orf_labels))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(data_type)
    plt.title(f"{data_type} / ORF")

    
    if data_type.lower() == "plddt":
        plt.axhline(70, color='blue', linestyle='--', label='Confidence threshold (~70)')
        plt.axhline(90, color='red', linestyle='--', label='Confidence threshold (~90)')
    elif data_type.lower() == "pae":
        plt.axhline(10, color='blue', linestyle='--', label='Confidence threshold (~10Å)')
        plt.axhline(5, color='red', linestyle='--', label='Confidence threshold (~5Å)')
    

    plt.legend()
    plt.tight_layout()

    filename = f"{data_type}_boxplot.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_iptm_vs_ptm(df, output_path:str = None):

    """
    Generates a scatterplot comparing ipTM and pTM values from a DataFrame.

    This function visualizes the relationship between ipTM and pTM scores using a scatterplot.
    It includes reference lines for typical confidence thresholds and saves the plot to the specified path.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'iPTM' and 'pTM' columns.
    output_path : str, optional
        Directory path where the plot will be saved.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If 'iPTM' or 'pTM' columns are missing in the DataFrame.
    FileNotFoundError
        If the output path does not exist.
    """

    plt.figure(figsize=(12,12))
    plt.scatter(df['ipTM'], df['pTM'], alpha=0.7)
    plt.axhline(0.5, color='blue', linestyle='--', label='pTM')
    plt.axvline(0.6, color='green', linestyle='--', label='ipTM')
    plt.axvline(0.8, color='red', linestyle='--', label='ipTM')
    plt.title("iPTM vs pTM")
    plt.xlabel("iPTM")
    plt.ylabel("pTM")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    filename = f"_scatterplot.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def batch_plotting_colabfold(ppi_folder: str, force: bool = False) -> list:
    """
    Generate PAE and pLDDT plots for all ranked models in a single ColabFold PPI output folder.

    Mirrors the behaviour of `batch_plotting` (AF3) but for ColabFold `.pdb` outputs.
    Plots are saved next to each model file as ``{stem}_pae.png`` / ``{stem}_plddt.png``.
    Existing files are reused without regeneration unless ``force=True``.

    Parameters
    ----------
    ppi_folder : str
        Path to a ColabFold PPI subfolder containing ``*_unrelaxed_rank_*.pdb`` model files.
    force : bool, optional
        If True, overwrite existing plot files. Default is False.

    Returns
    -------
    list[Path]
        Paths of all generated or reused plot files (PAE and pLDDT, interleaved by model rank).
    """
    ppi_folder = Path(ppi_folder)
    model_files = sorted(ppi_folder.glob("*_unrelaxed_rank_*.pdb"))

    outputs = []
    for mol_file in model_files:
        stem = mol_file.stem
        save_path_pae = ppi_folder / f"{stem}_pae.png"
        save_path_plddt = ppi_folder / f"{stem}_plddt.png"

        try:
            data = process_full_data_colabfold(str(mol_file))
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARNING] batch_plotting_colabfold: skipping {mol_file.name}: {e}")
            continue

        if save_path_pae.exists() and not force:
            outputs.append(save_path_pae)
        else:
            plot_paes(
                data["pae"],
                data["chain_boundaries_by_res"],
                save_name=str(save_path_pae),
            )
            outputs.append(save_path_pae)

        if save_path_plddt.exists() and not force:
            outputs.append(save_path_plddt)
        else:
            plot_plddt(
                data["ca_plddts"],
                data["chain_boundaries_by_res"],
                save_name=str(save_path_plddt),
            )
            outputs.append(save_path_plddt)

    return outputs




# ---------------------------------------------------------------------------
# Interactome-level plots (standalone; called by InteractomeAnalyzer wrappers)
# ---------------------------------------------------------------------------

def plot_confidence_landscape(
    df: pd.DataFrame,
    output_path=None,
    title: str = "Interactome Confidence Landscape",
) -> None:
    """Scatter plot of pDockQ2 vs ipSAE, sized by MSA depth, coloured by pLDDT.

    Uses the AlphaFold four-band colour scheme. A copy of *df* is used
    internally; the caller's DataFrame is not modified.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    df = df.copy()

    y_col = None
    for col in ["ipSAE_d0_dom_AB", "ipSAE_d0dom_AB", "ipSAE_AB", "ipSAE"]:
        if col in df.columns:
            y_col = col
            break

    pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
    plddt_col = "pLDDT_mean" if "pLDDT_mean" in df.columns else None
    msa_col = "msa_depth" if "msa_depth" in df.columns else None

    if not y_col or pdockq2_col not in df.columns:
        logger.error(f"Required columns for plotting not found (y={y_col}, x={pdockq2_col}).")
        return

    af_colors = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]
    cmap = ListedColormap(af_colors)
    norm = BoundaryNorm([0, 50, 70, 90, 100], cmap.N)

    plt.figure(figsize=(10, 8))

    sizes = np.sqrt(df[msa_col].fillna(0)) * 8 + 15 if msa_col and msa_col in df.columns else 40
    rng = np.random.default_rng(0)
    x_values = df[pdockq2_col] + rng.normal(0, 0.003, size=len(df))
    y_values = df[y_col] + rng.normal(0, 0.003, size=len(df))

    scatter = plt.scatter(
        x_values, y_values, s=sizes,
        c=df[plddt_col] if plddt_col else "gray",
        cmap=cmap, norm=norm, alpha=0.75, edgecolors="black", linewidths=0.5,
    )

    plt.axhline(0.4, color="gray", linestyle="--", alpha=0.4, label="ipSAE_dom 0.4")
    plt.axvline(0.23, color="gray", linestyle="--", alpha=0.4, label="pDockQ2 0.23")
    plt.xlabel("Physical Plausibility (pDockQ2)")
    plt.ylabel(f"Interface Confidence ({y_col})")
    plt.title(title)

    if plddt_col:
        cbar = plt.colorbar(scatter, ticks=[25, 60, 80, 95])
        cbar.set_ticklabels(["<50 (Very Low)", "50-70 (Low)", "70-90 (High)", ">90 (Very High)"])
        cbar.set_label("Mean pLDDT (Global Model Confidence)")

    plt.legend(loc="upper left", fontsize=9, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confidence landscape saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_interactive_landscape(
    df: pd.DataFrame,
    output_path=None,
    title: str = "Interactome Confidence Landscape",
) -> None:
    """Interactive Plotly scatter of pDockQ2 vs ipSAE with hover PPI info.

    Saves an HTML file when *output_path* is provided, otherwise calls
    ``fig.show()``. Requires ``plotly``.
    """
    try:
        import plotly.express as px
    except ImportError:
        logger.error("Plotly is required. Install with: pip install plotly")
        return

    df = df.copy()

    y_col = None
    for col in ["ipSAE_d0_dom_AB", "ipSAE_d0dom_AB", "ipSAE_AB", "ipSAE"]:
        if col in df.columns:
            y_col = col
            break

    pdockq2_col = "pDockQ2_AB" if "pDockQ2_AB" in df.columns else "pDockQ2"
    plddt_col = "pLDDT_mean" if "pLDDT_mean" in df.columns else None
    msa_col = "msa_depth" if "msa_depth" in df.columns else None

    if not y_col or pdockq2_col not in df.columns:
        logger.error("Required columns for interactive plotting not found.")
        return

    if plddt_col:
        df["Confidence_Level"] = pd.cut(
            df[plddt_col], bins=[0, 50, 70, 90, 100],
            labels=["Very Low (<50)", "Low (50-70)", "High (70-90)", "Very High (>90)"],
        )

    size_col = "Size"
    df[size_col] = np.sqrt(df[msa_col].fillna(0)) + 5 if msa_col else 10

    fig = px.scatter(
        df, x=pdockq2_col, y=y_col, size=size_col,
        color="Confidence_Level" if plddt_col else None,
        hover_name="PPI",
        hover_data={
            "ORF_A": True, "ORF_B": True, msa_col: True,
            y_col: ":.3f", pdockq2_col: ":.3f",
            size_col: False, "Confidence_Level": False,
        },
        color_discrete_map={
            "Very High (>90)": "#0053D6", "High (70-90)": "#65CBF3",
            "Low (50-70)": "#FFDB13", "Very Low (<50)": "#FF7D45",
        },
        title=f"{title}<br><sup>Bubble size = sqrt(MSA depth)</sup>",
        labels={
            y_col: "Interface Confidence (ipSAE_dom)",
            pdockq2_col: "Physical Plausibility (pDockQ2)",
        },
        template="plotly_white",
    )
    fig.add_hline(y=0.4, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.23, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(legend_title_text="Global pLDDT",
                      hoverlabel=dict(bgcolor="white", font_size=12))

    if output_path:
        out_file = str(Path(output_path).with_suffix(".html"))
        fig.write_html(out_file)
        logger.info(f"Interactive landscape saved to {out_file}")
    else:
        fig.show()


def plot_network(
    G,
    network_df: pd.DataFrame,
    color_by: str = "betweenness_centrality",
    size_by: str = "degree",
    label_top_n: int = 5,
    output_path=None,
    title: str = "Interactome Network",
) -> None:
    """Force-directed spring layout of the PPI network.

    Node size → *size_by*; node colour → *color_by* (viridis). Edge width
    scales with interaction weight. Top-*label_top_n* nodes are labelled.
    Hub nodes get a red border; bottleneck nodes a blue border.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected PPI graph (built by
        ``InteractomeAnalyzer._build_ppi_graph``).
    network_df : pd.DataFrame
        Per-protein metrics from ``InteractomeAnalyzer.compute_network_properties``.
    """
    import networkx as nx
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    if network_df.empty:
        logger.warning("plot_network: no nodes to plot.")
        return

    pos = nx.spring_layout(G, weight="weight", seed=42)
    node_lookup = network_df.set_index("protein")

    size_vals = network_df.set_index("protein")[size_by].reindex(G.nodes()).fillna(0)
    s_min, s_max = size_vals.min(), size_vals.max()
    node_sizes = (
        300 + 2200 * (size_vals - s_min) / (s_max - s_min)
        if s_max > s_min
        else pd.Series(1000, index=size_vals.index)
    )

    color_vals = network_df.set_index("protein")[color_by].reindex(G.nodes()).fillna(0)
    norm = mcolors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    cmap = cm.viridis
    node_colors = [cmap(norm(color_vals[n])) for n in G.nodes()]

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if len(edge_weights) > 0 and edge_weights.max() > edge_weights.min():
        edge_widths = 0.5 + 2.5 * (edge_weights - edge_weights.min()) / (
            edge_weights.max() - edge_weights.min()
        )
    else:
        edge_widths = np.full(len(edge_weights), 1.5)

    def _node_edge_color(node: str) -> str:
        if node not in node_lookup.index:
            return "grey"
        if node_lookup.at[node, "is_hub"]:
            return "red"
        if node_lookup.at[node, "is_bottleneck"]:
            return "blue"
        return "grey"

    node_edge_colors = [_node_edge_color(n) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.4, edge_color="grey")
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=[node_sizes[n] for n in G.nodes()],
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=2.0, alpha=0.9,
    )

    top_nodes = set(network_df.nlargest(label_top_n, size_by)["protein"].tolist())
    labels = {n: n for n in G.nodes() if n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=color_by.replace("_", " ").title(), shrink=0.7)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="red", markersize=10, label="Hub"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="blue", markersize=10, label="Bottleneck"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(title)
    ax.axis("off")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Network plot saved to {output_path}")
    else:
        plt.show()
    plt.close()
