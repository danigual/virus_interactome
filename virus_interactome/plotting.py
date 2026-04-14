
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from glob import glob
from .utils import load_json, process_full_data_af3, process_full_data_boltz, process_full_data_colabfold

def plot_paes(
    pae_matrix: np.ndarray,
    chain_boundaries: list = None,
    chain_ids: list = None,
    title: str = None,
    # ptm: float = None,
    # iptm: float = None,
    save_name: str = None,
    ax=None
):
    """
    Generates a Predicted Aligned Error (PAE) heatmap using AlphaFold3 output data.

    This function visualizes the PAE matrix, which estimates the positional error between residues
    in a predicted protein structure. It overlays chain boundaries and labels, and includes a colorbar
    indicating the expected error in Ångströms. The plot can be displayed or saved to a file.

    Parameters
    ----------
    summary_confidences_path : str
        Path to the JSON file containing summary confidence metrics (e.g., pTM, ipTM).
    fulldata_path : str
        Path to the JSON file containing the full PAE matrix and chain metadata.
    save_name : str, optional
        If provided, the plot will be saved to this filename. If None, the plot is shown interactively.

    Returns
    -------
    str or None
        The filename where the plot was saved, or None if the plot was displayed.

    Notes
    -----
    - The PAE matrix is visualized using a green colormap with values ranging from 0 to 25 Å.
    - Chain boundaries are marked with black lines.
    - Chain labels are placed at the midpoint of each chain.
    - The plot title includes pTM and ipTM values extracted from the summary file.
    """
    if ax is not None:
        plt.sca(ax)
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

        # if chain_ids is not None:
        #     ax.set_xticklabels(chain_ids)
        
        # Label axes
        # midpoints = [(start + end) / 2 for start, end in chain_boundaries]
        ax.set_xticks(midpoints)
        ax.set_yticks(midpoints)
        ax.set_xticklabels(chain_ids)

        if chain_ids is not None:
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

def plot_af3_output(af3_folder: list):
   
    """
    Processes a directory of AlphaFold3 output files and generates PAE and pLDDT plots.

    This function scans the specified directory for full data JSON files, identifies their corresponding
    summary confidence files, and generates visualizations for both Predicted Aligned Error (PAE) and
    per-residue confidence (pLDDT). If the plots already exist, they are reused; otherwise, they are created
    and saved.

    Parameters
    ----------
    af3_folders : list
        Path to the directory containing AlphaFold3 JSON output files.

    Returns
    -------
    list[Path]
        A list of file paths pointing to the generated (or reused) PNG plots for PAE and pLDDT.

    Raises
    ------
    FileNotFoundError
        If expected JSON files are missing or paths are incorrect.
    """
    # import pdb;pdb.set_trace()
    # results_dir = Path(af3_folders)

    # outputs = []
    # for tmp_folder in af3_folders:
    # results_dir = Path(tmp_folder)

    full_data_files= list(glob(f"{af3_folder}/fold_*_full_data_*.json"))
    

    for tmp_data_file in full_data_files:
        tmp_conf_file = tmp_data_file.replace("full_data", "summary_confidences")
        save_name_pae = tmp_data_file.replace(".json", "_pae").replace("full_data", "model")
        save_name_plddt = tmp_data_file.replace(".json", "_plddt").replace("full_data", "model")
        # save_path_pae = Path (save_name_pae)
        # save_path_plddt = Path (save_name_plddt)

        # if save_path_pae.exists():
        #     outputs.append(save_path_pae)
        
        # else:
        # outfile_pae= 
        plot_paes(tmp_conf_file, tmp_data_file, save_name=save_name_pae)
            # outputs.append(outfile_pae)
        
        # if save_path_plddt.exists():
        #     outputs.append(save_path_plddt)
        
        # else:
        # outfile_plddt = 
        plot_pLDDT(tmp_data_file, save_name=save_name_plddt)
            # outputs.append(outfile_plddt)

    # return outputs

def batch_plotting(results_dir: str):
   
    """
    Processes a directory of AlphaFold3 output files and generates PAE and pLDDT plots.

    This function scans the specified directory for full data JSON files, identifies their corresponding
    summary confidence files, and generates visualizations for both Predicted Aligned Error (PAE) and
    per-residue confidence (pLDDT). If the plots already exist, they are reused; otherwise, they are created
    and saved.

    Parameters
    ----------
    results_dir : str
        Path to the directory containing AlphaFold3 JSON output files.

    Returns
    -------
    list[Path]
        A list of file paths pointing to the generated (or reused) PNG plots for PAE and pLDDT.

    Raises
    ------
    FileNotFoundError
        If expected JSON files are missing or paths are incorrect.
    """
    results_dir = Path(results_dir)
   
    full_data_files= list(results_dir.glob("fold_*_full_data_*.json"))
    
    outputs = []
   
    for tmp_data_file in full_data_files:
        tmp_conf_file = str(tmp_data_file).replace("full_data", "summary_confidences")
        save_name_pae = str(tmp_data_file).replace(".json", "_pae").replace("full_data", "model")
        save_path_pae = Path (save_name_pae)
        save_name_plddt = str(tmp_data_file).replace(".json", "_plddt").replace("full_data", "model")
        save_path_plddt = Path (save_name_plddt)

        if save_path_pae.exists():
            outputs.append(save_path_pae)
        
        else:
            outfile_pae= plot_paes(tmp_conf_file, tmp_data_file, save_name=save_name_pae)
            outputs.append(outfile_pae)
        
        if save_path_plddt.exists():
            outputs.append(save_path_plddt)
        
        else:
            outfile_plddt = plot_pLDDT(tmp_data_file, save_name=save_name_plddt)
            outputs.append(outfile_plddt)

    return outputs

def plot_plddt(plddts: np.ndarray, 
               chain_boundaries: list = None,
               chain_ids: list = None,
               save_name: str = None, ax=None):
    """
    Generates a pLDDT confidence plot for atoms in a predicted protein structure.

    This function visualizes per-atom confidence scores (pLDDT) from AlphaFold3 predictions.
    It uses color-coded confidence bands and overlays chain boundaries. The plot can be displayed
    or saved to a file.

    Parameters
    ----------
    fulldata_path : str
        Path to the JSON file containing full AlphaFold3 output data, including atom-level pLDDT scores.
    save_name : str, optional
        If provided, the plot will be saved to this filename. If None, the plot is shown interactively.

    Returns
    -------
    str or None
        The filename where the plot was saved, or None if the plot was displayed.

    Raises
    ------
    FileNotFoundError
        If the input JSON file does not exist.
    KeyError
        If expected keys are missing in the input data.
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

        # Label axes
        ax.set_xticks(midpoints)

        # if chain_ids is not None:
        #     ax.set_xticklabels(chain_ids)
        
        ax.set_xticklabels(chain_ids)

    plddt_legend = {
        "Very high (pLDDT > 90)": "#024fcc86",
        "High (90 > pLDDT > 70)": "#60c2e886",
        "Low (70 > pLDDT > 50)": "#f3784286",
        "Very low (pLDDT < 50)": "#f9d61386",
    }

    ax.legend(plddt_legend, title="pLDDT Confidence", prop={'size': 10}, 
            #   loc="lower center",
              bbox_to_anchor=(0.85, 1.15),
            frameon=False,
            ncol=4)

    ax.set_ylabel("pLDDT")

    if save_name is None:
        plt.tight_layout
        plt.show()
    else:
        plt.tight_layout
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
        plt.tight_layout
        plt.show()
    else:
        plt.tight_layout
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
                labels=list(orf_labels))
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
    plt.scatter(df['iPTM'], df['pTM'], alpha=0.7)
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


def batch_plotting_colabfold(ppi_folder: str) -> list:
    """
    Generate PAE and pLDDT plots for all ranked models in a single ColabFold PPI output folder.

    Mirrors the behaviour of `batch_plotting` (AF3) but for ColabFold `.pdb` outputs.
    Plots are saved next to each model file as ``{stem}_pae.png`` / ``{stem}_plddt.png``.
    Existing files are reused without regeneration.

    Parameters
    ----------
    ppi_folder : str
        Path to a ColabFold PPI subfolder containing ``*_unrelaxed_rank_*.pdb`` model files.

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

        if save_path_pae.exists():
            outputs.append(save_path_pae)
        else:
            plot_paes(
                data["pae"],
                data["chain_boundaries_by_res"],
                save_name=str(save_path_pae),
            )
            outputs.append(save_path_pae)

        if save_path_plddt.exists():
            outputs.append(save_path_plddt)
        else:
            plot_plddt(
                data["ca_plddts"],
                data["chain_boundaries_by_res"],
                save_name=str(save_path_plddt),
            )
            outputs.append(save_path_plddt)

    return outputs


