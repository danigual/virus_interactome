from pathlib import Path

import numpy as np
from moleculekit.molecule import Molecule
from typing import Union
import pandas as pd
import itertools
import math

residue_set= {"ALA", "ARG", "ASN", "ASP", "CYS",
              "GLN", "GLU", "GLY", "HIS", "ILE",
              "LEU", "LYS", "MET", "PHE", "PRO",
              "SER", "THR", "TRP", "TYR", "VAL",
              "DA", "DC", "DT", "DG", "A", "C", "U", "G"}

nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

def ptm_func(x: float, d0: float) -> float:
    """TM-score sigmoid kernel: 1 / (1 + (x/d0)^2)."""
    return 1.0/(1+(x/d0)**2.0)

ptm_func_vec = np.vectorize(ptm_func)  # vectorized version for NumPy arrays


def calc_d0(L: float, pair_type: str) -> float:
    """Compute d0 normalisation constant for a chain pair.

    Formula from Yang & Skolnick, Proteins 57:702-710 (2004).
    Minimum value is 1.0 for proteins, 2.0 for nucleic acids.

    Parameters
    ----------
    L : float
        Total number of residues in the pair (clamped to ≥ 27).
    pair_type : str
        ``'protein'`` or ``'nucleic_acid'``.
    """
    L = float(L)
    if L < 27:
        L = 27
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calc_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
    """Vectorized version of :func:`calc_d0` for per-residue d0 calculation."""
    L = np.maximum(27, np.array(L, dtype=float))
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def init_chainpairdict_zeros(chainlist: list) -> dict:
    """Return nested ``{chain1: {chain2: 0}}`` dict for all ordered pairs."""
    return {c1: {c2: 0 for c2 in chainlist if c1 != c2} for c1 in chainlist}


def init_chainpairdict_npzeros(chainlist: list, arraysize: int) -> dict:
    """Return nested ``{chain1: {chain2: np.zeros(arraysize)}}`` dict."""
    return {c1: {c2: np.zeros(arraysize) for c2 in chainlist if c1 != c2} for c1 in chainlist}


def init_chainpairdict_set(chainlist: list) -> dict:
    """Return nested ``{chain1: {chain2: set()}}`` dict for all ordered pairs."""
    return {c1: {c2: set() for c2 in chainlist if c1 != c2} for c1 in chainlist}


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> dict:
    """Classify each chain as ``'protein'`` or ``'nucleic_acid'``.

    Parameters
    ----------
    chains : np.ndarray
        Per-atom/residue chain identifiers (e.g. ``mol.chain[cb_mask]``).
    residue_types : np.ndarray
        Per-atom/residue three-letter residue names (e.g. ``mol.resname``).

    Returns
    -------
    dict
        ``{chain_id: 'protein' | 'nucleic_acid'}``
    """
    chain_types = {}
    for chain in np.unique(chains):
        indices = np.where(chains == chain)[0]
        nuc_count = sum(r in nuc_residue_set for r in residue_types[indices])
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'
    return chain_types


def _load_mol(mol_file) -> Molecule:
    """Return a Molecule instance from a path or pass-through if already loaded."""
    if isinstance(mol_file, Molecule):
        return mol_file
    if not Path(mol_file).exists():
        raise FileNotFoundError(f"File {mol_file} does not exist")
    try:
        return Molecule(mol_file)
    except Exception as exc:
        raise ValueError(f"Could not parse structure file {mol_file}: {exc}") from exc


def calculate_pdockq(
    mol_file: Union[str, Molecule],
    plddt_by_res: np.ndarray,
    pDockQ_cutoff: float = 8.0,
) -> pd.DataFrame:
    """Compute pDockQ for every unordered chain pair (Bryant et al. 2022).

    Parameters
    ----------
    mol_file : str or Molecule
        Path to a ``.cif`` / ``.pdb`` file or a pre-loaded :class:`Molecule`.
    plddt_by_res : np.ndarray
        Per-residue pLDDT values (Cβ-indexed, same order as the Molecule).
    pDockQ_cutoff : float
        Cβ–Cβ distance cutoff (Å) defining interface residues.

    Returns
    -------
    pd.DataFrame
        Columns: ``chain1``, ``chain2``, ``pDockQ``.
        One row per unordered pair (combinations, not permutations).
    """
    mol = _load_mol(mol_file)
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    cb_plddt = plddt_by_res

    coordinates = mol.coords[cb_mask].reshape(-1,3)
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    chains = mol.chain[cb_mask]
    unique_chains = [str(i) for i in np.unique(chains)]

    pDockQ = pd.DataFrame({"chain1": [], "chain2": [], "pDockQ": []})
    for chain1, chain2 in itertools.combinations(unique_chains, 2):
        interface_distances = distances[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_matrix = interface_distances<=pDockQ_cutoff
        npairs = np.sum(interface_matrix)

        tmp_pdockq = 0.0          
        if npairs > 0:
            res_chain1_indices, res_chain2_indices = np.where(interface_matrix)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)

            cb_plddt_chain1 = cb_plddt[chains==chain1][unique_res_chain1_indexes]
            cb_plddt_chain2 = cb_plddt[chains==chain2][unique_res_chain2_indexes]
            mean_plddt = np.concatenate((cb_plddt_chain1, cb_plddt_chain2)).mean()
            x = mean_plddt * math.log10(npairs)
            tmp_pdockq = 0.724 / (1 + math.exp(-0.052*(x-152.611)))+0.018
        pDockQ = pd.concat([pDockQ, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], "pDockQ": [tmp_pdockq]})], ignore_index=True)

    return pDockQ

def calculate_pdockq2(
    mol_file: Union[str, Molecule],
    plddt_by_res: np.ndarray,
    pae_matrix: np.ndarray,
    pDockQ_cutoff: float = 8.0,
) -> pd.DataFrame:
    """Compute pDockQ2 for every ordered chain pair (Zhu et al. 2023).

    Unlike pDockQ, pDockQ2 incorporates PAE via the TM-score kernel and is
    computed for each ordered (chain1 → chain2) direction.

    Parameters
    ----------
    mol_file : str or Molecule
        Path to a ``.cif`` / ``.pdb`` file or a pre-loaded :class:`Molecule`.
    plddt_by_res : np.ndarray
        Per-residue pLDDT values (Cβ-indexed).
    pae_matrix : np.ndarray
        Full PAE matrix (N_res × N_res).
    pDockQ_cutoff : float
        Cβ–Cβ distance cutoff (Å) defining interface residues.

    Returns
    -------
    pd.DataFrame
        Columns: ``chain1``, ``chain2``, ``pDockQ2``.
        One row per ordered pair (permutations).
    """
    mol = _load_mol(mol_file)
        
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    cb_plddt = plddt_by_res

    coordinates = mol.coords[cb_mask].reshape(-1,3)
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    chains = mol.chain[cb_mask]
    unique_chains = [str(i) for i in np.unique(chains)]

    pDockQ2 = pd.DataFrame({"chain1": [], "chain2": [], "pDockQ2": []})
    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        interface_distances = distances[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_pae = pae_matrix[np.ix_(np.where(chains==chain1)[0], np.where(chains==chain2)[0])]
        interface_dist_matrix = interface_distances<=pDockQ_cutoff
        npairs = np.sum(interface_dist_matrix)

        tmp_pdockq2 = 0.0          
        if npairs > 0:
            pae_ptm_sum = np.sum(ptm_func_vec(interface_pae[interface_dist_matrix], 10))
            res_chain1_indices, res_chain2_indices = np.where(interface_dist_matrix)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)

            cb_plddt_chain1 = cb_plddt[chains==chain1][unique_res_chain1_indexes]
            cb_plddt_chain2 = cb_plddt[chains==chain2][unique_res_chain2_indexes]
            mean_plddt = np.concatenate((cb_plddt_chain1, cb_plddt_chain2)).mean()
            mean_ptm = pae_ptm_sum / npairs
            x = mean_plddt * mean_ptm
            tmp_pdockq2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        pDockQ2 = pd.concat([pDockQ2, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], "pDockQ2": [tmp_pdockq2]})], ignore_index=True)

    return pDockQ2

def calculate_LIS(mol_file, pae_matrix, threshold=12.0):
    """Backwards-compatible wrapper. Returns DataFrame with LIS only."""
    lis_df = calculate_LIS_family(mol_file, pae_matrix, pae_cutoff=threshold)
    return lis_df[["chain1", "chain2", "LIS"]].copy()


def calculate_LIS_family(
    mol_file,
    pae_matrix,
    pae_cutoff: float = 12.0,
    contact_dist: float = 8.0,
) -> pd.DataFrame:
    """Compute the full LIS metric family for every ordered inter-chain pair.

    Metrics (Kim et al. 2024/2025):
      LIS   — mean (1 - PAE/cutoff) for Cβ pairs with PAE ≤ cutoff
      LIA   — count of Cβ pairs with PAE ≤ cutoff
      cLIS  — same as LIS but also requiring Cβ–Cβ distance ≤ contact_dist
      cLIA  — count of pairs satisfying both PAE and distance filters
      iLIS  — sqrt(LIS * cLIS); 0 when cLIA == 0 (no physical contacts)
      iLIA  — sqrt(LIA * cLIA)
      LIR   — residue indices on chain1 involved in any valid PAE pair (chain1 side)
      cLIR  — residue indices on chain1 involved in any contact pair (chain1 side)

    Returns a DataFrame with columns:
        chain1, chain2, LIS, LIA, cLIS, cLIA, iLIS, iLIA, LIR, cLIR
    One row per ordered (chain1, chain2) pair.
    """
    mol = _load_mol(mol_file)

    cb_mask = np.logical_or(
        mol.name == "CB",
        np.logical_and(mol.resname == "GLY", mol.name == "CA"),
    )
    chains = mol.chain[cb_mask]
    unique_chains = [str(c) for c in np.unique(chains)]
    coords = mol.coords[cb_mask].reshape(-1, 3)  # (N_cb, 3)

    # Pairwise Cβ–Cβ distance matrix (full N×N, reused across chain pairs)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))              # (N, N)

    rows = []
    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        mask_c1 = chains == chain1
        mask_c2 = chains == chain2

        # Inter-chain PAE and distance sub-matrices
        sub_pae  = pae_matrix[np.ix_(mask_c1, mask_c2)]   # (n1, n2)
        sub_dist = dist_matrix[np.ix_(mask_c1, mask_c2)]  # (n1, n2)

        pae_mask     = sub_pae <= pae_cutoff
        contact_mask = pae_mask & (sub_dist <= contact_dist)

        # LIA / LIS
        lia = int(pae_mask.sum())
        if lia > 0:
            lis = float(np.mean((pae_cutoff - sub_pae[pae_mask]) / pae_cutoff))
        else:
            lis = 0.0

        # cLIA / cLIS
        clia = int(contact_mask.sum())
        if clia > 0:
            clis = float(np.mean((pae_cutoff - sub_pae[contact_mask]) / pae_cutoff))
        else:
            clis = 0.0

        # iLIS / iLIA — geometric mean; 0 when no physical contact
        ilis = float(np.sqrt(lis * clis))
        ilia = float(np.sqrt(lia * clia))

        # LIR / cLIR — 0-based residue indices on chain1 side
        lir_indices  = sorted(set(np.where(pae_mask)[0].tolist()))
        clir_indices = sorted(set(np.where(contact_mask)[0].tolist()))
        lir_str  = ",".join(map(str, lir_indices))
        clir_str = ",".join(map(str, clir_indices))

        rows.append({
            "chain1": chain1, "chain2": chain2,
            "LIS": lis, "LIA": lia,
            "cLIS": clis, "cLIA": clia,
            "iLIS": ilis, "iLIA": ilia,
            "LIR": lir_str, "cLIR": clir_str,
        })

    return pd.DataFrame(rows)

def calculate_ipsae(
    mol_file: Union[str, Molecule],
    pae_matrix: np.ndarray,
    pae_cutoff: float = 10.0,
) -> pd.DataFrame:
    """Compute ipSAE for every ordered chain pair (Dunbrack lab, 2025).

    Three variants are returned per pair:

    - ``ipSAE``       — per-residue d0 normalisation (d0res).
    - ``ipSAE_d0chn`` — combined-chain-length d0 normalisation.
    - ``ipSAE_d0dom`` — interface-domain-length d0 normalisation (preferred for ranking).

    Parameters
    ----------
    mol_file : str or Molecule
        Path to a ``.cif`` / ``.pdb`` file or a pre-loaded :class:`Molecule`.
    pae_matrix : np.ndarray
        Full PAE matrix (N_res × N_res).
    pae_cutoff : float
        PAE threshold (Å) for defining interface residue pairs.

    Returns
    -------
    pd.DataFrame
        Columns: ``chain1``, ``chain2``, ``ipSAE``, ``ipSAE_d0chn``, ``ipSAE_d0dom``.
        One row per ordered pair (permutations).
    """
    mol = _load_mol(mol_file)
        
    cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
    numres = cb_mask.sum()
    chains = mol.chain[cb_mask]
    unique_chains = [str(i) for i in np.unique(chains)]

    ipsae = pd.DataFrame({"chain1": [], "chain2": [], "ipSAE": [], "ipSAE_d0chn": [], "ipSAE_d0dom": []})
    chain_dict = classify_chains(chains, mol.resname)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1==chain2: continue
            if chain_dict[chain1] == 'nucleic_acid' or chain_dict[chain2] == 'nucleic_acid':
                chain_pair_type[chain1][chain2]='nucleic_acid'
            else:
                chain_pair_type[chain1][chain2]='protein'

    for chain1, chain2 in itertools.permutations(unique_chains, 2):
        n0chn = np.sum(chains==chain1) + np.sum(chains==chain2) # total number of residues in chain1 and chain2
        d0chn = calc_d0(n0chn, chain_pair_type[chain1][chain2])
        ptm_matrix_d0chn=ptm_func_vec(pae_matrix, d0chn)

        interface_pae = pae_matrix[chains == chain1][:, chains == chain2]
        interface_pae_mask = interface_pae < pae_cutoff

        if interface_pae_mask.any():
            interface_ptm_d0chn = ptm_matrix_d0chn[chains == chain1][:, chains == chain2]
            interface_ptm_d0chn[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0chn_byres = np.nanmean(interface_ptm_d0chn, axis=1)
            tmp_ipSAE_d0chn = np.nanmax(ipsae_d0chn_byres)

            res_chain1_indices, res_chain2_indices = np.where(interface_pae_mask)
            unique_res_chain1_indexes = np.unique(res_chain1_indices)
            unique_res_chain2_indexes = np.unique(res_chain2_indices)
            n0dom = len(unique_res_chain1_indexes) + len(unique_res_chain2_indexes)
            d0dom = calc_d0(n0dom, chain_pair_type[chain1][chain2])
            ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom)
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)
            interface_ptm_d0dom = ptm_matrix_d0dom[chains == chain1][:, chains == chain2]
            interface_ptm_d0dom[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0dom_byres = np.nanmean(interface_ptm_d0dom, axis=1)
            tmp_ipSAE_d0dom = np.nanmax(ipsae_d0dom_byres)
            
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])
            ptm_matrix_d0res = np.array([ptm_func_vec(pae_matrix[i], d0res_byres[i]) for i in range(numres)])
            interface_ptm_d0res = ptm_matrix_d0res[chains == chain1][:, chains == chain2]
            interface_ptm_d0res[np.logical_not(interface_pae_mask)] = np.nan
            ipsae_d0res_byres = np.nanmean(interface_ptm_d0res, axis=1)
            tmp_ipSAE_d0res = np.nanmax(ipsae_d0res_byres)
            # print(ipsae_d0res_byres.shape)

            ipsae = pd.concat([ipsae, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], 
                                                    "ipSAE": [tmp_ipSAE_d0res], "ipSAE_d0chn": [tmp_ipSAE_d0chn],
                                                    "ipSAE_d0dom": [tmp_ipSAE_d0dom]})])
        else: 
            ipsae = pd.concat([ipsae, pd.DataFrame({"chain1": [chain1], "chain2": [chain2], 
                                                    "ipSAE": [0], "ipSAE_d0chn": [0],
                                                    "ipSAE_d0dom": [0]})])
    return ipsae.reset_index(drop=True)


def calculate_all_metrics(
    mol_file: Union[str, Molecule],
    all_metrics: dict,
) -> dict:
    """Compute the full set of interface metrics for a heterodimer model.

    Calls :func:`calculate_ipsae`, :func:`calculate_LIS_family`,
    :func:`calculate_pdockq`, and :func:`calculate_pdockq2` and aggregates
    results into a flat dictionary keyed by column name.

    Parameters
    ----------
    mol_file : str or Molecule
        Path to the ``.cif`` model file or a pre-loaded :class:`Molecule`.
    all_metrics : dict
        Parsed model data as returned by ``process_full_data_af3 / boltz / colabfold``.
        Required keys: ``pae``, ``cb_plddts``, ``token_chain_ids``.

    Returns
    -------
    dict
        All metric values keyed by column name (e.g. ``ipSAE_AB``, ``LIS_AB``,
        ``Best_iLIS``, ``pDockQ2_AB``, …).
    """
    mol = _load_mol(mol_file)

    pae = all_metrics["pae"]
    plddt_by_residue = all_metrics["cb_plddts"]
    chain_by_res = np.array(all_metrics["token_chain_ids"])

    ipsae = calculate_ipsae(mol, pae)

    ipSAE_AB = ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE"].values[0]
    ipSAE_BA = ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE"].values[0]
    lis_df = calculate_LIS_family(mol, pae)
    pdockq = calculate_pdockq(mol, plddt_by_res=plddt_by_residue)
    pdockq2 = calculate_pdockq2(mol, plddt_by_res=plddt_by_residue, pae_matrix=pae)

    def _lis_val(chain1: str, chain2: str, col: str):
        """Safe lookup for a LIS-family column; returns 0.0 if row missing."""
        row = lis_df.loc[(lis_df.chain1 == chain1) & (lis_df.chain2 == chain2), col]
        return row.values[0] if len(row) > 0 else 0.0

    lis_ab   = _lis_val("A", "B", "LIS")
    lis_ba   = _lis_val("B", "A", "LIS")
    lia_ab   = _lis_val("A", "B", "LIA")
    lia_ba   = _lis_val("B", "A", "LIA")
    clis_ab  = _lis_val("A", "B", "cLIS")
    clis_ba  = _lis_val("B", "A", "cLIS")
    clia_ab  = _lis_val("A", "B", "cLIA")
    clia_ba  = _lis_val("B", "A", "cLIA")
    ilis_ab  = _lis_val("A", "B", "iLIS")
    ilis_ba  = _lis_val("B", "A", "iLIS")
    ilia_ab  = _lis_val("A", "B", "iLIA")
    ilia_ba  = _lis_val("B", "A", "iLIA")

    return {
        "pLDDT_mean": np.mean(plddt_by_residue),
        "pLDDT_mean_A": np.mean(plddt_by_residue[chain_by_res == "A"]),
        "pLDDT_mean_B": np.mean(plddt_by_residue[chain_by_res == "B"]),
        "pLDDT_median_A": np.median(plddt_by_residue[chain_by_res == "A"]),
        "pLDDT_median_B": np.median(plddt_by_residue[chain_by_res == "B"]),
        "pae_mean": np.mean(pae),
        "pae_mean_A": np.mean(pae[chain_by_res == "A"][:, chain_by_res == "A"]),
        "pae_mean_B": np.mean(pae[chain_by_res == "B"][:, chain_by_res == "B"]),
        "pae_mean_AB": np.mean([np.mean(pae[chain_by_res == "A"][:, chain_by_res == "B"]),
                                np.mean(pae[chain_by_res == "B"][:, chain_by_res == "A"])]),
        "pDockQ": pdockq.loc[(pdockq.chain1 == "A") & (pdockq.chain2 == "B"), "pDockQ"].values[0],
        "pDockQ2_AB": pdockq2.loc[(pdockq2.chain1 == "A") & (pdockq2.chain2 == "B"), "pDockQ2"].values[0],
        "pDockQ2_BA": pdockq2.loc[(pdockq2.chain1 == "B") & (pdockq2.chain2 == "A"), "pDockQ2"].values[0],
        # LIS family (Kim et al. 2024/2025)
        "LIS_AB": lis_ab,  "LIS_BA": lis_ba,
        "LIA_AB": lia_ab,  "LIA_BA": lia_ba,
        "cLIS_AB": clis_ab, "cLIS_BA": clis_ba,
        "cLIA_AB": clia_ab, "cLIA_BA": clia_ba,
        "iLIS_AB": ilis_ab, "iLIS_BA": ilis_ba,
        "iLIA_AB": ilia_ab, "iLIA_BA": ilia_ba,
        "Best_LIS":  float(max(lis_ab,  lis_ba)),
        "Best_LIA":  float(max(lia_ab,  lia_ba)),
        "Best_iLIS": float(max(ilis_ab, ilis_ba)),
        "Best_iLIA": float(max(ilia_ab, ilia_ba)),
        # LIR residue index strings (chain1 side for A→B direction)
        "LIR_AB":  _lis_val("A", "B", "LIR"),
        "cLIR_AB": _lis_val("A", "B", "cLIR"),
        # ipSAE (Dunbrack 2025)
        "ipSAE_AB": ipSAE_AB,
        "ipSAE_BA": ipSAE_BA,
        "max_ipSAE": float(np.max([ipSAE_AB, ipSAE_BA])),
        "ipSAE_d0chn_AB": ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE_d0chn"].values[0],
        "ipSAE_d0chn_BA": ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE_d0chn"].values[0],
        "ipSAE_d0dom_AB": ipsae.loc[(ipsae.chain1 == "A") & (ipsae.chain2 == "B"), "ipSAE_d0dom"].values[0],
        "ipSAE_d0dom_BA": ipsae.loc[(ipsae.chain1 == "B") & (ipsae.chain2 == "A"), "ipSAE_d0dom"].values[0],
    }