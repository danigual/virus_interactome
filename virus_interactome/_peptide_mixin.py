import os
import logging
import subprocess
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from moleculekit.molecule import Molecule
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class _PeptidePipelineMixin:
    """Peptide-protein structural analysis pipeline methods for InteractomeAnalyzer."""

    def run_full_pipeline(self, ipsae_filter: Optional[float] = None, **kwargs):
        """
        Executes the complete analysis pipeline.

        Parameters
        ----------
        ipsae_filter : float, optional
            If provided, logs how many PPIs are above this confidence threshold.
            Structural analysis will proceed for ALL candidates regardless.
        **kwargs
            Arguments passed to downstream methods:
            - cluster_ratio_threshold (float, default=7.0)
            - min_peptide_len (int, default=5)
        """
        if self._cluster_data is None:
            logger.warning("Cannot run pipeline: Cluster data is missing.")
            return

        logger.info("Starting peptide-protein analysis pipeline...")

        # Default cluster ratio for peptides set to 7.0 as requested
        kwargs.setdefault('cluster_ratio_threshold', 7.0)

        # Log filtering info if requested, but DO NOT filter the structural pipeline
        if ipsae_filter is not None and self._interactome_data is not None:
            df = self._interactome_data
            ipsae_col = None
            for col in ["ipSAE_d0dom_AB", "ipSAE_d0_dom_AB", "ipSAE_AB", "ipSAE"]:
                if col in df.columns:
                    ipsae_col = col
                    break

            if ipsae_col:
                high_conf_ppis = df[df[ipsae_col] > ipsae_filter]["PPI"].unique()
                logger.info(f"Report: {len(high_conf_ppis)} PPIs are above {ipsae_col} > {ipsae_filter}.")

        # Run structural analysis for ALL candidates
        self.analyze_peptide_proteins_pairs(**kwargs)

    def _get_candidate_clusters(self, cluster_ratio_threshold: float = 5.0,
                                min_peptide_len: int = 5) -> pd.DataFrame:
        """
        Identifies candidate peptide-protein interactions based on cluster geometry.

        Filters clusters that have a high aspect ratio (elongated shape), suggesting
        a peptide binding to a larger protein surface. It determines which chain
        corresponds to the 'Binder' (protein) and which to the 'Peptide' based on
        the dimensions of the interaction interface.

        Parameters
        ----------
        cluster_ratio_threshold : float, optional
            The minimum aspect ratio (max_side / min_side) required to consider
            a cluster as a peptide candidate. Defaults to 5.0.
        min_peptide_len : int, optional
            The minimum length (in residues) of the interface to be considered valid.
            Defaults to 5.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing candidate clusters with additional columns:
            - Binder_name, Binder_chain, Binder_start, Binder_end
            - Peptide_name, Peptide_chain, Peptide_start, Peptide_end
        """
        # 1. Validate Data Availability
        if self._cluster_data is None or self._cluster_data.empty:
            logger.warning("Cluster data is empty. No candidates to process.")
            return pd.DataFrame()

        #Generate a copy of the df with candidate clusters
        df = self._cluster_data.copy()

        # 2. Handle Legacy Column Naming (Cluster_ratio vs cluster_ratio)
        ratio_col = "cluster_ratio" if "cluster_ratio" in df.columns else "Cluster_ratio"

        # 3. Apply Filters (Geometry & Thresholds)
        # Ensure dimensions are positive and ratio meets threshold
        mask = (
            (df["x_len"] > 0) &
            (df["y_len"] > 0) &
            (df[ratio_col] > cluster_ratio_threshold) &
            (df["x_len"] >= min_peptide_len) &
            (df["y_len"] >= min_peptide_len)
        )
        candidate_clusters = df[mask].copy()

        # 4. Process Candidates (Identify Binder vs Peptide)
        new_cols = {
            "Binder_chain": [], "Binder_name": [], "Binder_start": [], "Binder_end": [],
            "Peptide_chain": [], "Peptide_name": [], "Peptide_start": [], "Peptide_end": []
        }

        for _, row in candidate_clusters.iterrows():
            # Robust split (take first two elements only) to handle names like "GenA__GenB__v2"
            parts = row["PPI"].split("__")
            if len(parts) >= 2:
                orf_a, orf_b = parts[:2]
            else:
                # Fallback for malformed PPI IDs
                orf_a, orf_b = row["PPI"], ""

            # Logic: Assign roles based on dimensions
            # If X dimension > Y dimension -> Chain A is Binder, Chain B is Peptide

            if row["x_len"] > row["y_len"]:
                # X is longer
                new_cols["Binder_chain"].append("A")
                new_cols["Peptide_chain"].append("B")
                new_cols["Peptide_start"].append(int(row["y_min"]))
                new_cols["Peptide_end"].append(int(row["y_max"]))
                new_cols["Binder_start"].append(int(row["x_min"]))
                new_cols["Binder_end"].append(int(row["x_max"]))
                new_cols["Binder_name"].append(orf_a)
                new_cols["Peptide_name"].append(orf_b)
            else:
                # Y is longer
                new_cols["Binder_chain"].append("B")
                new_cols["Peptide_chain"].append("A")
                new_cols["Peptide_start"].append(int(row["x_min"]))
                new_cols["Peptide_end"].append(int(row["x_max"]))
                new_cols["Binder_start"].append(int(row["y_min"]))
                new_cols["Binder_end"].append(int(row["y_max"]))
                new_cols["Binder_name"].append(orf_b)
                new_cols["Peptide_name"].append(orf_a)

        # 5. Assign new columns to DataFrame
        for col_name, data_list in new_cols.items():
            candidate_clusters[col_name] = data_list


        # Return with a clean index
        return candidate_clusters.reset_index(drop=True)

    def _curate_protein_peptide_models(self, data: pd.Series) -> Molecule:
        """
        Loads a PDB model and standardizes chain identifiers for analysis.

        This method ensures a consistent schema where:
        1. Chain A represents the 'Binder' (Protein).
        2. Chain B represents the 'Peptide'.
        3. Only the interface residues of the peptide are kept.

        Parameters
        ----------
        data : pd.Series
            A row from the candidate clusters DataFrame containing:
            - "path": Path to the PDB file.
            - "Peptide_chain": Original chain ID of the peptide ('A' or 'B').
            - "Peptide_start": Start residue index of the peptide interface.
            - "Peptide_end": End residue index of the peptide interface.

        Returns
        -------
        Molecule
            A MoleculeKit object with standardized chains and filtered atoms.
        """
        # Extract and cast types
        mol_path = str(data["path"])
        peptide_chain = str(data["Peptide_chain"])
        peptide_start = int(data["Peptide_start"])
        peptide_end = int(data["Peptide_end"])

        mol = Molecule(mol_path)

        # Standardize: Binder -> A, Peptide -> B
        if peptide_chain == "A":
            # Swap logic using temporary chain 'C'
            mol.set("chain", "C", "chain A") # Peptide(A) -> C
            mol.set("chain", "A", "chain B") # Binder(B) -> A
            mol.set("chain", "B", "chain C") # Peptide(C) -> B

        # Filter: Keep Binder (A) or Peptide (B) within interface range
        mol.filter(f"(chain A) or (chain B and resid {peptide_start} to {peptide_end})")
        return mol

    def _create_binder_alignments(self,
                                  model_to_align: str,
                                  reference_model: Union[str, Molecule]) -> Molecule:
        """
        Aligns a target protein model to a reference structure based on the Binder (Chain A).

        This method performs a structural alignment using the following steps:
        1. Loads the target model.
        2. Filters the target's Chain A to keep only residues present in the reference
           (ensuring a valid index-based alignment).
        3. Preserves Chain B (Peptide) entirely.
        4. Aligns the filtered target to the reference.

        Parameters
        ----------
        model_to_align : str
            Path to the PDB file of the model to be aligned.
        reference_model : str or Molecule
            The reference structure used as the fixed point. Can be a file path
            (str) or a loaded Molecule object.

        Returns
        -------
        Molecule
            The aligned Molecule object (containing filtered Chain A + Chain B).
        """
        # 1. Resolve Reference Model (Path vs Object)
        if isinstance(reference_model, str):
            reference_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            reference_mol = reference_model
        else:
            raise ValueError("reference_model should be a path (str) or a Molecule instance")

        # 2. Get Reference Residues
        residues_in_reference_chain = reference_mol.resid.astype(str)
        reference_resid_str = ' '.join(residues_in_reference_chain)

        # 3. Load Target Model
        tmp_mol = Molecule(model_to_align)

        # 4. Filter Target
        # Keep Chain A (only matching reference residues) OR Chain B (Peptide)
        tmp_mol.filter(f"(chain A and resid {reference_resid_str}) or (chain B)")

        # 5. Align
        # mode="index" requires strict atom-to-atom correspondence.
        tmp_mol.align("chain A",
                    refmol=reference_mol,
                    refsel="chain A",
                    mode="index"
                    )
        return tmp_mol

    def _get_reference_structure_for_binder(self, all_structs: List[str]) -> Molecule:
        """
        Selects the best structural model to serve as a reference for alignment.

        The selection relies on the pLDDT score.
        It selects the model with the highest median pLDDT for Chain A (Binder)
        and trims it to keep only the high-confidence residues (pLDDT > 70).

        Parameters
        ----------
        all_structs : List[str]
            A list of file paths to the PDB models.

        Returns
        -------
        Molecule
            A MoleculeKit object of the reference binder, containing only
            Chain A residues with high structural confidence.
        """
        # Load all molecules
        mol_list = [ Molecule(i) for i in all_structs]

        plddt_scores = []
        for mol in mol_list:
            # Select Chain A (Binder)
            # numpy boolean masking on the 'chain' attribute
            mask_a = mol.chain == "A"
            if np.any(mask_a):
                plddt_chain_A = mol.beta[mask_a]
                score = np.median(plddt_chain_A)
            else:
                score = 0.0 # Fallback if chain A is missing
            plddt_scores.append(score)

        plddt_scores = np.array(plddt_scores)

        # Select the best index (Highest median pLDDT)
        best_global_idx = np.argmax(plddt_scores)

        # Define the reference molecule, just one by binder
        reference_mol = mol_list[best_global_idx].copy()

        # 1. Filter: Keep only Chain A
        reference_mol.filter("chain A")

        # 2. Filter: Keep only high confidence residues (pLDDT > 70)
        # Get Residue IDs of CA atoms where Beta > 70
        # We use CA atoms as representative for the whole residue to avoid duplicates
        mask_ca = reference_mol.name == "CA"

        # Two-step masking:
        # 1. Get betas for CAs.
        # 2. Check which are > 70.
        # 3. Apply that boolean mask to the resids of CAs.
        high_conf_mask = reference_mol.beta[mask_ca] > 70
        reference_resids = reference_mol.resid[mask_ca][high_conf_mask]

        # Create selection string
        reference_resids = reference_resids.astype(str)
        reference_resid_str = ' '.join(reference_resids)

        # Apply final filter
        if reference_resid_str:
            reference_mol.filter(f"resid {reference_resid_str}")
        else:
            logger.warning("No high-confidence residues (pLDDT > 70) found in best model. Returning full Chain A.")

        return reference_mol

    def analyze_peptide_proteins_pairs(self, **kwargs):
        """
        Executes the structural analysis pipeline for peptide-protein interactions.
        """
        # 1. Setup Directories
        output_path = f"{self.output_path}/prot_peptide"
        os.makedirs(output_path, exist_ok=True)

        # Extract filtering arguments intended for _get_candidate_clusters
        filter_args = {
            'cluster_ratio_threshold': kwargs.pop('cluster_ratio_threshold', 5.0),
            'min_peptide_len': kwargs.pop('min_peptide_len', 5)
        }

        # 2. Identify Candidates
        self._candidate_clusters = self._get_candidate_clusters(**filter_args)

        if self._candidate_clusters is None or self._candidate_clusters.empty:
            logger.warning("No candidate peptide-protein clusters found. Skipping analysis.")
            return

        # Create folder structure for each unique binder
        for binder in self._candidate_clusters.Binder_name.unique():
            os.makedirs(f"{output_path}/{binder}/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/filtered/", exist_ok=True)
            os.makedirs(f"{output_path}/{binder}/aligned/", exist_ok=True)

        # 3. Prepare Working DataFrame
        cols = ["PPI", "model_num", "x_len", "y_len",
                "Binder_name", "Binder_chain", "Binder_start", "Binder_end",
                "Peptide_name", "Peptide_chain", "Peptide_start", "Peptide_end", "path"]

        df = self._candidate_clusters.loc[:,cols].copy()

        # Generate Unique IDs (PPI + Model + ClusterID)
        df.loc[: , "PPI"] = df.PPI + "_" +  self._candidate_clusters.model_num.astype(str) + "_" + self._candidate_clusters.cluster_id.astype(str)

        # 4. Filtering Loop (Curate Structures)
        filtered_names = []
        for idx, row in df.iterrows():
            output_name = f"{output_path}/{row.Binder_name}/filtered/{row.PPI}.pdb"
            filtered_names.append(output_name)

            if not os.path.exists(output_name):
                mol = self._curate_protein_peptide_models(row)
                mol.write(output_name)
            else:
                logger.info(f"Skipping {output_name}, already filtered...")
        df.loc[:, "filtered_path"] = filtered_names

        # 5. Analysis Loop per Binder
        binder_df = pd.DataFrame()

        for binder in self._candidate_clusters.Binder_name.unique():

            ppi_data = df.loc[self._candidate_clusters.Binder_name == binder,:].copy()
            all_structs = ppi_data.filtered_path.values

            if len(all_structs) == 0:
                logger.warning(f"No valid models for binder {binder} after filtering. Skipping...")
                continue

            # A. Get/Create Reference Structure
            reference_output_name = f"{output_path}/{binder}/reference_{binder}.pdb"
            if not os.path.exists(reference_output_name):
                reference_molecule = self._get_reference_structure_for_binder(all_structs)
                reference_molecule.write(reference_output_name)
            else:
                reference_molecule = Molecule(reference_output_name)
                logger.info(f"Skipping reference generation for {binder}, exists...")

            # B. Alignment Loop
            aligned_models = []
            for tmp_mol_name in all_structs:

                # Simple string replacement for path (assumes standard folder structure)
                tmp_mol_name_aligned = tmp_mol_name.replace("filtered", "aligned")

                if not os.path.exists(tmp_mol_name_aligned):
                    tmp_mol = self._create_binder_alignments(tmp_mol_name, reference_molecule)
                    tmp_mol.write(tmp_mol_name_aligned)
                else:
                    logger.debug(f"Skipping alignment for {tmp_mol_name_aligned}...")

                aligned_models.append(tmp_mol_name_aligned)

            # C. Clustering (DBSCAN)
            tmp_df, cluster_info = self.cluster_protein_peptides(aligned_models, reference_output_name, **kwargs)

            if tmp_df.empty or len(cluster_info.get("cluster_labels", [])) == 0:
                logger.warning(f"No spatial clusters found for {binder}. Skipping ChimeraX generation.")
                continue

            tmp_df.insert(0, "Binder", binder)
            binder_df = pd.concat([binder_df, tmp_df], ignore_index=True)

            # D. Visualization Preparation (ChimeraX)
            # Ensure lengths match before assignment
            labels = cluster_info.get("cluster_labels")
            centers = cluster_info.get("peptide_centers")

            if len(labels) == len(ppi_data):
                ppi_data.loc[:, "Cluster_info"] = labels
                ppi_data.loc[:, "Center_X"] = centers[:,0]
                ppi_data.loc[:, "Center_Y"] = centers[:,1]
                ppi_data.loc[:, "Center_Z"] = centers[:,2]

                self._create_chimera_session(ppi_data, reference_output_name, tmp_df)
            else:
                logger.error(f"Shape mismatch for {binder}: labels({len(labels)}) != data({len(ppi_data)}). Skipping.")

        # 6. Save Final Summary
        binder_df.to_csv(f"{self.output_path}/peptide_binder_info.csv", index=False)

    def _create_chimera_session(self,
                                ppi_data: pd.DataFrame,
                                ref_model: str,
                                cluster_info: pd.DataFrame):
        """
        Generates and executes a ChimeraX script (.cxc) to visualize the analysis.

        This method writes a set of ChimeraX commands to:
        1. Load the reference binder structure.
        2. Color the binder's interface residues according to the cluster they interact with.
        3. Load and align all peptide structures, colored by their cluster ID.
        4. Represent cluster centroids as spheres.
        5. Save the session as a .cxs file for easy reopening.

        Parameters
        ----------
        ppi_data : pd.DataFrame
            DataFrame containing details of the peptide models (paths, cluster IDs, etc.).
        ref_model : str
            Path to the reference PDB file of the binder.
        cluster_info : pd.DataFrame
            Summary DataFrame containing cluster labels, centroids, and interacting residues.
        """
        binder = ppi_data.Binder_name.values[0]

        # Use Pathlib for robust path handling
        base_dir = self.output_path / "prot_peptide" / binder
        script_path = base_dir / f"{binder}_peptide_binding.cxc"
        session_path = base_dir / f"{binder}_peptide_binding.cxs"

        available_colors = ["cyan", "yellow", "magenta", "orange", "cornflower blue"]
        available_colors_ref = ["light coral", "medium slate blue", "orange", "green", "red", "yellow"]

        with open(script_path, "w") as f:
            ## Global settings
            f.write("graphics silhouettes true\n")
            f.write("lighting soft\n")
            f.write("set bg white\n")

            ## Load reference
            f.write(f"\n# --- {binder} REFERENCE ---\n")
            f.write(f"open \"{ref_model}\"\n")
            f.write(f"rename #1 {binder}_ref\n")

            ## Color ref residues based on cluster interactions
            for idx, cluster_data in cluster_info.iterrows():
                cluster_id = int(cluster_data["Cluster_label"])
                if cluster_id == -1:
                    continue # Skip noise

                # Join residues for selection
                tmp_sel_str = ",".join(map(str, cluster_data["Residues"]))

                color_index = cluster_id % len(available_colors_ref)
                color_str = available_colors_ref[color_index]

                f.write(f"color #1:{tmp_sel_str} {color_str}\n")

                ## Draw centroid sphere for the cluster
                super_id = f"#5.{cluster_id + 1}"
                global_center_str = f"{cluster_data['Center_X']},{cluster_data['Center_Y']},{cluster_data['Center_Z']}"
                f.write(f"shape sphere name Centroid_{cluster_id+1}_Mean radius 3 center {global_center_str} color {color_str} model {super_id} \n")

            ## Load peptide_proteins and rename
            ppi_data["aligned_path"] = ppi_data["filtered_path"].str.replace("filtered", "aligned")

            for tmp_cluster in ppi_data["Cluster_info"].unique():
                tmp_cluster = int(tmp_cluster)
                current_sub_id = tmp_cluster + 2

                if tmp_cluster == -1:
                    color_str = "silver"
                    group_name = "Unclassified"

                else:
                    color_idx = tmp_cluster % len(available_colors)
                    color_str = available_colors[color_idx]
                    group_name = f"Cluster_{tmp_cluster + 1}"

                models_in_cluster = ppi_data.loc[ppi_data["Cluster_info"] == tmp_cluster, :]

                for i, (idx, row) in enumerate(models_in_cluster.iterrows()):
                    # Model ID hierarchy: #3.ClusterID.ModelIndex
                    pep_id = f"#3.{current_sub_id}.{i + 1}"
                    cen_id = f"#4.{current_sub_id}.{i + 1}"

                    # Open peptide
                    f.write(f"open \"{row['aligned_path']}\" id {pep_id}\n")

                    if tmp_cluster == -1:
                        model_name = f"{row['PPI']}_unclassified"
                    else:
                        model_name = f"{row['PPI']}_c{current_sub_id}"

                    f.write(f"rename {pep_id} {model_name}\n")
                    f.write(f"color {pep_id} {color_str}\n")

                    centroid_str = f"{row['Center_X']},{row['Center_Y']},{row['Center_Z']}"
                    f.write(f"shape sphere name {model_name} radius 1 center {centroid_str} color {color_str} model {cen_id}\n")
                f.write(f"rename #3.{current_sub_id} {group_name}\n")
                f.write(f"rename #4.{current_sub_id} {group_name}_Center_of_mass\n")


            ## Final cleanup and save
            f.write("lighting depthCue false\n")
            f.write("rename #3 Peptides\n")
            f.write("hide #3/A cartoon\n")  # Hide the Binder chain in the aligned peptide models
            f.write("hide atoms\n")
            f.write("rename #4 Peptide_centers\n")
            f.write(f"save {session_path}\n")
            f.write("exit\n")

        logger.info(f"Executing ChimeraX script: {script_path}")
        try:
            subprocess.run(["chimerax", "--nogui", str(script_path)], check=True)
        except FileNotFoundError:
            logger.error("ChimeraX executable not found in PATH. Script generated but not executed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ChimeraX execution failed: {e}")

    def cluster_protein_peptides(self, aligned_models: List[str],
                                 reference_model: Union[str, Molecule],
                                 **kwargs) -> tuple:
        """
        Clusters peptide structures based on their spatial position relative to the binder.

        This method uses the DBSCAN algorithm to group peptides that bind to similar
        regions on the reference protein surface. It calculates the centroid (Center of Mass)
        of the peptide backbones (Chain B, CA atoms) and clusters these points.

        Parameters
        ----------
        aligned_models : List[str]
            List of file paths to the aligned PDB models.
        reference_model : str or Molecule
            The reference binder structure used to map the binding sites.
        **kwargs
            Arguments passed to the DBSCAN constructor (e.g., `eps=5`, `min_samples=3`).

        Returns
        -------
        pd.DataFrame
            Summary table with columns:
            - 'Cluster_label': The ID assigned by DBSCAN (-1 indicates noise).
            - 'Center_X/Y/Z': Geometric center of the cluster.
            - 'Residues': List of binder residues within 8Å of the cluster center.
        dict
            Dictionary containing raw 'cluster_labels' and 'peptide_centers' arrays.
        """

        # 1. Load Molecules
        mols = [Molecule(str(i)) for i in aligned_models]

        # 2. Resolve Reference
        if isinstance(reference_model, str):
            ref_mol = Molecule(reference_model)
        elif isinstance(reference_model, Molecule):
            ref_mol = reference_model
        else:
            raise ValueError("reference_model must be a path (str) or Molecule object")

        # 3. Get Reference Geometry
        xyz = ref_mol.get("coords", sel="protein")
        ref_resids = ref_mol.get("resid", sel="protein")

        # 4. Calculate Peptide Centroids (Chain B, Alpha Carbons)

        valid_centroids = []
        valid_indices = []
        for i, tmp_mol in enumerate(mols):
            coords = tmp_mol.get("coords", sel="chain B and name CA")
            if coords.size > 0:
                valid_centroids.append(coords.mean(axis=0))
                valid_indices.append(i)
            else:
                logger.warning(f"Model {aligned_models[i]} has no Chain B CA atoms. Skipping from clustering.")

        if not valid_centroids:
            return pd.DataFrame(), {"cluster_labels": np.array([]), "peptide_centers": np.array([])}

        mols_centroids = np.array(valid_centroids)

        # 5. Perform Clustering (DBSCAN)
        # kwargs allows passing eps, min_samples, etc.
        clustering = DBSCAN(**kwargs).fit(mols_centroids)

        # Adjust cluster labels for original input list
        full_labels = np.full(len(mols), -1, dtype=int)
        for idx, label in zip(valid_indices, clustering.labels_):
            full_labels[idx] = label

        # 6. Analyze Clusters
        cluster_labels = []
        cluster_centers = []
        all_nearby_residues = []

        unique_labels = np.unique(clustering.labels_)

        for cluster_label in unique_labels:
            cluster_labels.append(cluster_label)

            # Calculate geometric center of the cluster (mean of peptide centroids)
            mask = clustering.labels_ == cluster_label
            cluster_center = mols_centroids[mask].mean(axis=0)
            cluster_centers.append(cluster_center)

            # 7. Identify Binding Site Residues
            # Calculate distance from every protein atom to the cluster center
            tmp_centroid_distance = xyz - cluster_center

            # Euclidean norm (distance)
            tmp_euc = np.linalg.norm(tmp_centroid_distance, axis = 1)

            nearby_residues = np.unique(ref_resids[tmp_euc < 8])
            all_nearby_residues.append(nearby_residues)

        # 8. Format Output
        cluster_centers = np.array(cluster_centers)

        results_df = pd.DataFrame({
            "Cluster_label": cluster_labels,
            "Center_X": cluster_centers[:, 0],
            "Center_Y": cluster_centers[:, 1],
            "Center_Z": cluster_centers[:, 2],
            "Residues": all_nearby_residues
        })

        extra_info = {
            "cluster_labels": full_labels,
            "peptide_centers": np.array([valid_centroids[valid_indices.index(i)] if i in valid_indices else [0,0,0] for i in range(len(mols))])
        }

        return results_df, extra_info
