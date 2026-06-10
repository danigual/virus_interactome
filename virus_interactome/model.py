from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import numpy as np
from moleculekit.molecule import Molecule
from virus_interactome.utils import load_json
import pandas as pd

# ── Supporting types ──────────────────────────────────────────────────────────

class Engine(Enum):
    AF3 = "af3"
    COLABFOLD = "colabfold"
    BOLTZ = "boltz"

class ModelType(Enum):
    MONOMER = auto()
    COMPLEX = auto()

@dataclass(frozen=True)
class ModelMetrics:
    pae: np.ndarray | None = None
    ptm: float | None = None
    # complex-only — None for monomers
    iptm: float | None = None
    iptm_chain_pair: np.ndarray | None = None
    atom_plddts: np.ndarray | None = None
    cb_plddts: np.ndarray | None = None
    ca_plddts: np.ndarray | None = None

@dataclass(frozen=True)
class ModelData:
    token_chain_ids: np.ndarray | None = None
    chain_boundaries_by_res: dict[str, tuple[int, int]] | None = None
    chain_boundaries_by_atom: dict[str, tuple[int, int]] | None = None

# ── Main class ────────────────────────────────────────────────────────────────
class Model:
    """
    Represents a single structure prediction (monomer or complex)
    from AF3, ColabFold, or Boltz.
    """

    def __init__(
        self,
        model_path: str | Path,
        engine: Engine | str = Engine.COLABFOLD,
        extra_files: dict | None = None,   # e.g. {"scores": "path/to/scores.json"}
        is_complex: bool | None = None,    # None → infer from structure
    ):
        self._engine = Engine(engine.lower()) if isinstance(engine, str) else engine
        self._model_path = Path(model_path)
        self._id = self._get_id_from_path()
        self._model_num = self._get_model_num_from_path()
        # auto-resolve extra files, then let explicit dict override
        self._extra_files = self._resolve_extra_files()
        if extra_files:
            self._extra_files.update(extra_files)

        # ── lazy: molecule is expensive (full structure parse) ────────────
        # these stay None until the property is first accessed
        self._molecule: Molecule | None = None
        self._plddt: np.ndarray | None = None
        self._chain: np.ndarray | None = None
        self._resname: np.ndarray | None = None
        self._resid: np.ndarray | None = None

        # ── eager: metrics are cheap (just JSON parsing) ──────────────────
        self._metrics, self._model_data = self._load_data()

        # ── model type: infer if not given ────────────────────────────────
        # we defer to the property so it can fall back to the structure
        self._model_type: ModelType | None = (
            ModelType.COMPLEX if is_complex
            else ModelType.MONOMER if is_complex is False
            else None          # None means "infer from molecule when needed"
        )

    # ── lazy properties ───────────────────────────────────────────────────────

    @property
    def molecule(self) -> Molecule:
        """Load and cache the Molecule object on first access."""
        if self._molecule is None:
            self._molecule = Molecule(str(self._model_path))
        return self._molecule

    @property
    def plddt(self) -> np.ndarray:
        """Per-residue pLDDT array, loaded lazily from the molecule."""
        if self._plddt is None:
            # moleculekit stores B-factor (where AF puts pLDDT) in .beta
            self._plddt = self.molecule.beta  # triggers molecule load if needed
        return self._plddt
    
    @property
    def plddt_summary(self) -> pd.Series:
        result = {}
        ca = self._metrics.ca_plddts
        result["all_mean"]   = float(ca.mean())
        result["all_median"] = float(np.median(ca))

        for chain_id, (start, end) in self._model_data.chain_boundaries_by_res.items():
            chain_plddt = ca[start:end+1]
            result[f"{chain_id}_mean"]   = float(chain_plddt.mean())
            result[f"{chain_id}_median"] = float(np.median(chain_plddt))

        return pd.Series(result)

    @property
    def pae_summary(self) -> pd.Series:
        result = {}
        pae = self._metrics.pae
        result["all_mean"] = float(pae.mean())

        for ci, (si, ei) in self._model_data.chain_boundaries_by_res.items():
            for cj, (sj, ej) in self._model_data.chain_boundaries_by_res.items():
                block = pae[si:ei+1, sj:ej+1]
                result[f"{ci}{cj}_mean"] = float(block.mean())

        return pd.Series(result)

    @property
    def chain(self) -> np.ndarray:
        if self._chain is None:
            self._chain = self.molecule.chain
        return self._chain
    
    @property
    def resname(self) -> np.ndarray:
        if self._resname is None:
            self._resname = self.molecule.resname
        return self._resname

    @property
    def resid(self) -> np.ndarray:
        if self._resid is None:
            self._resid = self.molecule.resid
        return self._resid

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    @property
    def model_data(self) -> ModelData:
        return self._model_data

    @property
    def model_type(self) -> ModelType:
        if self._model_type is None:
            # infer: more than one chain → complex
            n_chains = len(np.unique(self.molecule.chain))
            self._model_type = (
                ModelType.COMPLEX if n_chains > 1 else ModelType.MONOMER
            )
        return self._model_type
    
    # ── convenience accessors ─────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def model_num(self) -> int | None:
        return self._model_num
    
    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def model_path(self) -> Path:
        return self._model_path

    # ── public methods ────────────────────────────────────────────────────────
    def view(self, mode: str = "plddt", launch: bool = True) -> Path:
        """
        Write a ChimeraX script and optionally launch it.

        Parameters
        ----------
        mode : "plddt" | "chain" | "surface"
        launch : if True, subprocess.run(["chimerax", script_path])
        """
        script = self._build_chimerax_script(mode)
        script_path = self._model_path.parent / "visualize.cxc"
        script_path.write_text(script)
        if launch:
            import subprocess
            subprocess.run(["chimerax", str(script_path)])
        return script_path
    
    def _build_chimerax_script(self, mode: str) -> str:
        lines = [f'open {self._model_path}']
        if mode == "plddt":
            lines += ["color bfactor palette alphafold", "show cartoons"]
        elif mode == "chain":
            lines += ["rainbow chain", "show cartoons"]
        elif mode == "surface":
            lines += ["surface", "color bfactor palette alphafold"]
        
        lines += ["hide atoms; show cartoon"]
        lines += [f"alphafold pae #1 file {self._extra_files['scores']} palette paegreen"]
        return "\n".join(lines)

    def summary(self) -> str:
        return pd.Series({
            "id":          self._id,
            "model_num":   self._model_num,
            "engine":      self._engine.value,
            "ptm":         self._metrics.ptm,
            "iptm":        self._metrics.iptm,
            "mean_plddt":  self._metrics.ca_plddts.mean().round(2),
            "mean_pae":    self._metrics.pae.mean().round(2),
            "path":        str(self._model_path.resolve()),
        })

    def calculate_ipsae(self, pae_cutoff: float = 10.0) -> pd.DataFrame | None:
        """Inter-chain ipSAE table, if available (complex-only)."""
        if self.model_type == ModelType.MONOMER:
            return None
        if self._metrics.pae is None or self._model_data.chain_boundaries_by_res is None:
            return None

        from .metrics import calculate_ipsae
        return calculate_ipsae(self._molecule, self._metrics.pae, pae_cutoff)
    
    def __repr__(self) -> str:
        return self.summary().to_string()

    # ── private helpers ───────────────────────────────────────────────────────
    def _get_id_from_path(self) -> str:
        if self._engine == Engine.AF3:
            return self._model_path.stem.split("_model_")[0].split("fold_")[0]
        elif self._engine == Engine.COLABFOLD:
            return self._model_path.stem.split("_unrelaxed_")[0].split("_relaxed")[0]
        elif self._engine == Engine.BOLTZ:
            return self._model_path.stem.split("_model_")[0]

    def _get_model_num_from_path(self) -> int | None:
        if self._engine == Engine.AF3:
            return int(self._model_path.stem.split("_model_")[1].split("_")[0])
        elif self._engine == Engine.COLABFOLD:
            return int(self._model_path.stem.split("_rank_")[1].split("_")[0])
        elif self._engine == Engine.BOLTZ:
            return int(self._model_path.stem.split("_model_")[1].split(".")[0])

    def _resolve_extra_files(self) -> dict[str, Path]:
        """
        Auto-guess companion files based on engine conventions.
        Each engine stores scores in a predictable location.
        """
        parent = self._model_path.parent
        stem = self._model_path.stem
        if self._engine == Engine.AF3:
            score_stem = stem.replace("_model_", "_full_data_")
            summary_stem = stem.replace("_model_", "_summary_confidences_")
            return {
                "scores": parent / f"{score_stem}.json",
                "summary": parent / f"{summary_stem}.json",
            }
        elif self._engine == Engine.COLABFOLD:
            stem = stem.replace("unrelaxed", "scores").replace("relaxed", "scores")  # e.g. "myprotein_scores_rank_001_..."
            return {
                "scores": parent / f"{stem}.json",
            }
        elif self._engine == Engine.BOLTZ:
            return {
                "scores": parent / f"confidence_{stem}.json",
                "pae":    parent / f"pae_{stem}.npz",
                "plddt":  parent / f"plddt_{stem}.npz",
                "pde":    parent / f"pde_{stem}.npz",
            }

    def _load_data(self) -> tuple[ModelMetrics, ModelData]:
        """Dispatch to the right parser based on engine."""
        parsers = {
            Engine.AF3:       self._load_af3_metrics,
            Engine.COLABFOLD: self._load_colabfold_metrics,
            Engine.BOLTZ:     self._load_boltz_metrics,
        }
        return parsers[self._engine]()

    def _load_af3_metrics(self) -> ModelMetrics:
        full_data = load_json(self._extra_files["scores"])
        summary_data = load_json(self._extra_files["summary"])
        token_chain_ids = np.array(full_data.get("token_chain_ids"))
        atom_chain_ids = np.array(full_data.get("atom_chain_ids"))
        
        mol = self.molecule
        chain_by_res = mol.chain[mol.name == "CA"]
        ca_mask = mol.name == "CA"
        cb_mask = np.logical_or(mol.name == "CB", np.logical_and(mol.resname == "GLY",  mol.name == "CA"))
        cb_plddt = np.array(full_data.get("atom_plddts"))[cb_mask]
        ca_plddt = np.array(full_data.get("atom_plddts"))[ca_mask]
        
        chain_boundaries = {}
        chain_boundaries_by_atom = {}
        for chain_id in np.unique(token_chain_ids):
            chain_indexes = np.where(np.array(chain_by_res) == chain_id)
            chain_boundaries[str(chain_id)] = (int(np.min(chain_indexes)), int(np.max(chain_indexes)))
            
            atom_chain_indexes = np.where(atom_chain_ids == chain_id)
            chain_boundaries_by_atom[str(chain_id)] = (np.min(atom_chain_indexes), np.max(atom_chain_indexes))

        ## Convert pae, atom_plddts and contact_probs to np arrays
        return ModelMetrics(
            pae = np.array(full_data.get("pae", 100)),
            ptm = summary_data.get("ptm", 0),
            iptm = summary_data.get("iptm", 0),
            iptm_chain_pair = np.array(summary_data.get("chain_pair_iptm")),
            atom_plddts = np.array(full_data.get("atom_plddts")),
            cb_plddts = np.array(cb_plddt),
            ca_plddts = np.array(ca_plddt)) , ModelData(
            token_chain_ids = token_chain_ids,
            chain_boundaries_by_res = chain_boundaries,
            chain_boundaries_by_atom = chain_boundaries_by_atom
        )

    def _load_colabfold_metrics(self) -> ModelMetrics:
        mol = self.molecule
        full_data = load_json(self._extra_files["scores"])

        # CA-level arrays
        chain_by_res = mol.chain[mol.name == "CA"]
        pae_data = np.array(full_data["pae"])          # already in Angstroms
        plddt_data = np.array(full_data["plddt"])       # already 0-100

        n_res = len(chain_by_res)
        if pae_data.shape[0] != n_res:
            raise ValueError(
                f"PAE size ({pae_data.shape[0]}) != CA atom count ({n_res})"
            )
        if len(plddt_data) != n_res:
            raise ValueError(
                f"pLDDT length ({len(plddt_data)}) != CA atom count ({n_res})"
            )

        # Chain boundaries
        from typing import Dict, Any
        chain_boundaries: Dict[str, Any] = {}
        chain_boundaries_by_atom: Dict[str, Any] = {}
        unique_chains = np.unique(chain_by_res)

        for chain_id in unique_chains:
            res_idxs = np.where(chain_by_res == chain_id)[0]
            chain_boundaries[chain_id] = (int(res_idxs.min()), int(res_idxs.max()))
            atom_idxs = np.where(mol.chain == chain_id)[0]
            chain_boundaries_by_atom[chain_id] = (int(atom_idxs.min()), int(atom_idxs.max()))

        # ColabFold does not provide per-chain iptm; estimate manually from PAE
        n_chains = len(unique_chains)
        iptm_chain_pair = np.zeros((n_chains, n_chains))

        from .metrics import calc_d0, ptm_func_vec
        for i, chain_i in enumerate(unique_chains):
            for j, chain_j in enumerate(unique_chains):
                mask_i = (chain_by_res == chain_i)
                mask_j = (chain_by_res == chain_j)
                
                # Select PAE submatrix for these two chains
                pae_block = pae_data[mask_i][:, mask_j]
                
                # Calculate d0 based on chain lengths
                # For pTM (diagonal i==j), use single chain length.
                # For ipTM (off-diagonal i!=j), use sum of lengths.
                L_eff = np.sum(mask_i) + np.sum(mask_j) if i != j else np.sum(mask_i)
                d0 = calc_d0(L_eff, 'protein')
                
                # Calculate mean pTM/ipTM score for this pair/chain
                iptm_chain_pair[i, j] = np.mean(ptm_func_vec(pae_block, d0))

        return ModelMetrics(
                pae = np.array(full_data.get("pae", 100)),
                ptm = full_data.get("ptm", 0),
                iptm = full_data.get("iptm", iptm_chain_pair[0,0]),
                iptm_chain_pair = iptm_chain_pair,
                atom_plddts = np.array(mol.beta),
                # atom_plddts = np.array(full_data.get("atom_plddts")),
                ca_plddts = np.array(plddt_data), # CA used as proxy (no separate CB file)
                cb_plddts = np.array(plddt_data),
                ), ModelData(token_chain_ids = chain_by_res,
                chain_boundaries_by_res = chain_boundaries,
                chain_boundaries_by_atom = chain_boundaries_by_atom
            )
    
    def _load_boltz_metrics(self) -> tuple[ModelMetrics, ModelData]:
        mol = self.molecule
        confidence = load_json(self._extra_files["scores"])
        pae_data   = np.load(self._extra_files["pae"])["pae"]
        plddt_data = np.load(self._extra_files["plddt"])["plddt"] * 100  # 0–1 → 0–100

        chain_by_res = mol.chain[mol.name == "CA"]
        unique_chains = np.unique(chain_by_res)

        chain_boundaries = {}
        chain_boundaries_by_atom = {}
        for chain_id in unique_chains:
            res_idxs  = np.where(chain_by_res == chain_id)[0]
            atom_idxs = np.where(mol.chain == chain_id)[0]
            chain_boundaries[chain_id]         = (int(res_idxs.min()), int(res_idxs.max()))
            chain_boundaries_by_atom[chain_id] = (int(atom_idxs.min()), int(atom_idxs.max()))

        n_chains = len(unique_chains)
        iptm_chain_pair = np.zeros((n_chains, n_chains))
        pair_iptm = confidence.get("pair_chains_iptm", {})
        for i in range(n_chains):
            for j in range(n_chains):
                iptm_chain_pair[i, j] = pair_iptm.get(str(i), {}).get(str(j), 0.0)

        return ModelMetrics(
            pae             = pae_data,
            ptm             = confidence.get("ptm", 0),
            iptm            = confidence.get("iptm", 0),
            iptm_chain_pair = iptm_chain_pair,
            atom_plddts     = np.array(mol.beta),
            ca_plddts       = plddt_data,
            cb_plddts       = plddt_data,
        ), ModelData(
            token_chain_ids          = chain_by_res,
            chain_boundaries_by_res  = chain_boundaries,
            chain_boundaries_by_atom = chain_boundaries_by_atom,
        )

