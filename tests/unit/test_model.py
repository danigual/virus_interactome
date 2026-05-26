import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from virus_interactome.model import Model, Engine, ModelType, ModelMetrics, ModelData

DATA = Path(__file__).parent.parent / "data"
AF3_CIF    = DATA / "af3_dummy_example" / "fold_adv5_pvi_protease_model_0.cif"
BOLTZ_CIF  = DATA / "boltz_dummy_example" / "pvi__protease_model_0.cif"
CF_PDB     = DATA / "colabfold_dummy_example" / (
    "pVI__protease_unrelaxed_rank_001_alphafold2_multimer_v3_model_3_seed_000.pdb"
)


@pytest.fixture(scope="module")
def af3_model():
    return Model(AF3_CIF, engine="af3")


@pytest.fixture(scope="module")
def boltz_model():
    return Model(BOLTZ_CIF, engine="boltz")


@pytest.fixture(scope="module")
def cf_model():
    return Model(CF_PDB, engine="colabfold")


# ---------------------------------------------------------------------------
# Engine enum
# ---------------------------------------------------------------------------

class TestEngine:
    def test_af3_value(self):
        assert Engine("af3") == Engine.AF3

    def test_colabfold_value(self):
        assert Engine("colabfold") == Engine.COLABFOLD

    def test_boltz_value(self):
        assert Engine("boltz") == Engine.BOLTZ

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Engine("unknown_engine")


# ---------------------------------------------------------------------------
# Init — path parsing for all three engines
# ---------------------------------------------------------------------------

class TestModelInitAF3:
    def test_engine_set(self, af3_model):
        assert af3_model.engine == Engine.AF3

    def test_model_num(self, af3_model):
        assert af3_model.model_num == 0

    def test_model_path(self, af3_model):
        assert af3_model.model_path == Path(AF3_CIF)

    def test_id_is_string(self, af3_model):
        assert isinstance(af3_model.id, str)

    def test_metrics_loaded(self, af3_model):
        assert af3_model.metrics is not None
        assert af3_model.metrics.ca_plddts is not None
        assert len(af3_model.metrics.ca_plddts) > 0

    def test_model_data_loaded(self, af3_model):
        assert af3_model.model_data is not None
        assert af3_model.model_data.chain_boundaries_by_res is not None


class TestModelInitBoltz:
    def test_engine_set(self, boltz_model):
        assert boltz_model.engine == Engine.BOLTZ

    def test_id_parsed(self, boltz_model):
        assert "pvi__protease" in boltz_model.id

    def test_model_num(self, boltz_model):
        assert boltz_model.model_num == 0

    def test_pae_is_2d(self, boltz_model):
        pae = boltz_model.metrics.pae
        assert pae.ndim == 2
        assert pae.shape[0] == pae.shape[1]


class TestModelInitColabfold:
    def test_engine_set(self, cf_model):
        assert cf_model.engine == Engine.COLABFOLD

    def test_id_parsed(self, cf_model):
        assert cf_model.id == "pVI__protease"

    def test_model_num(self, cf_model):
        assert cf_model.model_num == 1

    def test_two_chains_detected(self, cf_model):
        chains = cf_model.model_data.chain_boundaries_by_res
        assert len(chains) == 2


# ---------------------------------------------------------------------------
# Shared properties and summary
# ---------------------------------------------------------------------------

class TestModelProperties:
    def test_pae_summary_has_all_mean(self, af3_model):
        s = af3_model.pae_summary
        assert "all_mean" in s.index
        assert isinstance(s["all_mean"], float)

    def test_plddt_summary_has_all_mean(self, af3_model):
        s = af3_model.plddt_summary
        assert "all_mean" in s.index
        assert 0 <= s["all_mean"] <= 100

    def test_plddt_summary_per_chain(self, af3_model):
        s = af3_model.plddt_summary
        # expect at least one chain-specific key
        chain_keys = [k for k in s.index if "_mean" in k and k != "all_mean"]
        assert len(chain_keys) >= 1

    def test_model_type_complex(self, af3_model):
        assert af3_model.model_type == ModelType.COMPLEX

    def test_model_type_explicit_monomer(self):
        m = Model(BOLTZ_CIF, engine="boltz", is_complex=False)
        assert m.model_type == ModelType.MONOMER

    def test_model_type_explicit_complex(self):
        m = Model(AF3_CIF, engine="af3", is_complex=True)
        assert m.model_type == ModelType.COMPLEX


class TestModelSummary:
    def test_summary_returns_series(self, af3_model):
        s = af3_model.summary()
        assert isinstance(s, pd.Series)

    def test_summary_keys(self, af3_model):
        s = af3_model.summary()
        for key in ("id", "model_num", "engine", "ptm", "iptm", "mean_plddt", "mean_pae", "path"):
            assert key in s.index

    def test_repr_is_string(self, af3_model):
        r = repr(af3_model)
        assert isinstance(r, str)
        assert len(r) > 0


# ---------------------------------------------------------------------------
# _build_chimerax_script (pure string building — no launch)
# ---------------------------------------------------------------------------

class TestBuildChimeraXScript:
    def test_plddt_mode(self, af3_model):
        script = af3_model._build_chimerax_script("plddt")
        assert "color bfactor" in script
        assert "cartoons" in script

    def test_chain_mode(self, af3_model):
        script = af3_model._build_chimerax_script("chain")
        assert "rainbow chain" in script

    def test_surface_mode(self, af3_model):
        script = af3_model._build_chimerax_script("surface")
        assert "surface" in script

    def test_script_contains_model_path(self, af3_model):
        script = af3_model._build_chimerax_script("plddt")
        assert str(af3_model.model_path) in script
