"""Phase 7 — InteractomeRunner tests using mock filesystem structures."""

import json
import pytest
import pandas as pd
import yaml
from pathlib import Path
from virus_interactome.interactome_runner import InteractomeRunner


# ---------------------------------------------------------------------------
# Fixtures — mock directory structures
# ---------------------------------------------------------------------------

def _write_af3_job(input_dir: Path, name: str, sequences: list) -> Path:
    """Writes a single AF3 input JSON and returns the path."""
    job = {"name": name, "sequences": sequences}
    path = input_dir / f"{name}.json"
    path.write_text(json.dumps(job))
    return path


def _write_boltz_job(input_dir: Path, name: str, sequences: list) -> Path:
    """Writes a single Boltz2 input YAML and returns the path."""
    yaml_data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": [chr(65 + i)], "sequence": s["proteinChain"]["sequence"]}}
            for i, s in enumerate(sequences)
        ],
    }
    path = input_dir / f"{name}.yaml"
    path.write_text(yaml.dump(yaml_data))
    return path


def _make_cif_outputs(output_dir: Path, name: str, count: int):
    """Creates empty CIF files to simulate completed model outputs."""
    job_dir = output_dir / name
    job_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (job_dir / f"fold_model_{i}.cif").touch()


def _make_pdb_outputs(output_dir: Path, name: str, count: int):
    """Creates empty PDB files to simulate ColabFold outputs."""
    job_dir = output_dir / name
    job_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (job_dir / f"{name}_rank_{i:03d}_model_{i+1}_seed_000.pdb").touch()


SEQS_AB = [
    {"proteinChain": {"sequence": "MKTAYIAK", "count": 1}},
    {"proteinChain": {"sequence": "GVALSKGE", "count": 1}},
]
SEQS_CD = [
    {"proteinChain": {"sequence": "LLKSDGQV", "count": 1}},
    {"proteinChain": {"sequence": "MKQLQKDL", "count": 2}},
]


@pytest.fixture
def af3_dirs(tmp_path):
    """AF3 input/output directory pair with 2 jobs."""
    inp = tmp_path / "af3_input"
    out = tmp_path / "af3_output"
    inp.mkdir()
    _write_af3_job(inp, "ProtA__ProtB", SEQS_AB)
    _write_af3_job(inp, "ProtC__ProtD", SEQS_CD)
    return inp, out


@pytest.fixture
def boltz_dirs(tmp_path):
    """Boltz2 input/output directory pair with 2 jobs."""
    inp = tmp_path / "boltz_input"
    out = tmp_path / "boltz_output"
    inp.mkdir()
    _write_boltz_job(inp, "ProtA__ProtB", SEQS_AB)
    _write_boltz_job(inp, "ProtC__ProtD", SEQS_CD)
    return inp, out


@pytest.fixture
def colabfold_dirs(tmp_path):
    """ColabFold input/output directory pair with 2 jobs."""
    inp = tmp_path / "cf_input"
    out = tmp_path / "cf_output"
    inp.mkdir()
    (inp / "ProtA__ProtB.fasta").write_text(">ProtA__ProtB\nMKTAYIAK:GVALSKGE\n")
    (inp / "ProtC__ProtD.fasta").write_text(">ProtC__ProtD\nLLKSDGQV:MKQLQKDL\n")
    return inp, out


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestRunnerInit:
    def test_init_af3(self, af3_dirs):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        assert runner.mode == "af3"
        assert len(runner.inputs) == 2
        assert runner.output_dir.exists()

    def test_init_boltz2(self, boltz_dirs):
        inp, out = boltz_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="boltz2")
        assert runner.mode == "boltz2"
        assert len(runner.inputs) == 2

    def test_init_colabfold(self, colabfold_dirs):
        inp, out = colabfold_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        assert runner.mode == "colabfold"
        assert len(runner.inputs) == 2

    def test_invalid_mode_raises(self, af3_dirs):
        inp, out = af3_dirs
        with pytest.raises(ValueError, match="invalid"):
            InteractomeRunner(str(inp), str(out), mode="rosetta")

    def test_missing_input_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InteractomeRunner(str(tmp_path / "nonexistent"), str(tmp_path / "out"))

    def test_output_dir_created(self, af3_dirs):
        inp, out = af3_dirs
        assert not out.exists()
        InteractomeRunner(str(inp), str(out), mode="af3")
        assert out.exists()


# ---------------------------------------------------------------------------
# check_run — AF3
# ---------------------------------------------------------------------------

class TestCheckRunAF3:
    def test_all_failed_no_outputs(self, af3_dirs):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        assert len(df) == 2
        assert (df["status"] == "FAILED").all()

    def test_all_completed(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 10)
        _make_cif_outputs(out, "ProtC__ProtD", 10)
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        assert (df["status"] == "COMPLETED").all()

    def test_partial_running(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 5)  # < 10 expected
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        row = df[df["PPI"] == "ProtA__ProtB"].iloc[0]
        assert row["status"] == "RUNNING"

    def test_mixed_statuses(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 10)  # COMPLETED
        # ProtC__ProtD has no output → FAILED
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        completed = df[df["status"] == "COMPLETED"]
        failed = df[df["status"] == "FAILED"]
        assert len(completed) == 1
        assert len(failed) == 1

    def test_custom_expected_models(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 3)
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.check_run(expected_models=3)
        row = df[df["PPI"] == "ProtA__ProtB"].iloc[0]
        assert row["status"] == "COMPLETED"

    def test_chain_and_residue_counts(self, af3_dirs):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        row_ab = df[df["PPI"] == "ProtA__ProtB"].iloc[0]
        assert row_ab["num_chain"] == 2  # 1 + 1
        assert row_ab["num_aa"] == 16    # 8 + 8
        row_cd = df[df["PPI"] == "ProtC__ProtD"].iloc[0]
        assert row_cd["num_chain"] == 3  # 1 + 2
        assert row_cd["num_aa"] == 24    # 8 + 8*2

    def test_status_is_categorical_sorted(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 10)
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        df = runner.status
        # FAILED should come before COMPLETED
        statuses = df["status"].tolist()
        assert statuses[0] == "FAILED"
        assert statuses[-1] == "COMPLETED"


# ---------------------------------------------------------------------------
# check_run — Boltz2
# ---------------------------------------------------------------------------

class TestCheckRunBoltz:
    def test_all_completed_boltz(self, boltz_dirs):
        inp, out = boltz_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 5)
        _make_cif_outputs(out, "ProtC__ProtD", 5)
        runner = InteractomeRunner(str(inp), str(out), mode="boltz2")
        df = runner.status
        assert (df["status"] == "COMPLETED").all()

    def test_all_failed_boltz(self, boltz_dirs):
        inp, out = boltz_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="boltz2")
        assert (runner.status["status"] == "FAILED").all()


# ---------------------------------------------------------------------------
# check_colabfold_run
# ---------------------------------------------------------------------------

class TestCheckColabfoldRun:
    def test_all_completed(self, colabfold_dirs):
        inp, out = colabfold_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_pdb_outputs(out, "ProtA__ProtB", 5)
        _make_pdb_outputs(out, "ProtC__ProtD", 5)
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        df = runner.status
        assert (df["status"] == "COMPLETED").all()

    def test_all_failed(self, colabfold_dirs):
        inp, out = colabfold_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        assert (runner.status["status"] == "FAILED").all()

    def test_partial_running(self, colabfold_dirs):
        inp, out = colabfold_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_pdb_outputs(out, "ProtA__ProtB", 3)  # < 5 expected
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        df = runner.status
        row = df[df["PPI"] == "ProtA__ProtB"].iloc[0]
        assert row["status"] == "RUNNING"

    def test_no_inputs_uses_output_dirs(self, tmp_path):
        """When no FASTA inputs exist, job names are derived from output subdirs."""
        inp = tmp_path / "empty_input"
        out = tmp_path / "cf_out"
        inp.mkdir()
        out.mkdir()
        _make_pdb_outputs(out, "JobX", 5)
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        df = runner.status
        assert len(df) == 1
        assert df.iloc[0]["PPI"] == "JobX"
        assert df.iloc[0]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------

class TestWriteStatus:
    def test_default_path(self, af3_dirs):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        runner.write_status()
        csv_path = inp / "JOB_STATUS.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "PPI" in df.columns
        assert "status" in df.columns
        assert len(df) == 2

    def test_custom_path(self, af3_dirs, tmp_path):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        custom = tmp_path / "custom_status.csv"
        runner.write_status(file_name=str(custom))
        assert custom.exists()

    def test_update_flag(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        assert (runner.status["status"] == "FAILED").all()
        # Now add outputs and update
        _make_cif_outputs(out, "ProtA__ProtB", 10)
        runner.write_status(update=True)
        csv_path = inp / "JOB_STATUS.csv"
        df = pd.read_csv(csv_path)
        assert (df.loc[df["PPI"] == "ProtA__ProtB", "status"] == "COMPLETED").all()


# ---------------------------------------------------------------------------
# write_missing_jobs
# ---------------------------------------------------------------------------

class TestWriteMissingJobs:
    def test_copies_failed_inputs(self, af3_dirs):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        missing_dir = inp.parent / "input_missing"
        runner.write_missing_jobs()
        assert missing_dir.exists()
        copied = list(missing_dir.glob("*.json"))
        assert len(copied) == 2  # Both jobs are FAILED

    def test_copies_to_custom_path(self, af3_dirs, tmp_path):
        inp, out = af3_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        custom = tmp_path / "my_missing"
        runner.write_missing_jobs(output_path=str(custom))
        assert custom.exists()
        assert len(list(custom.glob("*.json"))) == 2

    def test_no_missing_when_all_completed(self, af3_dirs):
        inp, out = af3_dirs
        out.mkdir(parents=True, exist_ok=True)
        _make_cif_outputs(out, "ProtA__ProtB", 10)
        _make_cif_outputs(out, "ProtC__ProtD", 10)
        runner = InteractomeRunner(str(inp), str(out), mode="af3")
        missing_dir = inp.parent / "input_missing"
        runner.write_missing_jobs()
        # Directory may or may not be created, but no files should exist
        if missing_dir.exists():
            assert len(list(missing_dir.glob("*.json"))) == 0

    def test_boltz_copies_yaml(self, boltz_dirs):
        inp, out = boltz_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="boltz2")
        runner.write_missing_jobs()
        missing_dir = inp.parent / "input_missing"
        assert len(list(missing_dir.glob("*.yaml"))) == 2


# ---------------------------------------------------------------------------
# _build_colabfold_command (static method)
# ---------------------------------------------------------------------------

class TestBuildColabfoldCommand:
    def test_basic_command(self):
        cmd = InteractomeRunner._build_colabfold_command(
            colabfold_bin="colabfold_batch",
            num_recycle=3, num_models=5, model_order="1,2,3,4,5",
            amber=True, templates=True, use_gpu_relax=True,
            random_seed=42, extra_flags=[],
        )
        assert cmd[0] == "colabfold_batch"
        assert "--num-recycle" in cmd
        assert "3" in cmd
        assert "--amber" in cmd
        assert "--templates" in cmd
        assert "--use-gpu-relax" in cmd
        assert "--random-seed" in cmd
        assert "42" in cmd

    def test_no_amber_no_templates(self):
        cmd = InteractomeRunner._build_colabfold_command(
            colabfold_bin="cf", num_recycle=1, num_models=3,
            model_order="1,2,3", amber=False, templates=False,
            use_gpu_relax=False, random_seed=0, extra_flags=["--custom"],
        )
        assert "--amber" not in cmd
        assert "--templates" not in cmd
        assert "--use-gpu-relax" not in cmd
        assert "--custom" in cmd

    def test_extra_flags_appended(self):
        cmd = InteractomeRunner._build_colabfold_command(
            colabfold_bin="cf", num_recycle=1, num_models=1,
            model_order="1", amber=False, templates=False,
            use_gpu_relax=False, random_seed=0,
            extra_flags=["--msa-mode", "single_sequence"],
        )
        assert "--msa-mode" in cmd
        assert "single_sequence" in cmd


# ---------------------------------------------------------------------------
# _run_single_colabfold_job — dry_run mode only (no real subprocess)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# check_colabfold_run — edge cases
# ---------------------------------------------------------------------------

class TestCheckColabfoldRunEdge:
    def test_empty_dirs_returns_empty_df(self, tmp_path):
        """No inputs and no output subdirs → empty DataFrame."""
        inp = tmp_path / "empty_in"
        out = tmp_path / "empty_out"
        inp.mkdir()
        out.mkdir()
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        df = runner.status
        assert len(df) == 0
        assert "PPI" in df.columns

    def test_relaxed_pdb_fallback(self, tmp_path):
        """When *rank_*_model_*.pdb missing, falls back to *relaxed*.pdb."""
        inp = tmp_path / "cf_in"
        out = tmp_path / "cf_out"
        inp.mkdir()
        (inp / "JobX.fasta").write_text(">JobX\nAAAA\n")
        job_dir = out / "JobX"
        job_dir.mkdir(parents=True)
        # Create relaxed PDB files (not rank pattern)
        for i in range(5):
            (job_dir / f"JobX_relaxed_rank_{i:03d}.pdb").touch()
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        row = runner.status[runner.status["PPI"] == "JobX"].iloc[0]
        assert row["status"] == "COMPLETED"


class TestWriteMissingJobsSourceMissing:
    def test_source_file_not_found_logs_error(self, tmp_path, caplog):
        """Line 1237: logs error when source input file is absent from input_dir."""
        import logging
        caplog.set_level(logging.ERROR)
        inp = tmp_path / "inp"
        out = tmp_path / "out"
        inp.mkdir()
        # Runner whose status reports a FAILED job but no .json exists for it
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = inp
        runner.output_dir = out
        runner.mode = "af3"
        runner.inputs = []
        runner.status = pd.DataFrame({"PPI": ["Ghost__Job"], "status": ["FAILED"]})
        missing_dir = tmp_path / "missing"
        runner.write_missing_jobs(output_path=str(missing_dir))
        assert "Source file not found" in caplog.text
        # No file should have been copied
        assert len(list(missing_dir.glob("*.json"))) == 0


class TestRequeueMissingJobsEdge:
    def test_empty_status_warns(self, tmp_path, caplog):
        """write_missing_jobs logs warning and returns when status is None."""
        import logging
        caplog.set_level(logging.WARNING)
        inp = tmp_path / "inp"
        out = tmp_path / "out"
        inp.mkdir()
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = inp
        runner.output_dir = out
        runner.mode = "af3"
        runner.inputs = []
        runner.status = None
        runner.write_missing_jobs()
        assert "empty" in caplog.text.lower() or "Status" in caplog.text

    def test_empty_df_status_warns(self, tmp_path, caplog):
        """write_missing_jobs logs warning when status is empty DataFrame."""
        import logging
        caplog.set_level(logging.WARNING)
        inp = tmp_path / "inp"
        out = tmp_path / "out"
        inp.mkdir()
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = inp
        runner.output_dir = out
        runner.mode = "af3"
        runner.inputs = []
        runner.status = pd.DataFrame()
        runner.write_missing_jobs()
        assert "empty" in caplog.text.lower() or "Status" in caplog.text


class TestRunColabfoldCsv:
    def test_missing_csv_raises(self, tmp_path):
        """run_colabfold_csv raises FileNotFoundError for missing CSV."""
        inp = tmp_path / "inp"
        out = tmp_path / "out"
        inp.mkdir()
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = inp
        runner.output_dir = out
        runner.mode = "colabfold"
        with pytest.raises(FileNotFoundError, match="not found"):
            runner.run_colabfold_csv(str(tmp_path / "missing.csv"), str(out))

    def test_csv_dry_run(self, tmp_path):
        """run_colabfold_csv in dry_run mode returns without subprocess."""
        inp = tmp_path / "inp"
        out = tmp_path / "out"
        inp.mkdir()
        csv = tmp_path / "batch.csv"
        csv.write_text("id,sequence\ntest,AAAA\n")
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = inp
        runner.output_dir = out
        runner.mode = "colabfold"
        result = runner.run_colabfold_csv(str(csv), str(out), dry_run=True)
        assert result["status"] == "DRY_RUN"
        assert result["returncode"] == 0


class TestRunSingleColabfoldJobMocked:
    def test_failed_subprocess(self, tmp_path):
        """Non-zero returncode yields FAILED status."""
        from unittest.mock import patch, MagicMock
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">test\nAAAA\n")
        out = tmp_path / "out"
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            result = InteractomeRunner._run_single_colabfold_job(
                name="fail_job", fasta_path=fasta, output_dir=out,
                base_cmd=["colabfold_batch"],
            )
        assert result["status"] == "FAILED"
        assert result["returncode"] == 1

    def test_missing_binary_raises(self, tmp_path):
        """FileNotFoundError raised when colabfold_batch not found."""
        from unittest.mock import patch
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">test\nAAAA\n")
        out = tmp_path / "out"
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                InteractomeRunner._run_single_colabfold_job(
                    name="missing", fasta_path=fasta, output_dir=out,
                    base_cmd=["colabfold_batch"],
                )

    def test_successful_subprocess(self, tmp_path):
        """Returncode 0 yields COMPLETED status."""
        from unittest.mock import patch, MagicMock
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">test\nAAAA\n")
        out = tmp_path / "out"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("subprocess.run", return_value=mock_proc):
            result = InteractomeRunner._run_single_colabfold_job(
                name="ok_job", fasta_path=fasta, output_dir=out,
                base_cmd=["echo"],
            )
        assert result["status"] == "COMPLETED"
        assert result["returncode"] == 0


class TestRunColabfoldFastas:
    def test_dry_run_sequential(self, colabfold_dirs):
        """L1293-1318: runs pending jobs sequentially in dry_run mode."""
        inp, out = colabfold_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        results = runner.run_colabfold_fastas(dry_run=True)
        assert len(results) == 2
        assert all(r["status"] == "DRY_RUN" for r in results)
        assert all(r["returncode"] == 0 for r in results)

    def test_dry_run_parallel(self, colabfold_dirs):
        """L1319-1326: ThreadPoolExecutor branch with max_workers=2."""
        inp, out = colabfold_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        results = runner.run_colabfold_fastas(dry_run=True, max_workers=2)
        assert len(results) == 2
        assert all(r["status"] == "DRY_RUN" for r in results)

    def test_all_completed_returns_empty(self, colabfold_dirs):
        """L1296-1298: all jobs completed → returns [] immediately."""
        inp, out = colabfold_dirs
        out.mkdir()
        _make_pdb_outputs(out, "ProtA__ProtB", 5)
        _make_pdb_outputs(out, "ProtC__ProtD", 5)
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        results = runner.run_colabfold_fastas()
        assert results == []

    def test_missing_fasta_skipped(self, colabfold_dirs, caplog):
        """L1305-1307: job in status with no FASTA → skipped with warning."""
        import logging
        from unittest.mock import patch
        caplog.set_level(logging.WARNING)
        inp, out = colabfold_dirs
        runner = InteractomeRunner(str(inp), str(out), mode="colabfold")
        # Inject a ghost job (no FASTA file) into the pending list
        fake_status = pd.DataFrame({
            "PPI": ["Ghost__Job", "ProtA__ProtB", "ProtC__ProtD"],
            "status": ["FAILED", "FAILED", "FAILED"],
        })
        with patch.object(runner, "check_colabfold_run", return_value=fake_status):
            results = runner.run_colabfold_fastas(dry_run=True)
        assert len(results) == 2  # Ghost__Job skipped, 2 real jobs ran
        assert "FASTA not found" in caplog.text


class TestRunColabfoldCsvMocked:
    def test_csv_failed_subprocess(self, tmp_path):
        """Non-zero returncode from CSV batch yields FAILED status."""
        from unittest.mock import patch, MagicMock
        csv = tmp_path / "batch.csv"
        csv.write_text("id,sequence\ntest,AAAA\n")
        out = tmp_path / "out"
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = tmp_path
        runner.output_dir = out
        runner.mode = "colabfold"
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            result = runner.run_colabfold_csv(str(csv), str(out))
        assert result["status"] == "FAILED"
        assert result["returncode"] == 1

    def test_csv_missing_binary_raises(self, tmp_path):
        """FileNotFoundError propagated when binary not found."""
        from unittest.mock import patch
        csv = tmp_path / "batch.csv"
        csv.write_text("id,sequence\ntest,AAAA\n")
        out = tmp_path / "out"
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = tmp_path
        runner.output_dir = out
        runner.mode = "colabfold"
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                runner.run_colabfold_csv(str(csv), str(out))

    def test_csv_successful_subprocess(self, tmp_path):
        """L1430: returncode=0 logs success and returns COMPLETED status."""
        from unittest.mock import patch, MagicMock
        csv = tmp_path / "batch.csv"
        csv.write_text("id,sequence\ntest,AAAA\n")
        out = tmp_path / "out"
        runner = InteractomeRunner.__new__(InteractomeRunner)
        runner.input_dir = tmp_path
        runner.output_dir = out
        runner.mode = "colabfold"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("subprocess.run", return_value=mock_proc):
            result = runner.run_colabfold_csv(str(csv), str(out))
        assert result["status"] == "COMPLETED"
        assert result["returncode"] == 0


class TestRunSingleColabfoldJob:
    def test_dry_run_returns_expected_dict(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">test\nAAAA\n")
        out = tmp_path / "job_out"
        result = InteractomeRunner._run_single_colabfold_job(
            name="test_job",
            fasta_path=fasta,
            output_dir=out,
            base_cmd=["colabfold_batch"],
            dry_run=True,
        )
        assert result["name"] == "test_job"
        assert result["status"] == "DRY_RUN"
        assert result["returncode"] == 0
        assert result["elapsed_s"] == 0.0

    def test_dry_run_creates_output_dir(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">test\nAAAA\n")
        out = tmp_path / "new_dir"
        InteractomeRunner._run_single_colabfold_job(
            name="test", fasta_path=fasta, output_dir=out,
            base_cmd=["echo"], dry_run=True,
        )
        assert out.exists()
