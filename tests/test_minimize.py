from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.minimize import prepare_minimization, run_minimization_worker


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "min_test",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {"bundle": str(bundle), "unit_name": "CONDENSEDSEQUENCE", "expected_heavy_atoms": 3, "expected_residues": 1},
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {"hydrogen_relax_steps": 1000, "solvent_min_steps": 5000, "all_min_steps": 5000, "heat_ps": 100, "npt_5_ps": 250, "npt_1_ps": 250, "npt_free_ps": 500},
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 72},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "32GB", "cores": 1},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    stage = ws / "04_solvate"
    stage.mkdir(parents=True)
    (stage / ".done").write_text("{}\n")
    (stage / "complex_solvated.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n", encoding="utf-8")
    (stage / "complex_solvated.rst7").write_text("solvated\n2\n  0.000000  0.000000  0.000000  1.000000  1.000000  1.000000\n", encoding="utf-8")
    return ws


def test_dry_run_writes_both_gpu_minimization_inputs_and_dependency_script(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_minimization(cfg, workspace=ws, dry_run=True)
    solvent = (ws / "05_minimize" / "solvent" / "min.in").read_text()
    all_text = (ws / "05_minimize" / "all" / "min.in").read_text()
    assert "maxcyc=5000" in solvent and "ncyc=2500" in solvent
    assert "restraint_wt=10.0" in solvent
    assert "maxcyc=5000" in all_text and "ncyc=2500" in all_text
    assert "restraint_wt=5.0" in all_text
    assert "ntb=1" in all_text and "cut=9.0" in all_text
    assert "restraintmask='(!:WAT,K+,Cl-)&(!@H=)'" in all_text
    assert '#BSUB -gpu "num=1/host"' in result.solvent_script.read_text()
    assert 'done(<SOLVENT_JOB_ID>)' in result.all_script.read_text()
    assert not (ws / "05_minimize" / ".done").exists()


def test_submission_chains_second_gpu_job_on_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    counter = iter(["Job <101> is submitted", "Job <102> is submitted"])
    import lyso_md.minimize as mod
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": next(counter)})())
    result = prepare_minimization(cfg, workspace=ws)
    assert result.solvent_job_id == "101"
    assert result.all_job_id == "102"
    assert 'done(101)' in result.all_script.read_text()
    submission = json.loads(result.submission_path.read_text())
    assert submission["solvent_job_id"] == "101"
    assert submission["all_job_id"] == "102"


def _fake_pmemd_cuda(path: Path, *, bad: bool = False) -> None:
    script = path / "pmemd.cuda"
    marker = "FINAL RESULTS\n NSTEP ENERGY RMS GMAX NAME NUMBER\n 5000 -100.0 0.0100 0.1000 TEST 1\n5.  TIMINGS\n"
    if bad:
        marker += "CUDA error: launch failed\n"
    script.write_text(
        "#!/bin/sh\n"
        "cat > min.out <<'EOF'\n" + marker + "EOF\n"
        "cat > min.rst7 <<'EOF'\n"
        "minimized\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        + "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def test_worker_validates_success_and_finalizes_phase10(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_minimization(cfg, workspace=ws, dry_run=True)
    _fake_pmemd_cuda(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    run_minimization_worker(cfg, workspace=ws, worker="solvent")
    run_minimization_worker(cfg, workspace=ws, worker="all")
    assert (ws / "05_minimize" / ".done").is_file()
    validation = json.loads((ws / "05_minimize" / "all" / "validation.json").read_text())
    assert validation["checks"]["passed"] is True


def test_worker_fails_on_cuda_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_minimization(cfg, workspace=ws, dry_run=True)
    _fake_pmemd_cuda(tmp_path, bad=True)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(ValueError, match="fatal/instability"):
        run_minimization_worker(cfg, workspace=ws, worker="solvent")
    assert not (ws / "05_minimize" / "solvent" / ".done").exists()
