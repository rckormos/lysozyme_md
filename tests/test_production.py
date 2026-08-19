from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.production import prepare_production, run_production_worker


def _config(tmp_path: Path, target_ns: float = 1000, chunk_ns: float = 250) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "production_test",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {"bundle": str(bundle), "unit_name": "CONDENSEDSEQUENCE", "expected_heavy_atoms": 3, "expected_residues": 1},
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {"hydrogen_relax_steps": 1000, "solvent_min_steps": 5000, "all_min_steps": 5000, "heat_ps": 100, "npt_5_ps": 250, "npt_1_ps": 250, "npt_free_ps": 500},
        "production": {"target_ns": target_ns, "chunk_ns": chunk_ns, "walltime_hours": 72},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "32GB", "cores": 1},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _parm(path: Path, natom: int = 2) -> None:
    path.write_text(f"%FLAG POINTERS\n%FORMAT(10I8)\n{natom:8d}\n", encoding="utf-8")


def _rst(path: Path, natom: int = 2) -> None:
    path.write_text("restart\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n", encoding="utf-8")


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    free = ws / "06_equilibrate" / "npt_equilibrate" / "free"
    free.mkdir(parents=True)
    _parm(free / "complex_solvated.parm7")
    _rst(free / "stage.rst7")
    (free / ".done").write_text("{}\n", encoding="utf-8")
    return ws


def test_dry_run_first_chunk_uses_free_restart(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_production(cfg, workspace=ws, dry_run=True)
    stage = ws / "07_production" / "chunk_001"
    assert result.chunk_number == 1
    assert "nstlim=125000000" in (stage / "production.in").read_text()
    assert "irest=1" in (stage / "production.in").read_text()
    assert "ntr=0" in (stage / "production.in").read_text()
    assert "-W 72:00" in (stage / "production.lsf").read_text()
    assert (stage / "start.rst7").is_symlink()
    assert not (ws / "07_production" / ".done").exists()


def test_resume_uses_completed_time_and_previous_restart(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    first = prepare_production(cfg, workspace=ws, dry_run=True)
    stage1 = first.stage
    (stage1 / "validation.json").write_text(json.dumps({"status": "done", "checks": {"passed": True}, "results": {"completed_ns": 250.0}}), encoding="utf-8")
    (stage1 / ".done").write_text("{}\n", encoding="utf-8")
    _rst(stage1 / "production.rst7")
    (stage1 / "submission.json").write_text(json.dumps({"status": "submitted", "job_id": "12345"}), encoding="utf-8")
    second = prepare_production(cfg, workspace=ws, dry_run=True)
    assert second.chunk_number == 2
    assert "done(12345)" in (second.stage / "production.lsf").read_text()
    assert (second.stage / "start.rst7").resolve() == (stage1 / "production.rst7").resolve()


def test_final_partial_chunk_and_aggregate_done(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path, target_ns=500, chunk_ns=250))
    ws = _workspace(tmp_path)
    for number, completed in ((1, 250.0), (2, 500.0)):
        stage = ws / "07_production" / f"chunk_{number:03d}"
        stage.mkdir(parents=True)
        _rst(stage / "production.rst7")
        (stage / "validation.json").write_text(json.dumps({"status": "done", "checks": {"passed": True}, "results": {"completed_ns": completed}}), encoding="utf-8")
        (stage / ".done").write_text("{}\n", encoding="utf-8")
    result = prepare_production(cfg, workspace=ws, dry_run=True)
    assert result.completed is True
    assert (ws / "07_production" / ".done").is_file()


def test_worker_validates_realistic_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_production(cfg, workspace=ws, dry_run=True)
    stage = result.stage
    script = tmp_path / "pmemd.cuda"
    script.write_text(
        "#!/bin/sh\n"
        "cat > production.out <<'EOF'\n"
        "NSTEP = 125000000 TIME(PS) = 250000.000 TEMP(K) = 300.2 PRESS = 2.1 DENSITY = 1.0345\n"
        "5.  TIMINGS\n"
        "|     Shake             0.00    0.00\n"
        "EOF\n"
        "cat > production.rst7 <<'EOF'\n"
        "production\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        "exit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    validation = run_production_worker(cfg, workspace=ws, chunk_number=1)
    assert validation["checks"]["passed"] is True
    assert validation["results"]["completed_ns"] == pytest.approx(250.0)
    assert (stage / ".done").is_file()
