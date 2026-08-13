from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.equilibration import prepare_npt_equilibration, run_npt_equilibration_worker


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "equil_test",
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
    smoke = ws / "06_equilibrate" / "npt_smoke"
    smoke.mkdir(parents=True)
    (smoke / "complex_solvated.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n", encoding="utf-8")
    (smoke / "npt_smoke.rst7").write_text("smoke\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n", encoding="utf-8")
    (smoke / ".done").write_text("{}\n", encoding="utf-8")
    return ws


def test_dry_run_writes_three_stages_and_dependencies(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_npt_equilibration(cfg, workspace=ws, dry_run=True)
    root = ws / "06_equilibrate" / "npt_equilibrate"
    assert [p.name for p in result.script_paths] == ["stage.lsf", "stage.lsf", "stage.lsf"]
    assert "nstlim=125000" in (root / "restraint5" / "stage.in").read_text()
    assert "restraint_wt=5.0" in (root / "restraint5" / "stage.in").read_text()
    assert "nstlim=125000" in (root / "restraint1" / "stage.in").read_text()
    assert "restraint_wt=1.0" in (root / "restraint1" / "stage.in").read_text()
    free = (root / "free" / "stage.in").read_text()
    assert "nstlim=250000" in free
    assert "ntr=0" in free
    assert "restraint_wt" not in free
    assert "pres0=1" in free
    assert "done(<NPT_5_JOB_ID>)" in (root / "restraint1" / "stage.lsf").read_text()
    assert "done(<NPT_1_JOB_ID>)" in (root / "free" / "stage.lsf").read_text()
    assert not (root / ".done").exists()


def test_dry_run_requires_smoke_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    (ws / "06_equilibrate" / "npt_smoke" / ".done").unlink()
    with pytest.raises(ValueError, match="Phase 13 requires"):
        prepare_npt_equilibration(cfg, workspace=ws, dry_run=True)


def test_dry_run_chains_to_smoke_job(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    (ws / "06_equilibrate" / "npt_smoke" / ".done").unlink()
    (ws / "06_equilibrate" / "npt_smoke" / "submission.json").write_text(json.dumps({"status": "submitted", "job_id": "1234"}), encoding="utf-8")
    result = prepare_npt_equilibration(cfg, workspace=ws, dry_run=True)
    assert 'done(1234)' in (result.stage / "restraint5" / "stage.lsf").read_text()


def test_worker_validates_stage_and_writes_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_npt_equilibration(cfg, workspace=ws, dry_run=True)
    stage = result.stage / "restraint5"
    script = tmp_path / "pmemd.cuda"
    script.write_text(
        "#!/bin/sh\n"
        "cat > stage.out <<'EOF'\n"
        "NSTEP = 0 TIME(PS) = 0.000 TEMP(K) = 300.0 PRESS = 0.0 DENSITY = 0.90\n"
        "NSTEP = 125000 TIME(PS) = 250.000 TEMP(K) = 301.2 PRESS = 1.1 DENSITY = 0.9912\n"
        "5.  TIMINGS\n"
        "EOF\n"
        "cat > stage.rst7 <<'EOF'\n"
        "npt\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    validation = run_npt_equilibration_worker(cfg, workspace=ws, stage_name="restraint5")
    assert validation["checks"]["passed"] is True
    assert validation["results"]["temperature_k"] == pytest.approx(301.2)
    assert (stage / ".done").is_file()
    assert not (result.stage / ".done").exists()
