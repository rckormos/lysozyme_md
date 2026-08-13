from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.npt import prepare_npt_smoke, run_npt_smoke_worker, _parse_step_zero_restraint, _parse_final_observables


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "npt_test",
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


def _workspace(tmp_path: Path, *, heat_done: bool = True) -> Path:
    ws = tmp_path / "workspace"
    heat = ws / "06_equilibrate" / "heat"
    heat.mkdir(parents=True)
    (heat / "complex_solvated.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n", encoding="utf-8")
    (heat / "heat.rst7").write_text("heated\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n", encoding="utf-8")
    if heat_done:
        (heat / ".done").write_text("{}\n")
    return ws


def test_dry_run_writes_conservative_npt_input_and_dependency(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
    text = (result.stage / "npt_smoke.in").read_text()
    assert "irest=0" in text and "ntx=1" in text
    assert "nstlim=5000" in text and "dt=0.001" in text
    assert "ntb=2" in text and "ntp=1" in text and "barostat=1" in text
    assert "taup=5.0" in text and "tempi=300" in text and "temp0=300" in text
    assert "restraint_wt=5.0" in text
    assert "restraintmask='(!:WAT,K+,Cl-)&(!@H=)'" in text
    assert "iwrap=0" in text
    assert '#BSUB -gpu "num=1/host"' in result.script_path.read_text()
    assert not (result.stage / ".done").exists()


def test_dry_run_chains_to_running_heat_job(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, heat_done=False)
    (ws / "06_equilibrate" / "heat" / "submission.json").write_text(json.dumps({"status": "submitted", "job_id": "1234"}), encoding="utf-8")
    result = prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
    assert 'done(1234)' in result.script_path.read_text()


def test_dry_run_uses_placeholder_for_pending_heat_job(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, heat_done=False)
    (ws / "06_equilibrate" / "heat" / "submission.json").write_text(json.dumps({"status": "dry_run"}), encoding="utf-8")
    result = prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
    assert 'done(<HEAT_JOB_ID>)' in result.script_path.read_text()


def test_parse_step_zero_restraint() -> None:
    text = "NSTEP = 0  TIME(PS) = 0.000  TEMP(K) = 300.00  PRESS = 0.0\nBOND = 1.0  RESTRAINT = 123.456\n"
    assert _parse_step_zero_restraint(text) == pytest.approx(123.456)


def test_parse_final_observables_ignores_average_sections() -> None:
    text = "NSTEP = 5000 TIME(PS) = 5.000 TEMP(K) = 301.2 PRESS = 1.1 DENSITY = 0.9912\nA V E R A G E S\nNSTEP = 5000 TEMP(K) = 300.1 PRESS = 1.0 DENSITY = 0.9900\n"
    result = _parse_final_observables(text)
    assert result["temperature_k"] == pytest.approx(301.2)
    assert result["density_g_cm3"] == pytest.approx(0.9912)


def test_worker_validates_success_and_writes_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
    script = tmp_path / "pmemd.cuda"
    script.write_text(
        "#!/bin/sh\n"
        "cat > npt_smoke.out <<'EOF'\n"
        "NSTEP = 0  TIME(PS) = 0.000  TEMP(K) = 300.00  PRESS = 0.0\n"
        "BOND = 10.0  ANGLE = 20.0  RESTRAINT = 12.5\n"
        "NSTEP = 5000  TIME(PS) = 5.000  TEMP(K) = 301.20  PRESS = 1.10  DENSITY = 0.9912\n"
        "5.  TIMINGS\n"
        "EOF\n"
        "cat > npt_smoke.rst7 <<'EOF'\n"
        "npt\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    validation = run_npt_smoke_worker(cfg, workspace=ws)
    assert validation["checks"]["passed"] is True
    assert validation["results"]["temperature_k"] == pytest.approx(301.2)
    assert (ws / "06_equilibrate" / "npt_smoke" / ".done").is_file()


def test_worker_fails_on_anomalous_step_zero_restraint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
    script = tmp_path / "pmemd.cuda"
    script.write_text(
        "#!/bin/sh\n"
        "cat > npt_smoke.out <<'EOF'\n"
        "NSTEP = 0  TIME(PS) = 0.000  TEMP(K) = 300.00  PRESS = 0.0\n"
        "RESTRAINT = 500000.0\n"
        "NSTEP = 5000  TIME(PS) = 5.000  TEMP(K) = 301.20  PRESS = 1.10  DENSITY = 0.9912\n"
        "5.  TIMINGS\n"
        "EOF\n"
        "cat > npt_smoke.rst7 <<'EOF'\n"
        "npt\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(ValueError, match="restraint energy"):
        run_npt_smoke_worker(cfg, workspace=ws)
    assert not (ws / "06_equilibrate" / "npt_smoke" / ".done").exists()


def test_missing_heat_checkpoint_fails(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, heat_done=False)
    with pytest.raises(ValueError, match="Phase 11"):
        prepare_npt_smoke(cfg, workspace=ws, dry_run=True)
