from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.heating import prepare_heating, run_heating_worker


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "heat_test",
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


def _workspace(tmp_path: Path, *, min_done: bool = True) -> Path:
    ws = tmp_path / "workspace"
    (ws / "04_solvate").mkdir(parents=True)
    (ws / "05_minimize" / "all").mkdir(parents=True)
    if min_done:
        (ws / "05_minimize" / ".done").write_text("{}\n")
    (ws / "04_solvate" / ".done").write_text("{}\n")
    (ws / "04_solvate" / "complex_solvated.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n", encoding="utf-8")
    (ws / "04_solvate" / "complex_solvated.rst7").write_text(
        "solvated\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n",
        encoding="utf-8",
    )
    (ws / "05_minimize" / "all" / "min.rst7").write_text(
        "minimized\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n",
        encoding="utf-8",
    )
    return ws


def test_dry_run_generates_10k_to_300k_nvt_input(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_heating(cfg, workspace=ws, dry_run=True)
    text = (result.stage / "heat.in").read_text()
    assert "irest=0" in text and "ntx=1" in text
    assert "nstlim=50000" in text
    assert "dt=0.002000" in text
    assert "ntb=1" in text and "ntp=0" in text
    assert "ntt=3" in text and "gamma_ln=5.0" in text
    assert "tempi=10.0" in text and "temp0=300" in text
    assert "ntc=2" in text and "ntf=2" in text
    assert "restraint_wt=5.0" in text
    assert "restraintmask='(!:WAT,K+,Cl-)&(!@H=)'" in text
    assert "iwrap=0" in text
    assert '#BSUB -gpu "num=1/host"' in result.script_path.read_text()
    assert not result.stage.joinpath(".done").exists()


def test_dry_run_uses_completed_phase10_without_dependency(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, min_done=True)
    result = prepare_heating(cfg, workspace=ws, dry_run=True)
    assert 'done(' not in result.script_path.read_text()


def test_dry_run_chains_to_phase10_job_when_minimization_is_running(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, min_done=False)
    (ws / "05_minimize" / "submission.json").write_text(json.dumps({"all_job_id": "90210"}), encoding="utf-8")
    result = prepare_heating(cfg, workspace=ws, dry_run=True)
    assert 'done(90210)' in result.script_path.read_text()



def test_dry_run_from_minimize_through_heat_uses_placeholder_dependency(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, min_done=False)
    from lyso_md.minimize import prepare_minimization
    prepare_minimization(cfg, workspace=ws, dry_run=True)
    result = prepare_heating(cfg, workspace=ws, dry_run=True)
    assert 'done(<ALL_JOB_ID>)' in result.script_path.read_text()

def _fake_pmemd_cuda(path: Path, *, temperature: float = 299.4, bad: bool = False) -> None:
    script = path / "pmemd.cuda"
    marker = (
        " NSTEP = 50000  TIME(PS) = 100.000  TEMP(K) = %.3f  PRESS = 0.000\n"
        "  FINAL RESULTS\n"
        "  NSTEP       ENERGY          RMS            GMAX         NAME    NUMBER\n"
        "  50000      -1.0000E+05     1.0000E-01     5.0000E+00     CG        834\n"
        "5.  TIMINGS\n"
    ) % temperature
    if bad:
        marker += "SHAKE FAILURE\n"
    script.write_text(
        "#!/bin/sh\n"
        "cat > heat.out <<'EOF'\n" + marker + "EOF\n"
        "cat > heat.rst7 <<'EOF'\n"
        "heated\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def test_worker_validates_temperature_and_writes_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_heating(cfg, workspace=ws, dry_run=True)
    _fake_pmemd_cuda(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    validation = run_heating_worker(cfg, workspace=ws)
    assert validation["results"]["temperature_k"] == pytest.approx(299.4)
    assert validation["checks"]["temperature_in_range"] is True
    assert (ws / "06_equilibrate" / "heat" / ".done").is_file()


def test_worker_fails_on_shake_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    prepare_heating(cfg, workspace=ws, dry_run=True)
    _fake_pmemd_cuda(tmp_path, bad=True)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(ValueError, match="fatal/instability"):
        run_heating_worker(cfg, workspace=ws)
    assert not (ws / "06_equilibrate" / "heat" / ".done").exists()


def test_missing_phase10_checkpoint_fails(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, min_done=False)
    with pytest.raises(ValueError, match="Phase 10"):
        prepare_heating(cfg, workspace=ws, dry_run=True)
