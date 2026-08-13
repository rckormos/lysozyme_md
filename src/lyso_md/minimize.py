from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .amber import _parse_restart_atom_count, _parse_restart_coordinates
from .config import PipelineConfig

_FATAL_PATTERNS = (
    re.compile(r"\bCUDA\s+(?:error|fatal)\b", re.IGNORECASE),
    re.compile(r"\bCUDA\s+error", re.IGNORECASE),
    re.compile(r"\bNaN\b", re.IGNORECASE),
    re.compile(r"\bNAN\b", re.IGNORECASE),
    re.compile(r"SHAKE", re.IGNORECASE),
    re.compile(r"\bvlimit\b", re.IGNORECASE),
    re.compile(r"\bFATAL\b", re.IGNORECASE),
)
_COMPLETION_MARKERS = ("FINAL RESULTS", "5.  TIMINGS", "5. TIMINGS")
_JOB_RE = re.compile(r"Job <(\d+)>")


@dataclass(frozen=True)
class MinimizationSubmission:
    stage: Path
    solvent_script: Path
    all_script: Path
    submission_path: Path
    solvent_job_id: str | None
    all_job_id: str | None
    dry_run: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_inputs(workspace: Path) -> tuple[Path, Path]:
    stage = workspace / "04_solvate"
    if not (stage / ".done").is_file():
        raise ValueError("Phase 10 requires a completed Phase 9 solvation/ionization checkpoint")
    parm7 = stage / "complex_solvated.parm7"
    rst7 = stage / "complex_solvated.rst7"
    for path in (parm7, rst7):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Phase 10 requires {path.name} from Phase 9")
    return parm7, rst7


def _mask() -> str:
    return "(!:WAT,K+,Cl-)&(!@H=)"


def _render_input(restraint_wt: float, steps: int, cfg: PipelineConfig) -> str:
    return f"""Periodic restrained minimization for lyso-md\n&cntrl\n  imin=1,\n  maxcyc={steps},\n  ncyc={steps // 2},\n  ntb=1,\n  cut={cfg.md.cutoff_angstrom},\n  ntr=1,\n  restraint_wt={restraint_wt},\n  restraintmask='{_mask()}',\n  ntpr=100,\n/\n"""


def _render_lsf(cfg: PipelineConfig, *, workspace: Path, stage: Path, worker: str, dependency: str | None, walltime: str) -> str:
    dep = f'\n#BSUB -w "done({dependency})"' if dependency else ""
    return f'''#!/bin/bash\n#BSUB -P {cfg.scheduler.project}\n#BSUB -J {cfg.name}_min_{worker}{dep}\n#BSUB -q {cfg.scheduler.gpu_queue}\n#BSUB -n {cfg.scheduler.cores}\n#BSUB -W {walltime}\n#BSUB -gpu "{cfg.scheduler.gpu_resource}"\n#BSUB -R "rusage[mem={cfg.scheduler.memory}]"\n#BSUB -oo {workspace}/logs/minimize_{worker}.%J.out\n#BSUB -eo {workspace}/logs/minimize_{worker}.%J.err\n\nset -euo pipefail\ncd {stage}\ntest -s complex_solvated.parm7\ntest -s start.rst7\nlyso-md _minimize-worker {worker} {workspace}/config.yaml\n'''


def _submit(script: str) -> tuple[str, str]:
    proc = subprocess.run(["bsub"], input=script, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"bsub failed with exit code {proc.returncode}: {output.strip()}")
    match = _JOB_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse LSF job ID from bsub output: {output.strip()!r}")
    return match.group(1), output.strip()


def prepare_minimization(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> MinimizationSubmission:
    workspace = Path(workspace).resolve()
    parm7, rst7 = _required_inputs(workspace)
    stage = workspace / "05_minimize"
    stage.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    solvent = stage / "solvent"
    all_stage = stage / "all"
    solvent.mkdir(parents=True, exist_ok=True)
    all_stage.mkdir(parents=True, exist_ok=True)
    for substage in (solvent, all_stage):
        for name in ("min.in", "min.out", "min.rst7", ".done", "validation.json"):
            p = substage / name
            if p.exists() or p.is_symlink():
                p.unlink()
    for substage, start in ((solvent, rst7), (all_stage, solvent / "min.rst7")):
        for source in (parm7, start):
            target = substage / source.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source.resolve())
    solvent_input = solvent / "min.in"
    all_input = all_stage / "min.in"
    for substage in (solvent, all_stage):
        target = substage / "start.rst7"
        source = rst7 if substage is solvent else solvent / "min.rst7"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
    solvent_input.write_text(_render_input(10.0, cfg.equilibration.solvent_min_steps, cfg), encoding="utf-8")
    all_input.write_text(_render_input(5.0, cfg.equilibration.all_min_steps, cfg), encoding="utf-8")
    script_solvent = stage / "minimize_solvent.lsf"
    script_all = stage / "minimize_all.lsf"
    walltime = "06:00"
    script_solvent.write_text(_render_lsf(cfg, workspace=workspace, stage=solvent, worker="solvent", dependency=None, walltime=walltime), encoding="utf-8")
    script_all.write_text(_render_lsf(cfg, workspace=workspace, stage=all_stage, worker="all", dependency="PENDING", walltime=walltime), encoding="utf-8")
    submission_path = stage / "submission.json"
    if dry_run:
        script_all.write_text(_render_lsf(cfg, workspace=workspace, stage=all_stage, worker="all", dependency="<SOLVENT_JOB_ID>", walltime=walltime), encoding="utf-8")
        submission_path.write_text(json.dumps({"stage": "minimize", "status": "dry_run", "submitted": False, "solvent_script": str(script_solvent), "all_script": str(script_all)}, indent=2) + "\n", encoding="utf-8")
        return MinimizationSubmission(stage, script_solvent, script_all, submission_path, None, None, True)
    solvent_id, solvent_output = _submit(script_solvent.read_text(encoding="utf-8"))
    script_all.write_text(_render_lsf(cfg, workspace=workspace, stage=all_stage, worker="all", dependency=solvent_id, walltime=walltime), encoding="utf-8")
    all_id, all_output = _submit(script_all.read_text(encoding="utf-8"))
    submission_path.write_text(json.dumps({"stage": "minimize", "status": "submitted", "submitted": True, "submitted_at": _utc_now(), "solvent_job_id": solvent_id, "all_job_id": all_id, "solvent_bsub_output": solvent_output, "all_bsub_output": all_output, "solvent_script": str(script_solvent), "all_script": str(script_all)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MinimizationSubmission(stage, script_solvent, script_all, submission_path, solvent_id, all_id, False)


def _parse_final(text: str) -> dict[str, float]:
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError("pmemd.cuda output does not contain a normal-completion marker")
    rows = re.findall(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+", text, re.MULTILINE)
    if not rows:
        raise ValueError("pmemd.cuda output lacks a parseable minimization result row")
    step, energy, rms, gmax = rows[-1]
    result = {"step": int(step), "energy": float(energy.replace("D", "E").replace("d", "e")), "rms": float(rms.replace("D", "E").replace("d", "e")), "gmax": float(gmax.replace("D", "E").replace("d", "e"))}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("pmemd.cuda final minimization values are non-finite")
    return result


def _validate_worker(workspace: Path, worker: str) -> dict[str, Any]:
    stage = workspace / "05_minimize" / worker
    log = stage / "min.out"
    restart = stage / "min.rst7"
    parm7 = stage / "complex_solvated.parm7"
    if not log.is_file() or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError(f"Phase 10 {worker} minimization did not produce required outputs")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [line.strip() for line in text.splitlines() if any(p.search(line) for p in _FATAL_PATTERNS)]
    if fatal:
        raise ValueError("pmemd.cuda reported fatal/instability diagnostics: " + " | ".join(fatal[:5]))
    final = _parse_final(text)
    natom = _parse_restart_atom_count(restart)
    _parse_restart_coordinates(restart, natom)
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"{worker} minimization restart atom count {natom} disagrees with topology NATOM {match.group(1)}")
    validation = {"stage": f"minimize_{worker}", "status": "done", "pipeline_version": __version__, "completed_at": _utc_now(), "results": final, "checks": {"normal_completion": True, "finite_energy_gradient": True, "restart_exists": True, "finite_coordinates": True, "matching_atom_counts": True, "no_cuda_or_nan_failures": True, "passed": True}, "outputs": {"input": str(stage / "min.in"), "log": str(log), "restart": str(restart)}, "sha256": {p.name: _sha256(p) for p in (stage / "min.in", log, restart)}}
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": f"minimize_{worker}", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(restart)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def run_minimization_worker(cfg: PipelineConfig, *, workspace: Path, worker: str) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    if worker not in {"solvent", "all"}:
        raise ValueError("Phase 10 worker must be 'solvent' or 'all'")
    stage = workspace / "05_minimize" / worker
    pmemd = shutil.which("pmemd.cuda")
    if not pmemd:
        raise RuntimeError("pmemd.cuda was not found in PATH; load Amber 22 and a CUDA-enabled environment before Phase 10")
    start = stage / "start.rst7"
    output = stage / "min.rst7"
    # For the first stage the local symlink is the Phase 9 starting restart; for
    # the second stage it is the first minimization output.
    proc = subprocess.run([pmemd, "-O", "-i", "min.in", "-o", "min.out", "-p", "complex_solvated.parm7", "-c", start.name, "-ref", start.name, "-r", output.name], cwd=stage, text=True, capture_output=True)
    if proc.stdout.strip() or proc.stderr.strip():
        with (stage / "min.out").open("a", encoding="utf-8") as handle:
            if proc.stdout.strip():
                handle.write("\n--- STDOUT ---\n" + proc.stdout)
            if proc.stderr.strip():
                handle.write("\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"pmemd.cuda exited with status {proc.returncode}; see {stage / 'min.out'}")
    validation = _validate_worker(workspace, worker)
    if worker == "all":
        parent = workspace / "05_minimize"
        (parent / ".done").write_text(json.dumps({"stage": "minimize", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(parent / "all" / "validation.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation
