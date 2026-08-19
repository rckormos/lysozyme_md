from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .amber import _parse_restart_atom_count, _parse_restart_coordinates
from .config import PipelineConfig
from .npt import _FATAL_PATTERNS, _COMPLETION_MARKERS, _float

_JOB_RE = re.compile(r"Job <(\d+)>")
_TIME_RE = re.compile(r"\bTIME\(PS\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_TEMP_RE = re.compile(r"\bTEMP\(K\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_DENSITY_RE = re.compile(r"\bDensity\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_PRESS_RE = re.compile(r"\bPRESS\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _steps(ns: float, cfg: PipelineConfig) -> int:
    return max(1, int(round(ns * 1_000_000.0 / cfg.md.production_timestep_fs)))


def _target_ps(cfg: PipelineConfig) -> float:
    return cfg.production.target_ns * 1000.0


def _chunk_ns_for_remaining(cfg: PipelineConfig, completed_ns: float) -> float:
    remaining = max(0.0, cfg.production.target_ns - completed_ns)
    return min(cfg.production.chunk_ns, remaining)


def _parse_production_time(text: str) -> float:
    times = [_float(x) for x in _TIME_RE.findall(text)]
    if not times:
        raise ValueError("production output does not contain a TIME(PS) record")
    value = max(times)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"production output contains invalid simulation time: {value}")
    return value


def _primary_section(text: str) -> str:
    section = text
    for marker in ("A V E R A G E S", "A V E R A G E", "R M S  F L U C T U A T I O N S", "R M S F L U C T U A T I O N S", "5.  TIMINGS", "5. TIMINGS"):
        position = section.find(marker)
        if position >= 0:
            section = section[:position]
    return section


def _parse_observables(text: str) -> dict[str, float]:
    section = _primary_section(text)
    temps = [_float(x) for x in _TEMP_RE.findall(section)]
    densities = [_float(x) for x in _DENSITY_RE.findall(section)]
    pressures = [_float(x) for x in _PRESS_RE.findall(section)]
    result = {
        "temperature_k": temps[-1] if temps else math.nan,
        "density_g_cm3": densities[-1] if densities else math.nan,
        "pressure_bar": pressures[-1] if pressures else math.nan,
    }
    if not math.isfinite(result["temperature_k"]):
        raise ValueError("production output does not contain a finite final dynamics temperature")
    if not math.isfinite(result["density_g_cm3"]) or result["density_g_cm3"] <= 0:
        raise ValueError("production output does not contain a finite positive final dynamics density")
    return result


def _render_input(cfg: PipelineConfig, chunk_ns: float) -> str:
    return "\n".join([
        f"Chunked production MD ({chunk_ns:g} ns) for lyso-md",
        "&cntrl",
        "  imin=0,",
        "  irest=1,",
        "  ntx=5,",
        f"  nstlim={_steps(chunk_ns, cfg)},",
        f"  dt={cfg.md.production_timestep_fs / 1000.0:.6f},",
        "  ntb=2,",
        "  ntp=1,",
        "  barostat=1,",
        "  taup=5.0,",
        f"  pres0={cfg.md.pressure_bar},",
        f"  cut={cfg.md.cutoff_angstrom},",
        "  ntt=3,",
        "  gamma_ln=5.0,",
        f"  temp0={cfg.md.temperature_k},",
        "  ntr=0,",
        "  ntc=2,",
        "  ntf=2,",
        "  ntpr=500,",
        "  ntwx=500,",
        "  iwrap=0,",
        "  ioutfm=1,",
        "/",
        "",
    ])


def _render_lsf(cfg: PipelineConfig, *, workspace: Path, stage: Path, chunk_number: int, dependency: str | None) -> str:
    dep = f'\n#BSUB -w "done({dependency})"' if dependency else ""
    label = f"prod_{chunk_number:03d}"
    wall_minutes = max(1, int(round(cfg.production.walltime_hours * 60)))
    hours, minutes = divmod(wall_minutes, 60)
    walltime = f"{hours:02d}:{minutes:02d}"
    return f'''#!/bin/bash
#BSUB -P {cfg.scheduler.project}
#BSUB -J {cfg.name}_{label}{dep}
#BSUB -q {cfg.scheduler.gpu_queue}
#BSUB -n {cfg.scheduler.cores}
#BSUB -W {walltime}
#BSUB -gpu "{cfg.scheduler.gpu_resource}"
#BSUB -R "rusage[mem={cfg.scheduler.memory}]"
#BSUB -oo {workspace}/logs/{label}.%J.out
#BSUB -eo {workspace}/logs/{label}.%J.err

set -euo pipefail
cd {stage}
test -s complex_solvated.parm7
test -s start.rst7
lyso-md _production-worker {chunk_number} {workspace}/config.yaml
'''


def _submit(script: str) -> tuple[str, str]:
    proc = subprocess.run(["bsub"], input=script, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"bsub failed with exit code {proc.returncode}: {output.strip()}")
    match = _JOB_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse LSF job ID from bsub output: {output.strip()!r}")
    return match.group(1), output.strip()


def _chunk_paths(workspace: Path, number: int) -> tuple[Path, Path, Path]:
    stage = workspace / "07_production" / f"chunk_{number:03d}"
    return stage, stage / "production.rst7", stage / ".done"


def _read_completed_chunks(workspace: Path) -> tuple[float, int]:
    root = workspace / "07_production"
    completed_ns = 0.0
    number = 0
    while True:
        next_number = number + 1
        stage, restart, done = _chunk_paths(workspace, next_number)
        if not done.is_file():
            break
        validation_path = stage / "validation.json"
        if not validation_path.is_file():
            raise ValueError(f"production chunk {next_number} has .done but no validation.json")
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        if payload.get("status") != "done" or not payload.get("checks", {}).get("passed", False):
            raise ValueError(f"production chunk {next_number} is marked done without a passing validation")
        completed_ns = float(payload["results"]["completed_ns"])
        number = next_number
    return completed_ns, number


def _required_start(workspace: Path, chunk_number: int) -> tuple[Path, Path]:
    if chunk_number == 1:
        root = workspace / "06_equilibrate" / "npt_equilibrate" / "free"
        return root / "stage.rst7", root / "complex_solvated.parm7"
    prev_stage, prev_restart, prev_done = _chunk_paths(workspace, chunk_number - 1)
    if not prev_done.is_file() or not prev_restart.is_file():
        raise ValueError(f"production chunk {chunk_number} requires completed chunk {chunk_number - 1}")
    return prev_restart, prev_stage / "complex_solvated.parm7"


def _validate_chunk(cfg: PipelineConfig, workspace: Path, chunk_number: int) -> dict[str, Any]:
    stage, restart, _ = _chunk_paths(workspace, chunk_number)
    log = stage / "production.out"
    parm7 = stage / "complex_solvated.parm7"
    input_path = stage / "production.in"
    if not log.is_file() or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError(f"production chunk {chunk_number} did not produce required outputs")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in _FATAL_PATTERNS)]
    if fatal:
        raise ValueError("pmemd.cuda reported fatal/instability diagnostics: " + " | ".join(fatal[:5]))
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError(f"production chunk {chunk_number} output does not contain a normal-completion marker")
    completed_ps = _parse_production_time(text)
    observables = _parse_observables(text)
    natom = _parse_restart_atom_count(restart)
    _parse_restart_coordinates(restart, natom)
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"production restart atom count {natom} disagrees with topology NATOM {match.group(1)}")
    completed_ns = completed_ps / 1000.0
    validation = {
        "stage": f"production_chunk_{chunk_number:03d}",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "results": {"completed_ps": completed_ps, "completed_ns": completed_ns, **observables},
        "checks": {"normal_completion": True, "finite_observables": True, "restart_exists": True, "finite_coordinates": True, "matching_atom_counts": True, "no_cuda_or_shake_or_vlimit_instability": True, "passed": True},
        "inputs": {"parm7": str(parm7), "restart": str(stage / "start.rst7")},
        "outputs": {"input": str(input_path), "log": str(log), "restart": str(restart), "trajectory": str(stage / "production.nc") if (stage / "production.nc").is_file() else None},
        "sha256": {p.name: _sha256(p) for p in (input_path, log, restart) if p.is_file()},
    }
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": validation["stage"], "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(restart)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def run_production_worker(cfg: PipelineConfig, *, workspace: Path, chunk_number: int) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    stage, restart, _ = _chunk_paths(workspace, chunk_number)
    input_path = stage / "production.in"
    log_path = stage / "production.out"
    if not input_path.is_file():
        raise ValueError(f"Phase 14 chunk {chunk_number} input is missing")
    pmemd = shutil.which("pmemd.cuda")
    if not pmemd:
        raise RuntimeError("pmemd.cuda was not found in PATH; load the Amber 22 module before running Phase 14")
    proc = subprocess.run([pmemd, "-O", "-i", input_path.name, "-o", log_path.name, "-p", "complex_solvated.parm7", "-c", "start.rst7", "-r", restart.name, "-x", "production.nc"], cwd=stage, text=True, capture_output=True)
    if proc.stdout and log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STDOUT ---\n" + proc.stdout)
    if proc.stderr and log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"pmemd.cuda exited with status {proc.returncode}; inspect {log_path}")
    return _validate_chunk(cfg, workspace, chunk_number)


@dataclass(frozen=True)
class ProductionSubmission:
    stage: Path
    chunk_number: int | None
    script_path: Path | None
    submission_path: Path
    job_id: str | None
    dry_run: bool
    completed: bool


def prepare_production(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> ProductionSubmission:
    workspace = Path(workspace).resolve()
    root = workspace / "07_production"
    root.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    submission_path = root / "submission.json"
    completed_ns, last_chunk = _read_completed_chunks(workspace)
    if completed_ns >= cfg.production.target_ns - 1e-9:
        if not (root / ".done").is_file():
            payload = {"stage": "production", "status": "done", "pipeline_version": __version__, "completed_at": _utc_now(), "completed_ns": completed_ns}
            (root / ".done").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return ProductionSubmission(root, None, None, submission_path, None, dry_run, True)
    chunk_number = last_chunk + 1
    stage, restart, done = _chunk_paths(workspace, chunk_number)
    existing_submission = stage / "submission.json"
    if existing_submission.is_file() and not done.is_file():
        payload = json.loads(existing_submission.read_text(encoding="utf-8"))
        if payload.get("status") == "submitted" and payload.get("job_id"):
            return ProductionSubmission(stage, chunk_number, stage / "production.lsf", existing_submission, str(payload["job_id"]), False, False)
    start_restart, parm7 = _required_start(workspace, chunk_number)
    chunk_ns = _chunk_ns_for_remaining(cfg, completed_ns)
    if chunk_ns <= 0:
        raise ValueError("production target is complete but no completion checkpoint was found")
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("production.in", "production.out", "production.rst7", "production.nc", "validation.json", ".done", "start.rst7", "complex_solvated.parm7"):
        path = stage / name
        if path.exists() or path.is_symlink():
            path.unlink()
    (stage / "start.rst7").symlink_to(start_restart.resolve())
    (stage / "complex_solvated.parm7").symlink_to(parm7.resolve())
    input_path = stage / "production.in"
    input_path.write_text(_render_input(cfg, chunk_ns), encoding="utf-8")
    script_path = stage / "production.lsf"
    dependency = None
    if last_chunk:
        prev_submission = _chunk_paths(workspace, last_chunk)[0] / "submission.json"
        if prev_submission.is_file():
            dependency = json.loads(prev_submission.read_text(encoding="utf-8")).get("job_id")
    script_path.write_text(_render_lsf(cfg, workspace=workspace, stage=stage, chunk_number=chunk_number, dependency=dependency), encoding="utf-8")
    if dry_run:
        payload = {"stage": f"production_chunk_{chunk_number:03d}", "status": "dry_run", "submitted": False, "chunk_number": chunk_number, "chunk_ns": chunk_ns, "completed_before_ns": completed_ns, "dependency_job_id": dependency, "script": str(script_path), "input": str(input_path)}
        submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return ProductionSubmission(stage, chunk_number, script_path, submission_path, None, True, False)
    job_id, output = _submit(script_path.read_text(encoding="utf-8"))
    payload = {"stage": f"production_chunk_{chunk_number:03d}", "status": "submitted", "submitted": True, "submitted_at": _utc_now(), "job_id": job_id, "chunk_number": chunk_number, "chunk_ns": chunk_ns, "completed_before_ns": completed_ns, "dependency_job_id": dependency, "bsub_output": output, "script": str(script_path), "input": str(input_path)}
    submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ProductionSubmission(stage, chunk_number, script_path, submission_path, job_id, False, False)
