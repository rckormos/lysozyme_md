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

_FATAL_PATTERNS = (
    re.compile(r"\bCUDA\s+(?:error|fatal)\b", re.IGNORECASE),
    re.compile(r"\bNaN\b", re.IGNORECASE),
    re.compile(r"\bInf(?:inity)?\b", re.IGNORECASE),
    re.compile(r"\bSHAKE\s+(?:FAILURE|FAILED|FAIL|CANNOT|NOT\s+CONVERG|DID\s+NOT\s+CONVERG)", re.IGNORECASE),
    re.compile(r"\bvlimit\s+(?:exceeded|violation|failure|failed)\b", re.IGNORECASE),
    re.compile(r"\bFATAL\b", re.IGNORECASE),
)
_COMPLETION_MARKERS = ("5.  TIMINGS", "5. TIMINGS")
_TEMP_EQUALS_RE = re.compile(r"TEMP\(K\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_TEMP_TABLE_RE = re.compile(
    r"^\s*\d+\s+[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+",
    re.MULTILINE,
)
_JOB_RE = re.compile(r"Job <(\d+)>")


@dataclass(frozen=True)
class HeatingSubmission:
    stage: Path
    script_path: Path
    submission_path: Path
    job_id: str | None
    dry_run: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_inputs(workspace: Path) -> tuple[Path, Path, str | None]:
    minimize = workspace / "05_minimize"
    solvation = workspace / "04_solvate"
    parm7 = solvation / "complex_solvated.parm7"
    restart = minimize / "all" / "min.rst7"
    if not (solvation / ".done").is_file():
        raise ValueError("Phase 11 requires a completed Phase 9 solvation/ionization checkpoint")
    if not parm7.is_file() or parm7.stat().st_size == 0:
        raise ValueError("Phase 11 requires complex_solvated.parm7 from Phase 9")
    if (minimize / ".done").is_file():
        if not restart.is_file() or restart.stat().st_size == 0:
            raise ValueError("Phase 11 requires the completed Phase 10 all-system minimization restart")
        dependency = None
    else:
        submission = minimize / "submission.json"
        if not submission.is_file():
            raise ValueError("Phase 11 requires a completed Phase 10 checkpoint or Phase 10 LSF submission metadata")
        try:
            payload = json.loads(submission.read_text(encoding="utf-8"))
            if payload.get("status") == "dry_run":
                dependency = "<ALL_JOB_ID>"
            else:
                dependency = str(payload["all_job_id"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 11 could not determine the Phase 10 all-system LSF job ID") from exc
    return parm7, restart, dependency


def _heat_steps(cfg: PipelineConfig) -> int:
    steps = int(round(cfg.equilibration.heat_ps * 1000.0 / cfg.md.production_timestep_fs))
    if steps <= 0:
        raise ValueError("equilibration.heat_ps and production timestep must produce a positive heating step count")
    return steps


def _render_input(cfg: PipelineConfig) -> str:
    steps = _heat_steps(cfg)
    return f"""Restrained NVT heating for lyso-md\n&cntrl\n  imin=0,\n  irest=0,\n  ntx=1,\n  nstlim={steps},\n  dt={cfg.md.production_timestep_fs / 1000.0:.6f},\n  ntb=1,\n  ntp=0,\n  cut={cfg.md.cutoff_angstrom},\n  ntt=3,\n  gamma_ln=5.0,\n  tempi=10.0,\n  temp0={cfg.md.temperature_k},\n  ntr=1,\n  restraint_wt=5.0,\n  restraintmask='(!:WAT,K+,Cl-)&(!@H=)',\n  ntc=2,\n  ntf=2,\n  ntpr=500,\n  ntwx=500,\n  iwrap=0,\n  ioutfm=1,\n/\n"""


def _render_lsf(cfg: PipelineConfig, *, workspace: Path, stage: Path, dependency: str | None) -> str:
    dep = f'\n#BSUB -w "done({dependency})"' if dependency else ""
    return f'''#!/bin/bash\n#BSUB -P {cfg.scheduler.project}\n#BSUB -J {cfg.name}_heat{dep}\n#BSUB -q {cfg.scheduler.gpu_queue}\n#BSUB -n {cfg.scheduler.cores}\n#BSUB -W 06:00\n#BSUB -gpu "{cfg.scheduler.gpu_resource}"\n#BSUB -R "rusage[mem={cfg.scheduler.memory}]"\n#BSUB -oo {workspace}/logs/heat.%J.out\n#BSUB -eo {workspace}/logs/heat.%J.err\n\nset -euo pipefail\ncd {stage}\ntest -s complex_solvated.parm7\ntest -s start.rst7\nlyso-md _heat-worker {workspace}/config.yaml\n'''


def _submit(script: str) -> tuple[str, str]:
    proc = subprocess.run(["bsub"], input=script, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"bsub failed with exit code {proc.returncode}: {output.strip()}")
    match = _JOB_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse LSF job ID from bsub output: {output.strip()!r}")
    return match.group(1), output.strip()


def _parse_final_temperature(text: str) -> float:
    """Return the final dynamics temperature, excluding summary statistics.

    Amber prints additional TEMP(K) records in the AVERAGES and RMS
    FLUCTUATIONS sections after the final dynamics record.  The latter can
    look like a temperature to a naive parser (for example, 1.83 K for an
    RMS fluctuation), so only the primary dynamics section is considered.
    """
    section = text

    for marker in (
        "A V E R A G E S",
        "A V E R A G E",
        "R M S  F L U C T U A T I O N S",
        "R M S F L U C T U A T I O N S",
        "5.  TIMINGS",
        "5. TIMINGS",
    ):
        position = section.find(marker)
        if position >= 0:
            section = section[:position]

    matches = _TEMP_EQUALS_RE.findall(section)
    if matches:
        value = float(matches[-1].replace("D", "E").replace("d", "e"))
    else:
        table = _TEMP_TABLE_RE.findall(section)
        if not table:
            raise ValueError("pmemd.cuda output does not contain a parseable final dynamics temperature")
        value = float(table[-1].replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("heating output contains a non-finite final temperature")
    return value


def _validate_worker(workspace: Path) -> dict[str, Any]:
    stage = workspace / "06_equilibrate" / "heat"
    log = stage / "heat.out"
    restart = stage / "heat.rst7"
    parm7 = stage / "complex_solvated.parm7"
    if not log.is_file() or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError("Phase 11 heating did not produce required outputs")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in _FATAL_PATTERNS)]
    if fatal:
        raise ValueError("pmemd.cuda reported fatal/instability diagnostics: " + " | ".join(fatal[:5]))
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError("pmemd.cuda heating output does not contain a normal-completion marker")
    temperature = _parse_final_temperature(text)
    target = 300.0
    if abs(temperature - target) > 50.0:
        raise ValueError(f"final heating temperature {temperature:.3f} K is outside the acceptable 250-350 K range")
    natom = _parse_restart_atom_count(restart)
    _parse_restart_coordinates(restart, natom)
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"heating restart atom count {natom} disagrees with topology NATOM {match.group(1)}")
    validation = {
        "stage": "heat",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "results": {"temperature_k": temperature, "target_temperature_k": target, "steps": _heat_steps_from_input(stage / "heat.in")},
        "checks": {"normal_completion": True, "finite_temperature": True, "temperature_in_range": True, "restart_exists": True, "finite_coordinates": True, "matching_atom_counts": True, "no_cuda_or_shake_instability": True, "passed": True},
        "inputs": {"parm7": str(parm7), "restart": str(workspace / "05_minimize" / "all" / "min.rst7")},
        "outputs": {"input": str(stage / "heat.in"), "log": str(log), "restart": str(restart), "trajectory": str(stage / "heat.nc") if (stage / "heat.nc").is_file() else None},
        "sha256": {p.name: _sha256(p) for p in (stage / "heat.in", log, restart) if p.is_file()},
    }
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": "heat", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(restart)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def _heat_steps_from_input(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"nstlim\s*=\s*(\d+)", text)
    if not match:
        raise ValueError("cannot determine heating step count from heat.in")
    return int(match.group(1))


def run_heating_worker(cfg: PipelineConfig, *, workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    stage = workspace / "06_equilibrate" / "heat"
    input_path = stage / "heat.in"
    log_path = stage / "heat.out"
    restart = stage / "heat.rst7"
    if not input_path.is_file():
        raise ValueError("Phase 11 heat.in is missing")
    pmemd = shutil.which("pmemd.cuda")
    if not pmemd:
        raise RuntimeError("pmemd.cuda was not found in PATH; load the Amber 22 module before running Phase 11")
    proc = subprocess.run([pmemd, "-O", "-i", input_path.name, "-o", log_path.name, "-p", "complex_solvated.parm7", "-c", "start.rst7", "-ref", "start.rst7", "-r", restart.name, "-x", "heat.nc"], cwd=stage, text=True, capture_output=True)
    if proc.stdout.strip() and log_path.is_file():
        current = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.stdout.strip() not in current:
            log_path.write_text(current + "\n--- STDOUT ---\n" + proc.stdout, encoding="utf-8")
    elif proc.stdout.strip() and not log_path.is_file():
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.stderr and log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"pmemd.cuda exited with status {proc.returncode}; inspect {log_path}")
    validation = _validate_worker(workspace)
    validation["process"] = {"returncode": proc.returncode}
    validation["sha256"] = {p.name: _sha256(p) for p in (input_path, log_path, restart) if p.is_file()}
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def prepare_heating(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> HeatingSubmission:
    workspace = Path(workspace).resolve()
    parm7, restart, dependency = _required_inputs(workspace)
    stage = workspace / "06_equilibrate" / "heat"
    stage.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("heat.in", "heat.out", "heat.rst7", "heat.nc", "validation.json", ".done", "start.rst7", "complex_solvated.parm7"):
        path = stage / name
        if path.exists() or path.is_symlink():
            path.unlink()
    for source in (parm7, restart):
        target = stage / ("complex_solvated.parm7" if source == parm7 else "start.rst7")
        target.symlink_to(source.resolve())
    input_path = stage / "heat.in"
    input_path.write_text(_render_input(cfg), encoding="utf-8")
    script = stage / "heat.lsf"
    script.write_text(_render_lsf(cfg, workspace=workspace, stage=stage, dependency=dependency), encoding="utf-8")
    submission_path = stage / "submission.json"
    if dry_run:
        payload = {"stage": "heat", "status": "dry_run", "submitted": False, "dependency_job_id": dependency, "script": str(script), "input": str(input_path)}
        submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return HeatingSubmission(stage, script, submission_path, None, True)
    job_id, output = _submit(script.read_text(encoding="utf-8"))
    payload = {"stage": "heat", "status": "submitted", "submitted": True, "submitted_at": _utc_now(), "job_id": job_id, "dependency_job_id": dependency, "bsub_output": output, "script": str(script), "input": str(input_path)}
    submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HeatingSubmission(stage, script, submission_path, job_id, False)
