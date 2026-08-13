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
_JOB_RE = re.compile(r"Job <(\d+)>")
_NSTEP_ZERO_RE = re.compile(r"\bNSTEP\s*=\s*0\b", re.IGNORECASE)
_RESTRAINT_RE = re.compile(r"\bRESTRAINT\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_DENSITY_RE = re.compile(r"\bDENSITY\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_TEMP_RE = re.compile(r"\bTEMP\(K\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_PRESS_RE = re.compile(r"\bPRESS\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)

# The Phase 12 smoke test is deliberately short. It is a stability check, not
# the first full density-equilibration stage from Phase 13.
SMOKE_TEST_PS = 5.0
SMOKE_RESTRAINT_ENERGY_MAX = 10_000.0


@dataclass(frozen=True)
class NptSmokeSubmission:
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


def _smoke_steps(cfg: PipelineConfig) -> int:
    steps = int(round(SMOKE_TEST_PS * 1000.0 / 1.0))
    if steps <= 0:
        raise ValueError("Phase 12 smoke-test duration must produce a positive step count")
    return steps


def _required_inputs(workspace: Path) -> tuple[Path, Path, str | None]:
    heat = workspace / "06_equilibrate" / "heat"
    parm7 = heat / "complex_solvated.parm7"
    restart = heat / "heat.rst7"
    if not (heat / ".done").is_file():
        submission = heat / "submission.json"
        if not submission.is_file():
            raise ValueError("Phase 12 requires a completed Phase 11 heating checkpoint or Phase 11 LSF submission metadata")
        try:
            payload = json.loads(submission.read_text(encoding="utf-8"))
            if payload.get("status") == "dry_run":
                dependency = "<HEAT_JOB_ID>"
            else:
                dependency = str(payload["job_id"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 12 could not determine the Phase 11 heating LSF job ID") from exc
    else:
        dependency = None
    if not parm7.is_file() or parm7.stat().st_size == 0:
        raise ValueError("Phase 12 requires complex_solvated.parm7 from Phase 9")
    if (heat / ".done").is_file() and (not restart.is_file() or restart.stat().st_size == 0):
        raise ValueError("Phase 12 requires heat.rst7 from the completed Phase 11 heating stage")
    return parm7, restart, dependency


def _render_input(cfg: PipelineConfig) -> str:
    return f"""Conservative NPT smoke test for lyso-md\n&cntrl\n  imin=0,\n  irest=0,\n  ntx=1,\n  nstlim={_smoke_steps(cfg)},\n  dt=0.001,\n  ntb=2,\n  ntp=1,\n  barostat=1,\n  taup=5.0,\n  cut={cfg.md.cutoff_angstrom},\n  ntt=3,\n  gamma_ln=5.0,\n  tempi={cfg.md.temperature_k},\n  temp0={cfg.md.temperature_k},\n  ntr=1,\n  restraint_wt=5.0,\n  restraintmask='(!:WAT,K+,Cl-)&(!@H=)',\n  ntc=2,\n  ntf=2,\n  ntpr=100,\n  ntwx=100,\n  iwrap=0,\n/\n"""


def _render_lsf(cfg: PipelineConfig, *, workspace: Path, stage: Path, dependency: str | None) -> str:
    dep = f'\n#BSUB -w "done({dependency})"' if dependency else ""
    return f'''#!/bin/bash
#BSUB -P {cfg.scheduler.project}
#BSUB -J {cfg.name}_npt_smoke{dep}
#BSUB -q {cfg.scheduler.gpu_queue}
#BSUB -n {cfg.scheduler.cores}
#BSUB -W 06:00
#BSUB -gpu "{cfg.scheduler.gpu_resource}"
#BSUB -R "rusage[mem={cfg.scheduler.memory}]"
#BSUB -oo {workspace}/logs/npt_smoke.%J.out
#BSUB -eo {workspace}/logs/npt_smoke.%J.err

set -euo pipefail
cd {stage}
test -s complex_solvated.parm7
test -s start.rst7
lyso-md _npt-smoke-worker {workspace}/config.yaml
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


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _parse_step_zero_restraint(text: str) -> float:
    match = _NSTEP_ZERO_RE.search(text)
    if not match:
        raise ValueError("pmemd.cuda NPT output does not contain a step-0 record")
    window = text[match.start() : match.start() + 5000]
    restraint = _RESTRAINT_RE.search(window)
    if not restraint:
        raise ValueError("pmemd.cuda NPT output does not contain a parseable step-0 restraint energy")
    value = _float(restraint.group(1))
    if not math.isfinite(value):
        raise ValueError("NPT step-0 restraint energy is non-finite")
    return value


def _primary_section(text: str) -> str:
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
    return section


def _parse_final_observables(text: str) -> dict[str, float]:
    section = _primary_section(text)
    temps = [_float(value) for value in _TEMP_RE.findall(section)]
    pressures = [_float(value) for value in _PRESS_RE.findall(section)]
    densities = [_float(value) for value in _DENSITY_RE.findall(section)]
    if not temps:
        raise ValueError("pmemd.cuda NPT output does not contain a final dynamics temperature")
    if not densities:
        raise ValueError("pmemd.cuda NPT output does not contain a final dynamics density")
    temp = temps[-1]
    density = densities[-1]
    pressure = pressures[-1] if pressures else math.nan
    if not math.isfinite(temp) or not math.isfinite(density):
        raise ValueError("NPT final temperature or density is non-finite")
    if density <= 0.0:
        raise ValueError(f"NPT final density is non-positive: {density}")
    return {"temperature_k": temp, "density_g_cm3": density, "pressure_bar": pressure}


def _validate_worker(workspace: Path) -> dict[str, Any]:
    stage = workspace / "06_equilibrate" / "npt_smoke"
    log = stage / "npt_smoke.out"
    restart = stage / "npt_smoke.rst7"
    parm7 = stage / "complex_solvated.parm7"
    if not log.is_file() or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError("Phase 12 NPT smoke test did not produce required outputs")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in _FATAL_PATTERNS)]
    if fatal:
        raise ValueError("pmemd.cuda reported fatal/instability diagnostics: " + " | ".join(fatal[:5]))
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError("pmemd.cuda NPT output does not contain a normal-completion marker")
    restraint = _parse_step_zero_restraint(text)
    if restraint > SMOKE_RESTRAINT_ENERGY_MAX:
        raise ValueError(f"NPT step-0 restraint energy {restraint:.3f} kcal/mol is anomalously large")
    observables = _parse_final_observables(text)
    natom = _parse_restart_atom_count(restart)
    _parse_restart_coordinates(restart, natom)
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"NPT smoke restart atom count {natom} disagrees with topology NATOM {match.group(1)}")
    validation = {
        "stage": "npt_smoke",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "results": {
            **observables,
            "step_zero_restraint_energy_kcal_mol": restraint,
            "smoke_test_ps": SMOKE_TEST_PS,
            "restraint_energy_threshold_kcal_mol": SMOKE_RESTRAINT_ENERGY_MAX,
        },
        "checks": {
            "normal_completion": True,
            "finite_temperature_density": True,
            "positive_density": True,
            "step_zero_restraint_reasonable": True,
            "restart_exists": True,
            "finite_coordinates": True,
            "matching_atom_counts": True,
            "no_cuda_or_shake_or_vlimit_instability": True,
            "passed": True,
        },
        "inputs": {
            "parm7": str(parm7),
            "restart": str(workspace / "06_equilibrate" / "heat" / "heat.rst7"),
        },
        "outputs": {
            "input": str(stage / "npt_smoke.in"),
            "log": str(log),
            "restart": str(restart),
            "trajectory": str(stage / "npt_smoke.nc") if (stage / "npt_smoke.nc").is_file() else None,
        },
        "sha256": {p.name: _sha256(p) for p in (stage / "npt_smoke.in", log, restart) if p.is_file()},
    }
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": "npt_smoke", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(restart)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def run_npt_smoke_worker(cfg: PipelineConfig, *, workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    stage = workspace / "06_equilibrate" / "npt_smoke"
    input_path = stage / "npt_smoke.in"
    log_path = stage / "npt_smoke.out"
    restart = stage / "npt_smoke.rst7"
    if not input_path.is_file():
        raise ValueError("Phase 12 npt_smoke.in is missing")
    pmemd = shutil.which("pmemd.cuda")
    if not pmemd:
        raise RuntimeError("pmemd.cuda was not found in PATH; load the Amber 22 module before running Phase 12")
    proc = subprocess.run(
        [pmemd, "-O", "-i", input_path.name, "-o", log_path.name, "-p", "complex_solvated.parm7", "-c", "start.rst7", "-ref", "start.rst7", "-r", restart.name, "-x", "npt_smoke.nc"],
        cwd=stage,
        text=True,
        capture_output=True,
    )
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
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def prepare_npt_smoke(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> NptSmokeSubmission:
    workspace = Path(workspace).resolve()
    parm7, restart, dependency = _required_inputs(workspace)
    stage = workspace / "06_equilibrate" / "npt_smoke"
    stage.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("npt_smoke.in", "npt_smoke.out", "npt_smoke.rst7", "npt_smoke.nc", "validation.json", ".done", "start.rst7", "complex_solvated.parm7"):
        path = stage / name
        if path.exists() or path.is_symlink():
            path.unlink()
    for source, name in ((parm7, "complex_solvated.parm7"), (restart, "start.rst7")):
        (stage / name).symlink_to(source.resolve())
    input_path = stage / "npt_smoke.in"
    input_path.write_text(_render_input(cfg), encoding="utf-8")
    script = stage / "npt_smoke.lsf"
    script.write_text(_render_lsf(cfg, workspace=workspace, stage=stage, dependency=dependency), encoding="utf-8")
    submission_path = stage / "submission.json"
    if dry_run:
        payload = {"stage": "npt_smoke", "status": "dry_run", "submitted": False, "dependency_job_id": dependency, "script": str(script), "input": str(input_path)}
        submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return NptSmokeSubmission(stage, script, submission_path, None, True)
    job_id, output = _submit(script.read_text(encoding="utf-8"))
    payload = {"stage": "npt_smoke", "status": "submitted", "submitted": True, "submitted_at": _utc_now(), "job_id": job_id, "dependency_job_id": dependency, "bsub_output": output, "script": str(script), "input": str(input_path)}
    submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return NptSmokeSubmission(stage, script, submission_path, job_id, False)
