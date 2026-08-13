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
from .npt import _FATAL_PATTERNS, _COMPLETION_MARKERS, _parse_final_observables, _float


@dataclass(frozen=True)
class NptEquilibrationSubmission:
    stage: Path
    script_paths: tuple[Path, Path, Path]
    submission_path: Path
    job_ids: tuple[str | None, str | None, str | None]
    dry_run: bool


_STAGE_NAMES = ("restraint5", "restraint1", "free")
_STAGE_LABELS = {"restraint5": "npt_5", "restraint1": "npt_1", "free": "npt_free"}
_JOB_RE = re.compile(r"Job <(\d+)>")
_TEMP_RE = re.compile(r"\bTEMP\(K\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_PRESS_RE = re.compile(r"\bPRESS\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_DENSITY_RE = re.compile(r"\bDENSITY\s*=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _steps(ps: float, cfg: PipelineConfig) -> int:
    steps = int(round(ps * 1000.0 / cfg.md.production_timestep_fs))
    if steps <= 0:
        raise ValueError("NPT equilibration duration must produce a positive step count")
    return steps


def _render_input(cfg: PipelineConfig, stage_name: str) -> str:
    if stage_name == "restraint5":
        ps, restraint = cfg.equilibration.npt_5_ps, 5.0
    elif stage_name == "restraint1":
        ps, restraint = cfg.equilibration.npt_1_ps, 1.0
    elif stage_name == "free":
        ps, restraint = cfg.equilibration.npt_free_ps, None
    else:
        raise ValueError(f"unknown NPT equilibration stage: {stage_name}")
    ntr = 1 if restraint is not None else 0
    lines = [
        f"NPT equilibration stage {stage_name} for lyso-md",
        "&cntrl",
        "  imin=0,",
        "  irest=1,",
        "  ntx=5,",
        f"  nstlim={_steps(ps, cfg)},",
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
        "  ntr=" + str(ntr) + ",",
    ]
    if restraint is not None:
        lines.extend([
            f"  restraint_wt={restraint:.1f},",
            "  restraintmask='(!:WAT,K+,Cl-)&(!@H=)',",
        ])
    lines.extend([
        "  ntc=2,",
        "  ntf=2,",
        "  ntpr=500,",
        "  ntwx=500,",
        "  iwrap=0,",
        "  ioutfm=1,",
        "/",
        "",
    ])
    return "\n".join(lines)


def _render_lsf(cfg: PipelineConfig, *, workspace: Path, stage: Path, stage_name: str, dependency: str | None) -> str:
    dep = f'\n#BSUB -w "done({dependency})"' if dependency else ""
    label = _STAGE_LABELS[stage_name]
    return f'''#!/bin/bash
#BSUB -P {cfg.scheduler.project}
#BSUB -J {cfg.name}_{label}{dep}
#BSUB -q {cfg.scheduler.gpu_queue}
#BSUB -n {cfg.scheduler.cores}
#BSUB -W 06:00
#BSUB -gpu "{cfg.scheduler.gpu_resource}"
#BSUB -R "rusage[mem={cfg.scheduler.memory}]"
#BSUB -oo {workspace}/logs/{label}.%J.out
#BSUB -eo {workspace}/logs/{label}.%J.err

set -euo pipefail
cd {stage}
test -s complex_solvated.parm7
test -s start.rst7
lyso-md _npt-equilibrate-worker {stage_name} {workspace}/config.yaml
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
    pressures = [_float(x) for x in _PRESS_RE.findall(section)]
    densities = [_float(x) for x in _DENSITY_RE.findall(section)]
    if not temps:
        raise ValueError("NPT equilibration output does not contain a final dynamics temperature")
    if not densities:
        raise ValueError("NPT equilibration output does not contain a final dynamics density")
    result = {"temperature_k": temps[-1], "density_g_cm3": densities[-1], "pressure_bar": pressures[-1] if pressures else math.nan}
    if not all(math.isfinite(v) for v in result.values() if not math.isnan(v)):
        raise ValueError("NPT equilibration observables contain non-finite values")
    if result["density_g_cm3"] <= 0:
        raise ValueError(f"NPT equilibration density is non-positive: {result['density_g_cm3']}")
    return result


def _validate_stage(workspace: Path, stage_name: str) -> dict[str, Any]:
    stage = workspace / "06_equilibrate" / "npt_equilibrate" / stage_name
    log = stage / "stage.out"
    restart = stage / "stage.rst7"
    parm7 = stage / "complex_solvated.parm7"
    if not log.is_file() or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError(f"Phase 13 {stage_name} did not produce required outputs")
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in _FATAL_PATTERNS)]
    if fatal:
        raise ValueError("pmemd.cuda reported fatal/instability diagnostics: " + " | ".join(fatal[:5]))
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError(f"Phase 13 {stage_name} output does not contain a normal-completion marker")
    observables = _parse_observables(text)
    natom = _parse_restart_atom_count(restart)
    _parse_restart_coordinates(restart, natom)
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"NPT equilibration restart atom count {natom} disagrees with topology NATOM {match.group(1)}")
    input_path = stage / "stage.in"
    validation = {
        "stage": f"npt_{stage_name}",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "results": {**observables},
        "checks": {
            "normal_completion": True,
            "finite_temperature_density": True,
            "positive_density": True,
            "restart_exists": True,
            "finite_coordinates": True,
            "matching_atom_counts": True,
            "no_cuda_or_shake_or_vlimit_instability": True,
            "passed": True,
        },
        "inputs": {"parm7": str(parm7), "restart": str(stage / "start.rst7")},
        "outputs": {"input": str(input_path), "log": str(log), "restart": str(restart), "trajectory": str(stage / "stage.nc") if (stage / "stage.nc").is_file() else None},
        "sha256": {p.name: _sha256(p) for p in (input_path, log, restart) if p.is_file()},
    }
    validation["results"].pop("steps", None)
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": validation["stage"], "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(restart)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def run_npt_equilibration_worker(cfg: PipelineConfig, *, workspace: Path, stage_name: str) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    if stage_name not in _STAGE_NAMES:
        raise ValueError(f"unknown Phase 13 stage: {stage_name}")
    stage = workspace / "06_equilibrate" / "npt_equilibrate" / stage_name
    input_path = stage / "stage.in"
    log_path = stage / "stage.out"
    restart = stage / "stage.rst7"
    if not input_path.is_file():
        raise ValueError(f"Phase 13 {stage_name} input is missing")
    pmemd = shutil.which("pmemd.cuda")
    if not pmemd:
        raise RuntimeError("pmemd.cuda was not found in PATH; load the Amber 22 module before running Phase 13")
    proc = subprocess.run([pmemd, "-O", "-i", input_path.name, "-o", log_path.name, "-p", "complex_solvated.parm7", "-c", "start.rst7", "-ref", "start.rst7", "-r", restart.name, "-x", "stage.nc"], cwd=stage, text=True, capture_output=True)
    if proc.stdout and log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STDOUT ---\n" + proc.stdout)
    if proc.stderr and log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"pmemd.cuda exited with status {proc.returncode}; inspect {log_path}")
    validation = _validate_stage(workspace, stage_name)
    validation["process"] = {"returncode": proc.returncode}
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if stage_name == "free":
        final = stage / "stage.rst7"
        aggregate = stage.parent / ".done"
        aggregate.write_text(json.dumps({"stage": "npt_equilibrate", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(stage / "validation.json"), "outputs": [str(final)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def _required_inputs(workspace: Path) -> tuple[Path, Path, str | None]:
    smoke = workspace / "06_equilibrate" / "npt_smoke"
    parm7 = smoke / "complex_solvated.parm7"
    restart = smoke / "npt_smoke.rst7"
    if not (smoke / ".done").is_file():
        submission = smoke / "submission.json"
        if not submission.is_file():
            raise ValueError("Phase 13 requires a completed Phase 12 NPT smoke checkpoint or submission metadata")
        payload = json.loads(submission.read_text(encoding="utf-8"))
        dependency = "<NPT_SMOKE_JOB_ID>" if payload.get("status") == "dry_run" else str(payload["job_id"])
    else:
        dependency = None
    if not parm7.is_file() or parm7.stat().st_size == 0 or not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError("Phase 13 requires the completed Phase 12 NPT smoke topology and restart")
    return parm7, restart, dependency


def prepare_npt_equilibration(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> NptEquilibrationSubmission:
    workspace = Path(workspace).resolve()
    parm7, restart, dependency = _required_inputs(workspace)
    root = workspace / "06_equilibrate" / "npt_equilibrate"
    root.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    jobs: list[str | None] = []
    dependencies: list[str | None] = [dependency, None, None]
    if dry_run:
        dependencies = [dependency, "<NPT_5_JOB_ID>", "<NPT_1_JOB_ID>"]
    for stage_name in _STAGE_NAMES:
        stage = root / stage_name
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("stage.in", "stage.out", "stage.rst7", "stage.nc", "validation.json", ".done", "start.rst7", "complex_solvated.parm7"):
            p = stage / name
            if p.exists() or p.is_symlink():
                p.unlink()
        (stage / "complex_solvated.parm7").symlink_to(parm7.resolve())
        input_path = stage / "stage.in"
        input_path.write_text(_render_input(cfg, stage_name), encoding="utf-8")
        if stage_name == "restraint5":
            start = restart
        elif stage_name == "restraint1":
            start = root / "restraint5" / "stage.rst7"
        else:
            start = root / "restraint1" / "stage.rst7"
        (stage / "start.rst7").symlink_to(start.resolve())
        script = stage / "stage.lsf"
        script.write_text(_render_lsf(cfg, workspace=workspace, stage=stage, stage_name=stage_name, dependency=dependencies[_STAGE_NAMES.index(stage_name)]), encoding="utf-8")
        scripts.append(script)
    submission_path = root / "submission.json"
    if dry_run:
        jobs = [None, None, None]
        payload = {"stage": "npt_equilibrate", "status": "dry_run", "submitted": False, "dependency_job_id": dependency, "scripts": [str(p) for p in scripts]}
        submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return NptEquilibrationSubmission(root, tuple(scripts), submission_path, tuple(jobs), True)
    previous = dependency
    outputs: list[dict[str, Any]] = []
    for i, stage_name in enumerate(_STAGE_NAMES):
        script = scripts[i]
        script_text = script.read_text(encoding="utf-8")
        if previous and f'done({previous})' not in script_text:
            script_text = _render_lsf(cfg, workspace=workspace, stage=script.parent, stage_name=stage_name, dependency=previous)
            script.write_text(script_text, encoding="utf-8")
        job_id, bsub_output = _submit(script_text)
        jobs.append(job_id)
        outputs.append({"stage": stage_name, "job_id": job_id, "dependency_job_id": previous, "bsub_output": bsub_output, "script": str(script)})
        previous = job_id
    payload = {"stage": "npt_equilibrate", "status": "submitted", "submitted": True, "submitted_at": _utc_now(), "jobs": outputs, "scripts": [str(p) for p in scripts]}
    submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return NptEquilibrationSubmission(root, tuple(scripts), submission_path, tuple(jobs), False)
