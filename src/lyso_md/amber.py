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

import numpy as np
from scipy.io import netcdf_file

from . import __version__
from .config import PipelineConfig

_FINAL_RE = re.compile(
    r"^\s*(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+(.+?)\s*$",
    re.MULTILINE,
)
_COMPLETION_MARKERS = ("FINAL RESULTS", "5.  TIMINGS", "5. TIMINGS")
_WARNING_PATTERNS = ("Floating point exception", "floating point exception", "SIGFPE", "floating-point exception")


@dataclass(frozen=True)
class HydrogenRelaxResult:
    stage: Path
    input_path: Path
    output_path: Path
    log_path: Path
    restart_path: Path
    validation_path: Path
    sentinel_path: Path
    dry_run: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_inputs(workspace: Path) -> tuple[Path, Path]:
    stage7 = workspace / "03_dry_relax"
    parm7 = stage7 / "complex_dry.parm7"
    rst7 = stage7 / "complex_dry.rst7"
    if not (stage7 / ".done").is_file():
        raise ValueError("Phase 8 requires a completed Phase 7 dry LEaP checkpoint")
    for path in (parm7, rst7):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Phase 8 requires {path.name} from Phase 7")
    return parm7, rst7


def _render_input(cfg: PipelineConfig) -> str:
    # Keep this intentionally explicit: this stage is a fixed scientific protocol,
    # with only the documented step count/configurable cutoff family carried in.
    return f"""Hydrogen relaxation for lyso-md\n&cntrl\n  imin=1,\n  maxcyc={cfg.equilibration.hydrogen_relax_steps},\n  ncyc={cfg.equilibration.hydrogen_relax_steps // 2},\n  ntb=0,\n  igb=0,\n  cut=1000.0,\n  ntr=1,\n  restraint_wt=100.0,\n  restraintmask='!@H=',\n  ntpr=50,\n/\n"""


def _read_restart(path: Path) -> tuple[int, list[float]]:
    """Read either an ASCII Amber restart or Amber NetCDF restart.

    Amber 22 defaults to ``ntxo=2``, which writes the restart as NetCDF.
    The Phase 8 output therefore cannot be parsed by assuming the second
    text line contains NATOM.
    """
    with path.open("rb") as handle:
        magic = handle.read(3)

    if magic == b"CDF":
        try:
            with netcdf_file(str(path), mode="r", mmap=False) as nc:
                if "atom" not in nc.dimensions or "coordinates" not in nc.variables:
                    raise ValueError("Amber NetCDF restart lacks atom/coordinates variables")
                natom = int(nc.dimensions["atom"])
                coords = np.asarray(nc.variables["coordinates"].data, dtype=float)
        except Exception as exc:
            raise ValueError(f"cannot parse Amber NetCDF restart: {path}") from exc
        if coords.shape != (natom, 3):
            raise ValueError(
                f"Amber NetCDF restart coordinates have shape {coords.shape}; expected ({natom}, 3)"
            )
        flat = coords.reshape(-1).tolist()
        if not np.isfinite(coords).all():
            raise ValueError("hydrogen-relaxation restart contains non-finite coordinates")
        return natom, flat

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Amber restart is too short: {path}")
    try:
        natom = int(lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot parse atom count from Amber restart: {path}") from exc
    values: list[float] = []
    for line in lines[2:]:
        for token in line.split():
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
    needed = 3 * natom
    if len(values) < needed:
        raise ValueError(f"Amber restart has only {len(values)} coordinate values; expected at least {needed}")
    coords = values[:needed]
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("hydrogen-relaxation restart contains non-finite coordinates")
    return natom, coords


def _parse_restart_atom_count(path: Path) -> int:
    return _read_restart(path)[0]


def _parse_restart_coordinates(path: Path, natom: int) -> list[float]:
    parsed_natom, coords = _read_restart(path)
    if parsed_natom != natom:
        raise ValueError(f"Amber restart atom count changed while reading {path}: {parsed_natom} != {natom}")
    return coords


def _parse_final_results(text: str) -> dict[str, Any]:
    if not any(marker in text for marker in _COMPLETION_MARKERS):
        raise ValueError("pmemd output does not contain a normal-completion marker")
    match = _FINAL_RE.search(text)
    if not match:
        raise ValueError("pmemd output does not contain a parseable FINAL RESULTS energy/gradient row")
    step = int(match.group(1))
    energy = float(match.group(2).replace("D", "E").replace("d", "e"))
    rms = float(match.group(3).replace("D", "E").replace("d", "e"))
    gmax = float(match.group(4).replace("D", "E").replace("d", "e"))
    for name, value in (("energy", energy), ("rms", rms), ("gmax", gmax)):
        if not math.isfinite(value):
            raise ValueError(f"pmemd FINAL RESULTS contains non-finite {name}")
    return {"step": step, "energy": energy, "rms": rms, "gmax": gmax}


def _validate_output(stage: Path, output_path: Path, restart_path: Path, log_path: Path, parm7: Path, proc_returncode: int) -> dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    final = _parse_final_results(log_text)
    warnings = [line.strip() for line in log_text.splitlines() if any(pattern in line for pattern in _WARNING_PATTERNS)]
    if not restart_path.is_file() or restart_path.stat().st_size == 0:
        raise ValueError(f"pmemd did not produce a valid restart: {restart_path}")
    natom = _parse_restart_atom_count(restart_path)
    _parse_restart_coordinates(restart_path, natom)
    # Parse NATOM from the Phase 7 topology using the same simple POINTERS layout
    # already validated by LEaP.  The restart count must agree.
    parm_text = parm7.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"%FLAG POINTERS\s+%FORMAT\([^\n]+\)\s*\n\s*(\d+)", parm_text)
    if match and int(match.group(1)) != natom:
        raise ValueError(f"hydrogen-relaxation restart atom count {natom} disagrees with parm7 NATOM {match.group(1)}")
    if proc_returncode != 0 and not final:
        raise RuntimeError(f"pmemd exited with status {proc_returncode}")
    return {
        "stage": "hydrogen_relax",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "process": {"returncode": proc_returncode, "nonzero_exit_with_normal_completion": proc_returncode != 0},
        "results": final,
        "checks": {
            "normal_completion": True,
            "finite_energy_gradient": True,
            "restart_exists": True,
            "finite_coordinates": True,
            "matching_atom_counts": True,
            "passed": True,
        },
        "warnings": warnings,
        "inputs": {"parm7": str(parm7), "restart": str(stage / "complex_dry.rst7")},
        "outputs": {"input": str(stage / "hrelax.in"), "log": str(log_path), "restart": str(output_path)},
        "sha256": {name: _sha256(path) for name, path in (("hrelax.in", stage / "hrelax.in"), ("hrelax.out", log_path), ("complex_hrelaxed.rst7", output_path))},
    }


def relax_hydrogens(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> HydrogenRelaxResult:
    """Phase 8: CPU pmemd nonperiodic restrained hydrogen relaxation."""
    workspace = Path(workspace).resolve()
    parm7, rst7 = _required_inputs(workspace)
    stage = workspace / "03_dry_relax" / "hydrogen_relax"
    stage.mkdir(parents=True, exist_ok=True)
    input_path = stage / "hrelax.in"
    log_path = stage / "hrelax.out"
    output_path = stage / "complex_hrelaxed.rst7"
    validation_path = stage / "validation.json"
    sentinel_path = stage / ".done"
    for path in (input_path, log_path, output_path, validation_path, sentinel_path):
        if path.exists() or path.is_symlink():
            path.unlink()
    # Amber reads these relative to its working directory.
    for source in (parm7, rst7):
        target = stage / source.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
    input_path.write_text(_render_input(cfg), encoding="utf-8")
    result = HydrogenRelaxResult(stage, input_path, output_path, log_path, output_path, validation_path, sentinel_path, dry_run)
    if dry_run:
        return result
    pmemd = shutil.which("pmemd")
    if not pmemd:
        raise RuntimeError("pmemd was not found in PATH; load the Amber 22 module before running Phase 8")
    cmd = [pmemd, "-O", "-i", input_path.name, "-o", log_path.name, "-p", parm7.name, "-c", rst7.name, "-ref", rst7.name, "-r", output_path.name]
    proc = subprocess.run(cmd, cwd=stage, text=True, capture_output=True)
    # pmemd writes the scientific report to -o. Preserve that file verbatim and
    # append stderr so warnings/status diagnostics are not lost. If a mock or
    # unusual build does not create -o, fall back to captured stdout.
    combined = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else proc.stdout
    if proc.stdout.strip() and proc.stdout.strip() not in combined:
        combined += "\n--- STDOUT ---\n" + proc.stdout
    if proc.stderr:
        combined += "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0 and not any(marker in combined for marker in _COMPLETION_MARKERS):
        raise RuntimeError(f"pmemd exited with status {proc.returncode}; see {log_path}")
    validation = _validate_output(stage, output_path, output_path, log_path, parm7, proc.returncode)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sentinel = {
        "stage": "hydrogen_relax",
        "status": "done",
        "completed_at": validation["completed_at"],
        "pipeline_version": __version__,
        "validation": str(validation_path),
        "outputs": [str(output_path)],
    }
    sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
