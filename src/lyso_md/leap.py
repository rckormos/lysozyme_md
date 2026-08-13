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

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import __version__
from .config import PipelineConfig

_FATAL_PATTERNS = (
    re.compile(r"\bFATAL\b", re.IGNORECASE),
    re.compile(r"\bunknown residue\b", re.IGNORECASE),
    re.compile(r"\bunknown atom\b", re.IGNORECASE),
    re.compile(r"\batom\s+without\s+(?:a\s+)?type\b", re.IGNORECASE),
    re.compile(r"\bmissing\s+(?:force[- ]field\s+)?parameter\b", re.IGNORECASE),
    re.compile(r"\bcould not find\b", re.IGNORECASE),
)
_CHARGE_RE = re.compile(r"Total\s+unperturbed\s+charge:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE)
_NATOM_RE = re.compile(r"%FLAG\s+POINTERS.*?%FORMAT\(10I8\)\s*\n\s*(\d+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LeapAssemblyResult:
    stage_dir: Path
    input_path: Path
    log_path: Path
    complex_pdb: Path
    parm7: Path
    rst7: Path
    validation_path: Path
    sentinel_path: Path
    dry_run: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _template_environment() -> Environment:
    template_dir = Path(__file__).resolve().parents[2] / "templates" / "leap"
    if not template_dir.is_dir():
        raise ValueError(f"LEaP template directory is missing: {template_dir}")
    return Environment(loader=FileSystemLoader(str(template_dir)), undefined=StrictUndefined, keep_trailing_newline=True)


def _required_inputs(cfg: PipelineConfig, workspace: Path) -> dict[str, Path]:
    glycam_stage = workspace / "02_prepare" / "glycam"
    protein_stage = workspace / "02_prepare" / "protein"
    coord_stage = workspace / "02_prepare" / "coordinate_transfer"
    extracted = glycam_stage / "extracted"
    structure_off = extracted / "structure" / "structure.off"
    if not structure_off.is_file():
        candidates = list(extracted.rglob("structure.off"))
        if len(candidates) != 1:
            raise ValueError("Phase 7 requires exactly one extracted structure.off")
        structure_off = candidates[0]
    bacterial = list(extracted.rglob(cfg.glycam.bacterial_frcmod))
    acid = list(extracted.rglob(cfg.glycam.acid_frcmod))
    if len(bacterial) != 1 or len(acid) != 1:
        raise ValueError("Phase 7 requires exactly one copy of each configured GLYCAM frcmod file")
    protein = protein_stage / "protein_chai.pdb"
    bonds = protein_stage / "disulfide_bonds.leap"
    aligned = coord_stage / "glycan_aligned.off"
    for path, label in ((protein, "protein_chai.pdb"), (bonds, "disulfide_bonds.leap"), (aligned, "glycan_aligned.off")):
        if not path.is_file():
            raise ValueError(f"Phase 7 requires {label}; complete Phase 5/6 first")
    if not (glycam_stage / ".done").is_file():
        raise ValueError("Phase 7 requires completed Phase 3 GLYCAM inspection")
    if not (coord_stage / ".done").is_file():
        raise ValueError("Phase 7 requires completed Phase 5 coordinate transfer")
    if not (protein_stage / ".done").is_file():
        raise ValueError("Phase 7 requires completed Phase 6 protein preparation")
    return {
        "protein": protein,
        "glycan": aligned,
        "structure_off": structure_off,
        "bacterial_frcmod": bacterial[0],
        "acid_frcmod": acid[0],
        "disulfide_bonds": bonds,
    }


def _render_input(cfg: PipelineConfig, paths: dict[str, Path], output_dir: Path) -> str:
    env = _template_environment()
    template = env.get_template("dry_complex.in.j2")
    return template.render(
        protein_ff=cfg.forcefield.protein,
        glycan_ff=cfg.forcefield.glycan,
        bacterial_frcmod=paths["bacterial_frcmod"].name,
        acid_frcmod=paths["acid_frcmod"].name,
        glycan_off=paths["glycan"].name,
        protein_pdb=paths["protein"].name,
        disulfide_commands=paths["disulfide_bonds"].read_text(encoding="utf-8").strip(),
        output_pdb="complex_dry.pdb",
        output_parm7="complex_dry.parm7",
        output_rst7="complex_dry.rst7",
    )


def _parse_total_charge(log_text: str) -> float:
    matches = _CHARGE_RE.findall(log_text)
    if not matches:
        raise ValueError("LEaP log does not report 'Total unperturbed charge'")
    return float(matches[-1])


def _find_fatal_lines(log_text: str) -> list[str]:
    return [line.strip() for line in log_text.splitlines() if any(p.search(line) for p in _FATAL_PATTERNS)]


def _count_pdb_atoms(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(("ATOM  ", "HETATM")))


def _count_restart_atoms(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    if len(lines) < 2:
        raise ValueError(f"Amber restart is too short: {path}")
    try:
        return int(lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot parse atom count from Amber restart: {path}") from exc


def _count_parm7_atoms(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _NATOM_RE.search(text)
    if not match:
        raise ValueError(f"cannot parse NATOM from Amber topology: {path}")
    return int(match.group(1))


def _validate_outputs(stage: Path, log_path: Path, input_path: Path, paths: dict[str, Path], cfg: PipelineConfig) -> dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    fatal_lines = _find_fatal_lines(log_text)
    if fatal_lines:
        raise ValueError("LEaP reported fatal/parameter errors: " + " | ".join(fatal_lines[:5]))
    charge = _parse_total_charge(log_text)
    charge_integral = abs(charge - round(charge)) <= 1e-4
    if not charge_integral:
        raise ValueError(f"LEaP complex charge is non-integral: {charge:.8f}")

    pdb = stage / "complex_dry.pdb"
    parm7 = stage / "complex_dry.parm7"
    rst7 = stage / "complex_dry.rst7"
    for path in (pdb, parm7, rst7):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"LEaP did not produce a valid required output: {path}")

    pdb_atoms = _count_pdb_atoms(pdb)
    parm_atoms = _count_parm7_atoms(parm7)
    rst_atoms = _count_restart_atoms(rst7)
    if not (pdb_atoms == parm_atoms == rst_atoms):
        raise ValueError(f"LEaP output atom counts disagree: PDB={pdb_atoms}, parm7={parm_atoms}, rst7={rst_atoms}")

    finite_pdb = True
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"malformed LEaP output PDB coordinate line: {line!r}") from exc
        if not all(map(lambda x: x == x and abs(x) != float("inf"), coords)):
            finite_pdb = False
            break
    if not finite_pdb:
        raise ValueError("LEaP output contains non-finite coordinates")

    return {
        "stage": "dry_leap",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "forcefield": {"protein": cfg.forcefield.protein, "glycan": cfg.forcefield.glycan},
        "charge": {"total_unperturbed": charge, "integral": charge_integral},
        "counts": {"pdb_atoms": pdb_atoms, "parm7_atoms": parm_atoms, "rst7_atoms": rst_atoms},
        "checks": {"fatal_errors_absent": True, "charge_integral": charge_integral, "matching_atom_counts": True, "finite_coordinates": finite_pdb, "passed": True},
        "inputs": {key: str(value) for key, value in paths.items()},
        "outputs": {name: str(stage / name) for name in ("complex_dry.pdb", "complex_dry.parm7", "complex_dry.rst7", input_path.name, log_path.name)},
        "sha256": {name: _sha256(stage / name) for name in ("complex_dry.pdb", "complex_dry.parm7", "complex_dry.rst7", input_path.name, log_path.name)},
    }


def assemble_dry_complex(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> LeapAssemblyResult:
    """Phase 7: assemble the prepared protein/glycan complex in Amber LEaP."""
    workspace = Path(workspace).resolve()
    paths = _required_inputs(cfg, workspace)
    stage = workspace / "03_dry_relax"
    stage.mkdir(parents=True, exist_ok=True)
    input_path = stage / "leap.in"
    log_path = stage / "leap.log"
    validation_path = stage / "validation.json"
    sentinel_path = stage / ".done"
    for path in (input_path, log_path, validation_path, sentinel_path):
        if path.exists():
            path.unlink()

    # LEaP resolves frcmods/off/PDBs relative to its working directory.
    for key in ("protein", "glycan", "bacterial_frcmod", "acid_frcmod", "disulfide_bonds"):
        target = stage / paths[key].name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(paths[key].resolve())
    rendered = _render_input(cfg, paths, stage)
    input_path.write_text(rendered, encoding="utf-8")

    result = LeapAssemblyResult(stage, input_path, log_path, stage / "complex_dry.pdb", stage / "complex_dry.parm7", stage / "complex_dry.rst7", validation_path, sentinel_path, dry_run)
    if dry_run:
        return result

    tleap = shutil.which("tleap")
    if not tleap:
        raise RuntimeError("tleap was not found in PATH; load the Amber 22 module before running Phase 7")
    proc = subprocess.run([tleap, "-f", input_path.name], cwd=stage, text=True, capture_output=True)
    log_path.write_text(proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"tleap exited with status {proc.returncode}; see {log_path}")
    validation = _validate_outputs(stage, log_path, input_path, paths, cfg)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sentinel_path.write_text(json.dumps({"stage": "dry_leap", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(validation_path), "outputs": [str(result.complex_pdb), str(result.parm7), str(result.rst7)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
