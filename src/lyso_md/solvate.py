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
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scipy.io import netcdf_file

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
_CHARGE_RE = re.compile(r"Total\s+unperturbed\s+charge:\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)", re.IGNORECASE)
_NATOM_RE = re.compile(r"%FLAG\s+POINTERS.*?%FORMAT\(10I8\)\s*\n\s*(\d+)", re.IGNORECASE | re.DOTALL)
_CDF_MAGIC = b"CDF"
AVOGADRO_SCALED = 6.02214076e-4


@dataclass(frozen=True)
class RestartData:
    natom: int
    coordinates: np.ndarray
    box: tuple[float, float, float, float, float, float] | None


@dataclass(frozen=True)
class SolvationResult:
    stage: Path
    input_path: Path
    log_path: Path
    parm7: Path
    rst7: Path
    pdb: Path
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
    return Environment(loader=FileSystemLoader(str(template_dir)), undefined=StrictUndefined, keep_trailing_newline=True)


def _required_inputs(cfg: PipelineConfig, workspace: Path) -> dict[str, Any]:
    dry = workspace / "03_dry_relax"
    hydrogen = dry / "hydrogen_relax"
    protein_stage = workspace / "02_prepare" / "protein"
    coord_stage = workspace / "02_prepare" / "coordinate_transfer"
    extracted = workspace / "02_prepare" / "glycam" / "extracted"
    protein = protein_stage / "protein_chai.pdb"
    glycan = coord_stage / "glycan_aligned.off"
    bonds = protein_stage / "disulfide_bonds.leap"
    hrelaxed = hydrogen / "complex_hrelaxed.rst7"
    dry_pdb = dry / "complex_dry.pdb"
    dry_parm7 = dry / "complex_dry.parm7"
    for path, label in ((protein, "protein_chai.pdb"), (glycan, "glycan_aligned.off"), (bonds, "disulfide_bonds.leap"), (hrelaxed, "complex_hrelaxed.rst7"), (dry_pdb, "complex_dry.pdb"), (dry_parm7, "complex_dry.parm7")):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Phase 9 requires {label}; complete the earlier preparation stages first")
    for marker, label in ((dry / ".done", "Phase 7 dry LEaP"), (hydrogen / ".done", "Phase 8 hydrogen relaxation"), (coord_stage / ".done", "Phase 5 coordinate transfer"), (protein_stage / ".done", "Phase 6 protein preparation"), (workspace / "02_prepare" / "glycam" / ".done", "Phase 3 GLYCAM inspection")):
        if not marker.is_file():
            raise ValueError(f"Phase 9 requires completed {label} checkpoint")
    bacterial = list(extracted.rglob(cfg.glycam.bacterial_frcmod))
    acid = list(extracted.rglob(cfg.glycam.acid_frcmod))
    if len(bacterial) != 1 or len(acid) != 1:
        raise ValueError("Phase 9 requires exactly one copy of each configured GLYCAM frcmod file")
    try:
        validation = json.loads((dry / "validation.json").read_text(encoding="utf-8"))
        charge = float(validation["charge"]["total_unperturbed"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Phase 9 cannot read the dry-complex total charge from Phase 7 validation.json") from exc
    if not math.isfinite(charge) or abs(charge - round(charge)) > 1e-4:
        raise ValueError(f"Phase 9 requires an integral finite dry-complex charge; found {charge!r}")
    return {"protein": protein, "glycan": glycan, "disulfide_bonds": bonds, "bacterial_frcmod": bacterial[0], "acid_frcmod": acid[0], "dry_pdb": dry_pdb, "dry_parm7": dry_parm7, "hrelaxed": hrelaxed, "dry_validation": dry / "validation.json", "dry_charge": charge}


def _render_probe(cfg: PipelineConfig, paths: dict[str, Any]) -> str:
    env = _template_environment()
    return env.get_template("solvate_probe.in.j2").render(
        protein_ff=cfg.forcefield.protein,
        water_ff=cfg.forcefield.water,
        bacterial_frcmod=paths["bacterial_frcmod"].name,
        acid_frcmod=paths["acid_frcmod"].name,
        glycan_off=paths["glycan"].name,
        protein_pdb=paths["protein"].name,
        disulfide_commands=paths["disulfide_bonds"].read_text(encoding="utf-8").strip(),
        buffer_angstrom=cfg.solvent.buffer_angstrom,
    )


def _render_final(cfg: PipelineConfig, paths: dict[str, Any], nk: int, ncl: int) -> str:
    env = _template_environment()
    return env.get_template("solvate_ions.in.j2").render(
        protein_ff=cfg.forcefield.protein,
        water_ff=cfg.forcefield.water,
        bacterial_frcmod=paths["bacterial_frcmod"].name,
        acid_frcmod=paths["acid_frcmod"].name,
        glycan_off=paths["glycan"].name,
        protein_pdb=paths["protein"].name,
        disulfide_commands=paths["disulfide_bonds"].read_text(encoding="utf-8").strip(),
        buffer_angstrom=cfg.solvent.buffer_angstrom,
        potassium=nk,
        chloride=ncl,
        output_parm7="complex_solvated.parm7",
        output_rst7="complex_solvated.rst7",
        output_pdb="complex_solvated.pdb",
    )


def _read_restart(path: Path) -> RestartData:
    with path.open("rb") as handle:
        magic = handle.read(3)
    if magic == _CDF_MAGIC:
        try:
            with netcdf_file(str(path), mode="r", mmap=False) as nc:
                if "atom" not in nc.dimensions or "coordinates" not in nc.variables:
                    raise ValueError("Amber NetCDF restart lacks atom/coordinates variables")
                natom = int(nc.dimensions["atom"])
                coords = np.asarray(nc.variables["coordinates"].data, dtype=float).copy()
                box = None
                if "cell_lengths" in nc.variables and "cell_angles" in nc.variables:
                    lengths = np.asarray(nc.variables["cell_lengths"].data, dtype=float).reshape(-1)
                    angles = np.asarray(nc.variables["cell_angles"].data, dtype=float).reshape(-1)
                    if len(lengths) >= 3 and len(angles) >= 3:
                        box = tuple(float(x) for x in (*lengths[:3], *angles[:3]))
        except Exception as exc:
            raise ValueError(f"cannot parse Amber NetCDF restart: {path}") from exc
        if coords.shape != (natom, 3):
            raise ValueError(f"Amber NetCDF restart coordinates have shape {coords.shape}; expected ({natom}, 3)")
        if not np.isfinite(coords).all():
            raise ValueError(f"Amber restart contains non-finite coordinates: {path}")
        return RestartData(natom, coords, box)
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
        raise ValueError(f"Amber restart has only {len(values)} numeric values; expected at least {needed}")
    coords = np.asarray(values[:needed], dtype=float).reshape(natom, 3)
    box = None
    if len(values) >= needed + 6:
        box_values = values[-6:]
        if all(math.isfinite(x) for x in box_values):
            box = tuple(float(x) for x in box_values)
    if not np.isfinite(coords).all():
        raise ValueError(f"Amber restart contains non-finite coordinates: {path}")
    return RestartData(natom, coords, box)


def _write_restart_coordinates(path: Path, coordinates: np.ndarray, box: tuple[float, float, float, float, float, float]) -> None:
    data = _read_restart(path)
    if coordinates.shape != (data.natom, 3):
        raise ValueError(f"cannot write {path}: coordinate shape {coordinates.shape} != ({data.natom}, 3)")
    if path.read_bytes()[:3] == _CDF_MAGIC:
        with netcdf_file(str(path), mode="a", mmap=False) as nc:
            nc.variables["coordinates"][:] = coordinates
            if "cell_lengths" in nc.variables:
                nc.variables["cell_lengths"][:] = np.asarray(box[:3], dtype=float)
            if "cell_angles" in nc.variables:
                nc.variables["cell_angles"][:] = np.asarray(box[3:], dtype=float)
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = lines[0] if lines else "lyso-md solvated restart"
    numeric: list[float] = []
    for line in lines[2:]:
        for token in line.split():
            try:
                numeric.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                pass
    if len(numeric) < 3 * data.natom + 6:
        raise ValueError("ASCII solvated restart lacks the expected periodic box record")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n{data.natom:6d}\n")
        flat = coordinates.reshape(-1)
        for i in range(0, len(flat), 6):
            handle.write("".join(f"{value:12.7f}" for value in flat[i:i + 6]) + "\n")
        handle.write("".join(f"{value:12.7f}" for value in box) + "\n")


def _transfer_pdb_solute_coordinates(path: Path, coordinates: np.ndarray) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_indices = [i for i, line in enumerate(lines) if line.startswith(("ATOM  ", "HETATM"))]
    if len(atom_indices) < len(coordinates):
        raise ValueError(f"solvated PDB contains {len(atom_indices)} atoms; expected at least {len(coordinates)} solute atoms")
    for idx, xyz in zip(atom_indices[: len(coordinates)], coordinates):
        line = lines[idx]
        if len(line) < 54:
            raise ValueError(f"malformed solvated PDB coordinate line: {line!r}")
        lines[idx] = line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_parm7_atoms(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _NATOM_RE.search(text)
    if not match:
        raise ValueError(f"cannot parse NATOM from Amber topology: {path}")
    return int(match.group(1))


def _count_ions(path: Path) -> tuple[int, int]:
    potassium = chloride = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue = line[17:20].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        if residue == "K+" or element == "K":
            potassium += 1
        elif residue in {"Cl-", "CL-"} or element.upper() == "CL":
            chloride += 1
    return potassium, chloride


def _box_volume(box: tuple[float, float, float, float, float, float]) -> float:
    a, b, c, alpha, beta, gamma = box
    if min(a, b, c) <= 0:
        raise ValueError(f"invalid periodic box lengths: {box[:3]}")
    ar, br, gr = map(math.radians, (alpha, beta, gamma))
    value = 1.0 - math.cos(ar) ** 2 - math.cos(br) ** 2 - math.cos(gr) ** 2 + 2.0 * math.cos(ar) * math.cos(br) * math.cos(gr)
    if value <= 0:
        raise ValueError(f"invalid periodic box angles: {box[3:]}")
    return a * b * c * math.sqrt(value)


def calculate_kcl_counts(concentration_molar: float, volume_angstrom3: float, solute_charge: float) -> tuple[int, int, int]:
    if concentration_molar < 0 or volume_angstrom3 <= 0:
        raise ValueError("salt concentration must be nonnegative and box volume must be positive")
    if abs(solute_charge - round(solute_charge)) > 1e-4:
        raise ValueError(f"solute charge must be integral before ion counting: {solute_charge}")
    q = int(round(solute_charge))
    pairs = int(math.floor(concentration_molar * AVOGADRO_SCALED * volume_angstrom3 + 0.5))
    if q > 0:
        return pairs, pairs, pairs + q
    if q < 0:
        return pairs, pairs + abs(q), pairs
    return pairs, pairs, pairs


def _parse_charge(text: str) -> float:
    matches = _CHARGE_RE.findall(text)
    if not matches:
        raise ValueError("LEaP log does not report 'Total unperturbed charge'")
    return float(matches[-1].replace("D", "E").replace("d", "e"))


def _find_fatal_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in _FATAL_PATTERNS)]


def _run_tleap(stage: Path, input_name: str, log_name: str) -> Path:
    tleap = shutil.which("tleap")
    if not tleap:
        raise RuntimeError("tleap was not found in PATH; load the Amber 22 module before running Phase 9")
    proc = subprocess.run([tleap, "-f", input_name], cwd=stage, text=True, capture_output=True)
    log_path = stage / log_name
    log_path.write_text(proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    fatal = _find_fatal_lines(log_path.read_text(encoding="utf-8", errors="replace"))
    if proc.returncode != 0:
        raise RuntimeError(f"tleap exited with status {proc.returncode}; see {log_path}")
    if fatal:
        raise ValueError("LEaP reported fatal/parameter errors: " + " | ".join(fatal[:5]))
    return log_path


def solvate_and_ionize(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> SolvationResult:
    """Phase 9: exact-box LEaP probe followed by one integrated solvation/ionization LEaP session."""
    workspace = Path(workspace).resolve()
    paths = _required_inputs(cfg, workspace)
    stage = workspace / "04_solvate"
    stage.mkdir(parents=True, exist_ok=True)
    probe_input = stage / "solvate_probe.in"
    probe_log = stage / "solvate_probe.log"
    probe_parm7 = stage / "solvated_probe.parm7"
    probe_rst7 = stage / "solvated_probe.rst7"
    probe_pdb = stage / "solvated_probe.pdb"
    input_path = stage / "solvate_ionize.in"
    log_path = stage / "solvate.log"
    parm7 = stage / "complex_solvated.parm7"
    rst7 = stage / "complex_solvated.rst7"
    pdb = stage / "complex_solvated.pdb"
    validation_path = stage / "validation.json"
    sentinel_path = stage / ".done"
    for path in (probe_input, probe_log, probe_parm7, probe_rst7, probe_pdb, input_path, log_path, parm7, rst7, pdb, validation_path, sentinel_path):
        if path.exists() or path.is_symlink():
            path.unlink()

    for key in ("protein", "glycan", "bacterial_frcmod", "acid_frcmod", "disulfide_bonds"):
        target = stage / paths[key].name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(paths[key].resolve())

    probe_input.write_text(_render_probe(cfg, paths), encoding="utf-8")
    result = SolvationResult(stage, input_path, log_path, parm7, rst7, pdb, validation_path, sentinel_path, dry_run)
    if dry_run:
        # The exact ionized LEaP script is intentionally not emitted until the
        # probe supplies the actual box volume. This avoids guessing solvateOct's
        # geometry and keeps the scientific salt calculation tied to Amber's box.
        return result

    _run_tleap(stage, probe_input.name, probe_log.name)
    probe = _read_restart(probe_rst7)
    if probe.box is None:
        raise ValueError("LEaP solvation probe did not produce periodic box vectors")
    volume = _box_volume(probe.box)
    dry = _read_restart(paths["hrelaxed"])
    dry_atoms = _count_parm7_atoms(paths["dry_parm7"])
    if dry.natom != dry_atoms:
        raise ValueError(f"Phase 8 hydrogen-relaxed restart disagrees with dry-complex topology atom count: {dry.natom} != {dry_atoms}")
    if probe.natom <= dry.natom:
        raise ValueError(f"solvation probe produced {probe.natom} atoms; expected more than dry solute {dry.natom}")
    charge = float(paths["dry_charge"])
    pairs, nk, ncl = calculate_kcl_counts(cfg.solvent.concentration_molar, volume, charge)
    input_path.write_text(_render_final(cfg, paths, nk, ncl), encoding="utf-8")

    _run_tleap(stage, input_path.name, log_path.name)
    final = _read_restart(rst7)
    if final.box is None:
        raise ValueError("final solvated restart does not contain periodic box vectors")
    if not np.allclose(np.asarray(final.box), np.asarray(probe.box), rtol=0, atol=1e-5):
        raise ValueError("final solvated restart box vectors differ from the solvation probe box")
    final_volume = _box_volume(final.box)
    final_pairs, expected_nk, expected_ncl = calculate_kcl_counts(cfg.solvent.concentration_molar, final_volume, charge)
    if (final_pairs, expected_nk, expected_ncl) != (pairs, nk, ncl):
        raise ValueError(f"final box changes the requested KCl count: probe={pairs}/{nk}/{ncl}, final={final_pairs}/{expected_nk}/{expected_ncl}")
    if final.natom <= dry.natom:
        raise ValueError(f"final solvated system has {final.natom} atoms; expected more than dry solute {dry.natom}")
    transferred = final.coordinates.copy()
    transferred[: dry.natom] = dry.coordinates
    _write_restart_coordinates(rst7, transferred, final.box)
    _transfer_pdb_solute_coordinates(pdb, dry.coordinates)
    final = _read_restart(rst7)
    if not np.allclose(final.coordinates[: dry.natom], dry.coordinates, rtol=0, atol=1e-6):
        raise ValueError("hydrogen-relaxed solute coordinates were not transferred exactly into the solvated restart")
    if not np.isfinite(final.coordinates).all():
        raise ValueError("final solvated restart contains non-finite coordinates")
    parm_atoms = _count_parm7_atoms(parm7)
    pdb_atoms = sum(1 for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(("ATOM  ", "HETATM")))
    if not (parm_atoms == final.natom == pdb_atoms):
        raise ValueError(f"solvated output atom counts disagree: parm7={parm_atoms}, rst7={final.natom}, pdb={pdb_atoms}")
    potassium, chloride = _count_ions(pdb)
    if (potassium, chloride) != (nk, ncl):
        raise ValueError(f"final ion counts disagree with requested counts: K+={potassium}/{nk}, Cl-={chloride}/{ncl}")
    charge_final = _parse_charge(log_path.read_text(encoding="utf-8", errors="replace"))
    if abs(charge_final) > 1e-4:
        raise ValueError(f"final solvated system is not neutral: {charge_final:.8f}")

    validation = {
        "stage": "solvate",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "water": {"model": cfg.forcefield.water, "buffer_angstrom": cfg.solvent.buffer_angstrom},
        "box": {"lengths_angstrom": list(final.box[:3]), "angles_degrees": list(final.box[3:]), "volume_angstrom3": final_volume},
        "salt": {"type": cfg.solvent.salt, "concentration_molar": cfg.solvent.concentration_molar, "pairs": pairs, "potassium": nk, "chloride": ncl},
        "charge": {"dry_solute": charge, "final": charge_final, "neutral": abs(charge_final) <= 1e-4},
        "counts": {"dry_solute_atoms": dry.natom, "probe_atoms": probe.natom, "final_atoms": final.natom, "added_ions": nk + ncl},
        "checks": {"periodic_box": True, "matching_atom_counts": True, "solute_coordinates_transferred": True, "box_preserved": True, "finite_coordinates": True, "ion_counts_match": True, "neutral_final_system": True, "typed_sources_used_in_final_leap": True, "no_loadAmberParm": True, "passed": True},
        "inputs": {k: str(v) for k, v in paths.items() if k != "dry_charge"},
        "outputs": {p.name: str(p) for p in (probe_input, probe_log, input_path, log_path, parm7, rst7, pdb)},
        "sha256": {p.name: _sha256(p) for p in (probe_input, probe_log, input_path, log_path, parm7, rst7, pdb)},
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sentinel_path.write_text(json.dumps({"stage": "solvate", "status": "done", "completed_at": validation["completed_at"], "pipeline_version": __version__, "validation": str(validation_path), "outputs": [str(parm7), str(rst7), str(pdb)]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
