from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import PipelineConfig

_SECTION_RE = re.compile(r"^!entry\.([^.]+)\.unit\.([^\s]+)\s+(.*)$")


@dataclass(frozen=True)
class OffUnit:
    name: str
    atoms: list[dict[str, Any]]
    residues: list[dict[str, Any]]
    positions: list[dict[str, float]]
    connectivity: list[dict[str, int]]

    @property
    def heavy_atom_count(self) -> int:
        return sum(1 for atom in self.atoms if atom.get("atomic_number") != 1)


@dataclass(frozen=True)
class GlycamInspectionResult:
    stage_dir: Path
    extracted_dir: Path
    summary_path: Path
    structure_off: Path
    bacterial_frcmod: Path
    acid_frcmod: Path
    sentinel_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_ignored_zip_member(name: str) -> bool:
    parts = Path(name).parts
    return "__MACOSX" in parts or any(part == ".DS_Store" for part in parts)


def safe_extract_zip(bundle: Path, destination: Path) -> list[Path]:
    """Extract a ZIP without permitting traversal, absolute paths, or symlinks."""
    bundle = Path(bundle).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if _is_ignored_zip_member(name):
                continue
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink entries are not permitted: {info.filename!r}")
            target = (destination / member).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"unsafe ZIP member path: {info.filename!r}") from exc
            if info.is_dir() or name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)
    return extracted


def _parse_schema(schema: str) -> list[tuple[str, str]]:
    tokens = shlex.split(schema)
    if len(tokens) % 2:
        raise ValueError(f"malformed OFF table schema: {schema!r}")
    return [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]


def _convert_off_value(type_name: str, value: str) -> Any:
    if type_name == "str":
        return value
    if type_name == "int":
        return int(value)
    if type_name == "dbl":
        return float(value)
    raise ValueError(f"unsupported OFF field type: {type_name}")


def _parse_table_rows(lines: list[str], schema: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        values = shlex.split(line, posix=True)
        if len(values) != len(schema):
            raise ValueError(f"OFF table row has {len(values)} fields; expected {len(schema)}: {line!r}")
        rows.append({name: _convert_off_value(type_name, value) for (type_name, name), value in zip(schema, values)})
    return rows


def list_off_units(path: Path) -> list[str]:
    units: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        match = _SECTION_RE.match(raw.strip())
        if match:
            units.add(match.group(1))
    return sorted(units)


def parse_off_unit(path: Path, unit_name: str) -> OffUnit:
    """Parse the OFF atom/residue/position/connectivity tables for one LEaP unit."""
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    sections: dict[str, tuple[str, list[str]]] = {}
    current_name: str | None = None

    for raw in lines:
        stripped = raw.strip()
        match = _SECTION_RE.match(stripped)
        if match:
            unit, section, schema = match.groups()
            current_name = section if unit == unit_name else None
            if current_name is not None:
                sections[current_name] = (schema, [])
            continue
        if current_name is not None:
            if stripped.startswith("!"):
                current_name = None
                continue
            sections[current_name][1].append(raw)

    required = {"atoms", "residues", "positions", "connectivity"}
    missing = required.difference(sections)
    if missing:
        available = ", ".join(list_off_units(path)) or "none"
        raise ValueError(
            f"OFF unit {unit_name!r} is missing required section(s) {sorted(missing)}; available units: {available}"
        )

    parsed: dict[str, list[dict[str, Any]]] = {}
    for section in required:
        schema_text, rows = sections[section]
        if not schema_text.startswith("table "):
            raise ValueError(f"OFF section {section!r} is not a table")
        schema = _parse_schema(schema_text[len("table ") :])
        parsed[section] = _parse_table_rows(rows, schema)

    atoms = parsed["atoms"]
    positions = parsed["positions"]
    residues = parsed["residues"]
    connectivity = parsed["connectivity"]
    if len(atoms) != len(positions):
        raise ValueError(f"OFF atom/position count mismatch: {len(atoms)} atoms vs {len(positions)} positions")

    residue_indices = {int(row.get("seq", i + 1)) for i, row in enumerate(residues)}
    atom_records: list[dict[str, Any]] = []
    for index, (atom, pos) in enumerate(zip(atoms, positions), start=1):
        res_index = int(atom.get("resx", 0))
        if res_index not in residue_indices:
            raise ValueError(f"atom {index} references unknown residue index {res_index}")
        atom_records.append(
            {
                "index": index,
                "name": atom.get("name"),
                "type": atom.get("type"),
                "residue_index": res_index,
                "atomic_number": int(atom.get("elmnt", 0)),
                "charge": float(atom.get("chg", 0.0)),
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0)),
            }
        )

    residue_records = [
        {
            "index": int(row.get("seq", i + 1)),
            "name": row.get("name"),
            "start_atom_index": int(row.get("startatomx", 0)),
            "residue_type": row.get("restype"),
        }
        for i, row in enumerate(residues)
    ]

    edge_records: list[dict[str, int]] = []
    for edge in connectivity:
        a = int(edge.get("atom1x", 0))
        b = int(edge.get("atom2x", 0))
        if not (1 <= a <= len(atom_records) and 1 <= b <= len(atom_records)):
            raise ValueError(f"OFF connectivity references out-of-range atom index: {a}, {b}")
        edge_records.append({"atom1": a, "atom2": b, "flags": int(edge.get("flags", 0))})

    return OffUnit(
        name=unit_name,
        atoms=atom_records,
        residues=residue_records,
        positions=[{"x": a["x"], "y": a["y"], "z": a["z"]} for a in atom_records],
        connectivity=edge_records,
    )


def _find_unique_by_basename(root: Path, basename: str) -> Path:
    matches = [p for p in root.rglob(basename) if p.is_file() and not _is_ignored_zip_member(str(p.relative_to(root)))]
    if not matches:
        raise ValueError(f"required GLYCAM bundle file not found: {basename}")
    if len(matches) > 1:
        rel = [str(p.relative_to(root)) for p in matches]
        raise ValueError(f"required GLYCAM bundle file is ambiguous ({basename}): {rel}")
    return matches[0]


def inspect_glycam_bundle(cfg: PipelineConfig, *, workspace: Path) -> GlycamInspectionResult:
    """Inspect the authoritative GLYCAM-Web ZIP and emit an auditable Phase 3 summary."""
    workspace = Path(workspace).resolve()
    stage = workspace / "02_prepare" / "glycam"
    stage.mkdir(parents=True, exist_ok=True)
    extracted = stage / "extracted"
    summary_path = stage / "glycam_summary.json"
    sentinel = stage / ".done"
    if sentinel.exists():
        sentinel.unlink()

    temp_parent = stage
    temp_dir = Path(tempfile.mkdtemp(prefix="extracted.tmp-", dir=temp_parent))
    try:
        safe_extract_zip(cfg.glycam.bundle, temp_dir)
        structure_off = _find_unique_by_basename(temp_dir, "structure.off")
        bacterial = _find_unique_by_basename(temp_dir, cfg.glycam.bacterial_frcmod)
        acid = _find_unique_by_basename(temp_dir, cfg.glycam.acid_frcmod)
        unit = parse_off_unit(structure_off, cfg.glycam.unit_name)

        if cfg.glycam.expected_heavy_atoms is not None and unit.heavy_atom_count != cfg.glycam.expected_heavy_atoms:
            raise ValueError(
                f"GLYCAM heavy-atom count {unit.heavy_atom_count} does not match glycam.expected_heavy_atoms "
                f"{cfg.glycam.expected_heavy_atoms}"
            )
        if cfg.glycam.expected_residues is not None and len(unit.residues) != cfg.glycam.expected_residues:
            raise ValueError(
                f"GLYCAM residue count {len(unit.residues)} does not match glycam.expected_residues "
                f"{cfg.glycam.expected_residues}"
            )

        if extracted.exists():
            shutil.rmtree(extracted)
        temp_dir.rename(extracted)
        temp_dir = Path()  # mark ownership transferred

        structure_off = _find_unique_by_basename(extracted, "structure.off")
        bacterial = _find_unique_by_basename(extracted, cfg.glycam.bacterial_frcmod)
        acid = _find_unique_by_basename(extracted, cfg.glycam.acid_frcmod)

        summary = {
            "stage": "glycam_inspection",
            "status": "done",
            "unit_name": unit.name,
            "available_units": list_off_units(structure_off),
            "source_bundle": str(cfg.glycam.bundle),
            "source_bundle_sha256": _sha256(cfg.glycam.bundle),
            "files": {
                "structure_off": str(structure_off.relative_to(workspace)),
                "structure_off_sha256": _sha256(structure_off),
                "bacterial_frcmod": str(bacterial.relative_to(workspace)),
                "bacterial_frcmod_sha256": _sha256(bacterial),
                "acid_frcmod": str(acid.relative_to(workspace)),
                "acid_frcmod_sha256": _sha256(acid),
            },
            "counts": {
                "atoms": len(unit.atoms),
                "heavy_atoms": unit.heavy_atom_count,
                "residues": len(unit.residues),
                "bonds": len(unit.connectivity),
            },
            "residues": unit.residues,
            "atoms": unit.atoms,
            "connectivity": unit.connectivity,
            "validation": {
                "unit_found": True,
                "expected_heavy_atoms": cfg.glycam.expected_heavy_atoms,
                "heavy_atom_count_ok": cfg.glycam.expected_heavy_atoms is None
                or unit.heavy_atom_count == cfg.glycam.expected_heavy_atoms,
                "expected_residues": cfg.glycam.expected_residues,
                "residue_count_ok": cfg.glycam.expected_residues is None
                or len(unit.residues) == cfg.glycam.expected_residues,
                "passed": True,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sentinel_payload = {
            "stage": "glycam_inspection",
            "status": "done",
            "completed_at": _utc_now(),
            "pipeline_version": __version__,
            "source_bundle_sha256": _sha256(cfg.glycam.bundle),
            "summary_sha256": _sha256(summary_path),
            "unit_name": unit.name,
            "atom_count": len(unit.atoms),
            "heavy_atom_count": unit.heavy_atom_count,
            "residue_count": len(unit.residues),
            "bond_count": len(unit.connectivity),
            "lsf_job_id": os.environ.get("LSB_JOBID"),
        }
        sentinel.write_text(json.dumps(sentinel_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return GlycamInspectionResult(stage, extracted, summary_path, structure_off, bacterial, acid, sentinel)
    finally:
        if temp_dir and temp_dir.exists() and temp_dir != Path("."):
            shutil.rmtree(temp_dir, ignore_errors=True)
