from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import PipelineConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _find_validation_files(workspace: Path) -> list[Path]:
    roots = [
        workspace / "01_chai",
        workspace / "02_prepare",
        workspace / "03_dry_relax",
        workspace / "04_solvate",
        workspace / "05_minimize",
        workspace / "06_equilibrate",
        workspace / "07_production",
    ]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(root.rglob("validation.json"))
    analysis = workspace / "07_analysis" / "validation.json"
    if analysis.is_file():
        found.append(analysis)
    return sorted(set(found))


def _stage_name(path: Path, workspace: Path, payload: dict[str, Any]) -> str:
    if payload.get("stage"):
        return str(payload["stage"])
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return path.name


def _number_from_dat(path: Path) -> list[float]:
    values: list[float] = []
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            for token in line.split():
                try:
                    value = float(token.replace("D", "E").replace("d", "e"))
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
    return values


def _analysis_metrics(stage: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    mappings = {
        "rmsd_protein_ca.dat": "rmsd_protein_ca",
        "rmsd_glycan.dat": "rmsd_glycan",
        "rmsf_ca.dat": "rmsf_ca",
        "rg_protein.dat": "rg_protein",
        "rg_glycan.dat": "rg_glycan",
        "protein_glycan_contacts.dat": "protein_glycan_contacts",
        "dccm.dat": "dccm",
        "pairwise_rmsd.dat": "pairwise_rmsd",
        "ca_projection.dat": "ca_projection",
        "cluster_summary.dat": "cluster_summary",
    }
    for filename, key in mappings.items():
        path = stage / filename
        if path.is_file() and path.stat().st_size:
            vals = _number_from_dat(path)
            metrics[key] = {"file": str(path), "bytes": path.stat().st_size, "numeric_values": len(vals)}
            if vals:
                metrics[key].update({"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)})
    for filename in ("mode1.nc", "mode2.nc", "mode3.nc", "average_structure.pdb", "processed.nc", "processed.parm7"):
        path = stage / filename
        if path.is_file() and path.stat().st_size:
            metrics.setdefault("files", {})[filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return metrics


def build_qc(cfg: PipelineConfig, *, workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    validations: list[dict[str, Any]] = []

    for path in _find_validation_files(workspace):
        payload = _json(path)
        if payload is None:
            warnings.append({"type": "invalid_validation_json", "path": str(path)})
            continue
        validations.append({"stage": _stage_name(path, workspace, payload), "path": str(path), "status": payload.get("status"), "passed": payload.get("checks", {}).get("passed")})
        checks = payload.get("checks", {})
        if checks.get("passed") is False or payload.get("status") not in (None, "done"):
            failures.append({"type": "stage_validation_failed", "stage": _stage_name(path, workspace, payload), "path": str(path)})

    stage_sentinels = {
        "chai": workspace / "01_chai/.done",
        "glycam": workspace / "02_prepare/glycam/.done",
        "mapping": workspace / "02_prepare/mapping/.done",
        "coordinates": workspace / "02_prepare/coordinate_transfer/.done",
        "protein": workspace / "02_prepare/protein/.done",
        "leap": workspace / "03_dry_relax/.done",
        "hydrogen_relax": workspace / "03_dry_relax/hydrogen_relax/.done",
        "solvate": workspace / "04_solvate/.done",
        "minimize": workspace / "05_minimize/.done",
        "heat": workspace / "06_equilibrate/heat/.done",
        "npt_smoke": workspace / "06_equilibrate/npt_smoke/.done",
        "npt_equilibrate": workspace / "06_equilibrate/.done",
        "production": workspace / "07_production/.done",
        "analysis": workspace / "07_analysis/.done",
    }
    info["checkpoints"] = {name: path.is_file() for name, path in stage_sentinels.items()}
    for name, path in stage_sentinels.items():
        if not path.is_file():
            warnings.append({"type": "missing_checkpoint", "stage": name, "path": str(path)})

    production = _json(workspace / "07_production/production_validation.json")
    if production:
        info["production"] = production.get("results", {})
        info["production"]["checks"] = production.get("checks", {})
    else:
        chunk_validations: list[dict[str, Any]] = []
        for path in sorted((workspace / "07_production").glob("chunk_*/validation.json")):
            payload = _json(path)
            if payload:
                chunk_validations.append(payload)
        if chunk_validations:
            last = chunk_validations[-1]
            info["production"] = {
                "chunks": len(chunk_validations),
                "completed_ns": last.get("results", {}).get("completed_ns"),
                "target_ns": cfg.production.target_ns,
            }

    analysis_stage = workspace / "07_analysis"
    info["analysis"] = _analysis_metrics(analysis_stage) if analysis_stage.is_dir() else {}

    if info.get("production", {}).get("completed_ns") is not None:
        completed = float(info["production"]["completed_ns"])
        if completed < float(cfg.production.target_ns) - 5e-6:
            failures.append({"type": "production_target_not_reached", "completed_ns": completed, "target_ns": cfg.production.target_ns})

    info["validation_files"] = validations
    info["design"] = cfg.name
    info["target_production_ns"] = cfg.production.target_ns
    return {
        "stage": "qc",
        "status": "done" if not failures else "failed",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "hard_failures": failures,
        "warnings": warnings,
        "information": info,
        "summary": {
            "hard_failure_count": len(failures),
            "warning_count": len(warnings),
            "validation_file_count": len(validations),
            "analysis_checkpoint": stage_sentinels["analysis"].is_file(),
        },
    }


def write_qc_report(cfg: PipelineConfig, *, workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    stage = workspace / "07_analysis"
    stage.mkdir(parents=True, exist_ok=True)
    qc = build_qc(cfg, workspace=workspace)
    summary_path = stage / "qc_summary.json"
    report_path = stage / "qc_report.md"
    summary_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# QC Report — {cfg.name}",
        "",
        f"Generated: `{qc['completed_at']}`",
        f"Pipeline version: `{qc['pipeline_version']}`",
        "",
        "## Executive status",
        "",
        f"- Status: **{qc['status'].upper()}**",
        f"- Hard failures: **{qc['summary']['hard_failure_count']}**",
        f"- Warnings: **{qc['summary']['warning_count']}**",
        "",
        "## Hard failures",
        "",
    ]
    if qc["hard_failures"]:
        lines.extend(f"- `{item}`" for item in qc["hard_failures"])
    else:
        lines.append("- None")
    lines += ["", "## Warnings", ""]
    if qc["warnings"]:
        lines.extend(f"- `{item}`" for item in qc["warnings"])
    else:
        lines.append("- None")
    lines += ["", "## Checkpoints", "", "| Stage | Status |", "|---|---|"]
    for name, done in qc["information"]["checkpoints"].items():
        lines.append(f"| {name} | {'DONE' if done else 'PENDING'} |")
    lines += ["", "## Production", ""]
    prod = qc["information"].get("production", {})
    for key in ("completed_ns", "target_ns", "chunks", "temperature_k", "density_g_cm3", "pressure_bar"):
        if key in prod:
            lines.append(f"- {key}: `{prod[key]}`")
    lines += ["", "## Analysis outputs", ""]
    analysis = qc["information"].get("analysis", {})
    for key, value in analysis.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Validation records", "", "| Stage | Status | Passed |", "|---|---|---|"]
    for record in qc["information"]["validation_files"]:
        lines.append(f"| {record['stage']} | {record['status']} | {record['passed']} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    qc["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    return qc
