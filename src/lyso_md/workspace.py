from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import PipelineConfig, config_as_yaml_dict

WORKSPACE_DIRS = (
    "input",
    "01_chai",
    "02_prepare",
    "03_dry_relax",
    "04_solvate",
    "05_equilibrate",
    "06_production",
    "07_analysis",
    "logs",
    ".lyso-md/init",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_existing(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.backup-{stamp}")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.backup-{stamp}-{index}")
        index += 1
    path.rename(candidate)
    return candidate


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.symlink_to(src.resolve())


def _workspace_config(cfg: PipelineConfig) -> dict[str, Any]:
    data = config_as_yaml_dict(cfg)
    data["protein"]["fasta"] = "input/sequence.fasta"
    data["glycam"]["bundle"] = "input/glycam_structure.zip"
    return data


def initialize_workspace(
    cfg: PipelineConfig,
    *,
    source_config: Path,
    workspace_root: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path | None]:
    """Create a design workspace without deleting pre-existing results.

    If ``force`` is requested and the target exists, the old workspace is renamed
    to a timestamped sibling backup before a fresh workspace is created.
    """
    source_config = source_config.expanduser().resolve()
    root = (workspace_root or source_config.parent).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / cfg.name
    backup: Path | None = None

    if workspace.exists() or workspace.is_symlink():
        if not force:
            raise FileExistsError(
                f"workspace already exists: {workspace}; use --force to preserve it as a timestamped backup"
            )
        backup = _backup_existing(workspace)

    try:
        for rel in WORKSPACE_DIRS:
            (workspace / rel).mkdir(parents=True, exist_ok=True)

        fasta_dst = workspace / "input/sequence.fasta"
        glycam_dst = workspace / "input/glycam_structure.zip"
        _symlink(cfg.protein.fasta, fasta_dst)
        _symlink(cfg.glycam.bundle, glycam_dst)

        smiles_path = workspace / "input/glycan.smiles"
        smiles_path.write_text(cfg.chai.glycan_smiles.strip() + "\n", encoding="utf-8")

        normalized = _workspace_config(cfg)
        config_dst = workspace / "config.yaml"
        config_dst.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")

        input_entries = []
        for logical_name, source, linked in (
            ("protein_fasta", cfg.protein.fasta, fasta_dst),
            ("glycam_bundle", cfg.glycam.bundle, glycam_dst),
        ):
            source = source.resolve()
            input_entries.append(
                {
                    "name": logical_name,
                    "source": str(source),
                    "workspace_path": str(linked.relative_to(workspace)),
                    "mode": "symlink",
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                }
            )
        input_entries.append(
            {
                "name": "glycan_smiles",
                "source": "configuration",
                "workspace_path": "input/glycan.smiles",
                "mode": "materialized",
                "sha256": sha256_file(smiles_path),
                "size_bytes": smiles_path.stat().st_size,
            }
        )

        manifest = {
            "schema_version": 1,
            "pipeline_version": __version__,
            "created_at": utc_now(),
            "source_config": str(source_config),
            "workspace": str(workspace),
            "inputs": input_entries,
            "normalized_config": normalized,
        }
        manifest_path = workspace / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        sentinel = {
            "stage": "init",
            "status": "done",
            "completed_at": utc_now(),
            "pipeline_version": __version__,
            "manifest_sha256": sha256_file(manifest_path),
            "config_sha256": sha256_file(config_dst),
            "validation": {"inputs_exist": True, "workspace_created": True},
        }
        sentinel_path = workspace / ".lyso-md/init/.done"
        sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return workspace, backup
    except Exception:
        # Never leave a half-created fresh workspace behind. If --force moved an
        # old workspace, restore it if possible.
        if workspace.exists():
            shutil.rmtree(workspace)
        if backup is not None and backup.exists() and not workspace.exists():
            backup.rename(workspace)
        raise
