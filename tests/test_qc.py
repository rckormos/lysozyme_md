from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.qc import build_qc, write_qc_report


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "qc_test",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {"bundle": str(bundle), "unit_name": "CONDENSEDSEQUENCE", "expected_heavy_atoms": 3, "expected_residues": 1},
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {"hydrogen_relax_steps": 1000, "solvent_min_steps": 5000, "all_min_steps": 5000, "heat_ps": 100, "npt_5_ps": 250, "npt_1_ps": 250, "npt_free_ps": 500},
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 80},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "32GB", "cores": 1},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_qc_separates_failures_warnings_and_info(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = tmp_path / "workspace"
    (ws / "07_analysis").mkdir(parents=True)
    (ws / "07_analysis" / ".done").write_text("{}\n", encoding="utf-8")
    (ws / "07_analysis" / "validation.json").write_text(json.dumps({"stage": "analysis", "status": "done", "checks": {"passed": True}}), encoding="utf-8")
    (ws / "07_analysis" / "rmsd_protein_ca.dat").write_text("#Frame RMSD\n1 0.0\n2 2.0\n", encoding="utf-8")
    qc = build_qc(cfg, workspace=ws)
    assert qc["status"] == "done"
    assert qc["hard_failures"] == []
    assert qc["summary"]["warning_count"] > 0
    assert qc["information"]["analysis"]["rmsd_protein_ca"]["numeric_values"] == 4


def test_qc_detects_unreached_production_target(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = tmp_path / "workspace"
    prod = ws / "07_production"
    prod.mkdir(parents=True)
    (prod / "production_validation.json").write_text(json.dumps({"results": {"completed_ns": 999.0, "target_ns": 1000}, "checks": {"passed": True}}), encoding="utf-8")
    qc = build_qc(cfg, workspace=ws)
    assert any(item["type"] == "production_target_not_reached" for item in qc["hard_failures"])


def test_write_qc_report_creates_both_outputs(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = tmp_path / "workspace"
    (ws / "07_analysis").mkdir(parents=True)
    qc = write_qc_report(cfg, workspace=ws)
    assert (ws / "07_analysis/qc_summary.json").is_file()
    assert (ws / "07_analysis/qc_report.md").is_file()
    assert qc["outputs"]["report"].endswith("qc_report.md")
