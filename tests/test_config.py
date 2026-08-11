from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lyso_md.config import load_config


def _write_config(tmp_path: Path, **overrides) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nACDE\n", encoding="utf-8")
    bundle = tmp_path / "glycam.zip"
    bundle.write_bytes(b"zip-placeholder")
    data = {
        "name": "design_001",
        "protein": {"fasta": "sequence.fasta", "expected_residues": 4},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {"bundle": "glycam.zip", "unit_name": "CONDENSEDSEQUENCE"},
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12.0, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {
            "hydrogen_relax_steps": 1000,
            "solvent_min_steps": 5000,
            "all_min_steps": 5000,
            "heat_ps": 100,
            "npt_5_ps": 250,
            "npt_1_ps": 250,
            "npt_free_ps": 500,
        },
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 72},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "16GB"},
    }
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        data[section][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_valid_config_resolves_input_paths(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    cfg = load_config(path)
    assert cfg.protein.fasta == (tmp_path / "sequence.fasta").resolve()
    assert cfg.glycam.bundle == (tmp_path / "glycam.zip").resolve()


@pytest.mark.parametrize(
    "overrides",
    [
        {"protein__expected_residues": 0},
        {"solvent__concentration_molar": -0.1},
        {"production__target_ns": 0},
        {"production__chunk_ns": 0},
    ],
)
def test_invalid_numeric_bounds(tmp_path: Path, overrides: dict) -> None:
    path = _write_config(tmp_path, **overrides)
    with pytest.raises(ValidationError):
        load_config(path)


def test_chunk_cannot_exceed_target(tmp_path: Path) -> None:
    path = _write_config(tmp_path, production__target_ns=10, production__chunk_ns=20)
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_fasta_fails_closed(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    (tmp_path / "sequence.fasta").unlink()
    with pytest.raises(ValueError, match="protein.fasta"):
        load_config(path)


def test_missing_bundle_fails_closed(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    (tmp_path / "glycam.zip").unlink()
    with pytest.raises(ValueError, match="glycam.bundle"):
        load_config(path)
