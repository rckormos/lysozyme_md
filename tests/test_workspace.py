import hashlib
import json
from pathlib import Path

import yaml

from lyso_md.config import load_config
from lyso_md.workspace import initialize_workspace


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    src = tmp_path / "source"
    src.mkdir()
    fasta = src / "seq.fasta"
    fasta.write_text(">design\nACDE\n", encoding="utf-8")
    bundle = src / "shared_glycam.zip"
    bundle.write_bytes(b"immutable-glycam")
    config = src / "design.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "design_042",
                "protein": {"fasta": "seq.fasta", "expected_residues": 4},
                "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "C.C"},
                "glycam": {"bundle": "shared_glycam.zip", "unit_name": "CONDENSEDSEQUENCE"},
                "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
                "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
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
                "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "32GB"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, fasta, bundle


def test_init_creates_expected_tree_and_symlinks(tmp_path: Path) -> None:
    config, fasta, bundle = _fixture(tmp_path)
    cfg = load_config(config)
    work_root = tmp_path / "work"
    workspace, backup = initialize_workspace(cfg, source_config=config, workspace_root=work_root)
    assert backup is None
    for name in ["01_chai", "02_prepare", "03_dry_relax", "04_solvate", "05_equilibrate", "06_production", "07_analysis", "logs"]:
        assert (workspace / name).is_dir()
    assert (workspace / "input/sequence.fasta").is_symlink()
    assert (workspace / "input/glycam_structure.zip").is_symlink()
    assert (workspace / "input/sequence.fasta").resolve() == fasta.resolve()
    assert (workspace / "input/glycam_structure.zip").resolve() == bundle.resolve()
    assert (workspace / "input/glycan.smiles").read_text().strip() == "C.C"

    normalized = yaml.safe_load((workspace / "config.yaml").read_text())
    assert normalized["protein"]["fasta"] == "input/sequence.fasta"
    assert normalized["glycam"]["bundle"] == "input/glycam_structure.zip"
    assert load_config(workspace / "config.yaml").name == "design_042"


def test_manifest_and_sentinel_record_provenance(tmp_path: Path) -> None:
    config, fasta, bundle = _fixture(tmp_path)
    cfg = load_config(config)
    workspace, _ = initialize_workspace(cfg, source_config=config, workspace_root=tmp_path / "work")
    manifest = json.loads((workspace / "manifest.json").read_text())
    by_name = {entry["name"]: entry for entry in manifest["inputs"]}
    assert by_name["protein_fasta"]["sha256"] == _sha(fasta)
    assert by_name["glycam_bundle"]["sha256"] == _sha(bundle)
    assert by_name["glycam_bundle"]["mode"] == "symlink"
    assert manifest["normalized_config"]["protein"]["fasta"] == "input/sequence.fasta"

    sentinel = json.loads((workspace / ".lyso-md/init/.done").read_text())
    assert sentinel["stage"] == "init"
    assert sentinel["status"] == "done"
    assert sentinel["validation"]["inputs_exist"] is True


def test_force_preserves_existing_workspace_as_backup(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    cfg = load_config(config)
    root = tmp_path / "work"
    workspace, _ = initialize_workspace(cfg, source_config=config, workspace_root=root)
    marker = workspace / "do-not-delete.txt"
    marker.write_text("prior results", encoding="utf-8")

    new_workspace, backup = initialize_workspace(cfg, source_config=config, workspace_root=root, force=True)
    assert new_workspace == workspace
    assert backup is not None
    assert (backup / "do-not-delete.txt").read_text() == "prior results"
    assert not (new_workspace / "do-not-delete.txt").exists()
