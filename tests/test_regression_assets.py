from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.glycam import inspect_glycam_bundle, parse_off_unit
from lyso_md.mapping import map_chai_to_glycam


REQUIRED_ASSETS = {
    "input/sequence.fasta",
    "input/chai_raw.cif",
    "input/chai_prody.pdb",
    "input/glycam_structure.zip",
    "reference/complex_glycam_reference.pdb",
    "reference/atom_mapping.tsv",
    "reference/complex_dry.parm7",
    "reference/complex_dry.rst7",
    "reference/min_all.out",
}


def _regression_root() -> Path | None:
    value = os.environ.get("LYSO_MD_REGRESSION_DATA")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        pytest.fail(f"LYSO_MD_REGRESSION_DATA does not point to a directory: {root}")
    return root


def _config(root: Path, path: Path) -> Path:
    data = {
        "name": "human_lysozyme_pg_regression",
        "protein": {"fasta": str(root / "input/sequence.fasta"), "expected_residues": 130},
        "chai": {
            "enabled": True,
            "model_index": 0,
            "ligand_resname": "LIG",
            "glycan_smiles": (
                "CC(=O)N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O2)[C@@H]1O."
                "C[C@@H](O[C@H]3[C@H](O4)[C@@H](CO)O[C@@H]2[C@@H]3NC(C)=O)C(O)=O."
                "CC(=O)N[C@H]5[C@H]4O[C@H](CO)[C@@H](O6)[C@@H]5O."
                "C[C@@H](O[C@H]7[C@H](O)[C@@H](CO)O[C@@H]6[C@@H]7NC(C)=O)C(O)=O"
            ),
        },
        "glycam": {
            "bundle": str(root / "input/glycam_structure.zip"),
            "unit_name": "CONDENSEDSEQUENCE",
            "bacterial_frcmod": "frcmod.glycam06_bacterial_K3O",
            "acid_frcmod": "frcmod.glycam06_intraring_doublebond_protonatedacids",
            "expected_heavy_atoms": 67,
            "expected_residues": 5,
        },
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
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 80},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "32GB", "cores": 1},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_real_regression_bundle_is_complete() -> None:
    root = _regression_root()
    if root is None:
        pytest.skip("set LYSO_MD_REGRESSION_DATA to enable the high-fidelity regression fixture")
    missing = sorted(str(path) for path in REQUIRED_ASSETS if not (root / path).is_file())
    assert not missing, f"regression bundle is missing required assets: {missing}"


def test_real_regression_off_counts_and_residues(tmp_path: Path) -> None:
    root = _regression_root()
    if root is None:
        pytest.skip("set LYSO_MD_REGRESSION_DATA to enable the high-fidelity regression fixture")
    config = load_config(_config(root, tmp_path / "config.yaml"))
    workspace = tmp_path / "workspace"
    result = inspect_glycam_bundle(config, workspace=workspace)
    assert result.summary_path.is_file()
    unit = parse_off_unit(workspace / "02_prepare/glycam/extracted/structure/structure.off", "CONDENSEDSEQUENCE")
    assert unit.heavy_atom_count == 67
    assert len(unit.atoms) == 127
    assert len(unit.residues) == 5
    assert [r["name"] for r in unit.residues] == ["ROH", "4YB", "4Mr", "4YB", "0Mr"]


def test_real_regression_mapping_matches_historical_reference(tmp_path: Path) -> None:
    root = _regression_root()
    if root is None:
        pytest.skip("set LYSO_MD_REGRESSION_DATA to enable the high-fidelity regression fixture")
    config = load_config(_config(root, tmp_path / "config.yaml"))
    workspace = tmp_path / "workspace"
    inspect_glycam_bundle(config, workspace=workspace)
    chai_stage = workspace / "01_chai"
    chai_stage.mkdir(parents=True)
    (chai_stage / "pred.model_idx_0.pdb").symlink_to((root / "input/chai_prody.pdb").resolve())
    (chai_stage / ".done").write_text("{}\n", encoding="utf-8")
    result = map_chai_to_glycam(config, workspace=workspace)
    assert result.mapping_path.is_file()

    with result.mapping_path.open(newline="", encoding="utf-8") as fh:
        generated = list(csv.DictReader(fh, delimiter="\t"))
    with (root / "reference/atom_mapping.tsv").open(newline="", encoding="utf-8") as fh:
        reference = list(csv.DictReader(fh, delimiter="\t"))

    key = lambda row: (row["chai_serial"], row["glycam_resid"], row["glycam_atom"])
    assert sorted(map(key, generated)) == sorted(map(key, reference))
    assert len(generated) == 67
