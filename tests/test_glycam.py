from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.glycam import inspect_glycam_bundle, list_off_units, parse_off_unit, safe_extract_zip


OFF_TEXT = '''!index array str
 "CONDENSEDSEQUENCE"
!entry.CONDENSEDSEQUENCE.unit.atoms table str name str type int typex int resx int flags int seq int elmnt dbl chg
 "C1" "Cg" 0 1 0 1 6 0.100000
 "H1" "H1" 0 1 0 2 1 0.050000
 "O1" "Oh" 0 2 0 3 8 -0.150000
!entry.CONDENSEDSEQUENCE.unit.residues table str name int seq int childseq int startatomx str restype int imagingx
 "ROH" 1 2 1 "?" 0
 "0Mr" 2 3 3 "?" 0
!entry.CONDENSEDSEQUENCE.unit.positions table dbl x dbl y dbl z
 0.0 0.0 0.0
 0.0 0.0 1.0
 1.0 0.0 0.0
!entry.CONDENSEDSEQUENCE.unit.connectivity table int atom1x int atom2x int flags
 1 2 1
 1 3 1
'''


def _write_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("glycam/structure.off", OFF_TEXT)
        zf.writestr("glycam/frcmod.glycam06_bacterial_K3O", "bacterial fixture\n")
        zf.writestr("glycam/frcmod.glycam06_intraring_doublebond_protonatedacids", "acid fixture\n")
        zf.writestr("__MACOSX/glycam/._structure.off", "metadata")
        zf.writestr(".DS_Store", "metadata")


def _write_config(tmp_path: Path, bundle: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    data = {
        "name": "test_design",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {
            "bundle": str(bundle),
            "unit_name": "CONDENSEDSEQUENCE",
            "bacterial_frcmod": "frcmod.glycam06_bacterial_K3O",
            "acid_frcmod": "frcmod.glycam06_intraring_doublebond_protonatedacids",
            "expected_heavy_atoms": 2,
            "expected_residues": 2,
        },
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {"hydrogen_relax_steps": 1000, "solvent_min_steps": 5000, "all_min_steps": 5000, "heat_ps": 100, "npt_5_ps": 250, "npt_1_ps": 250, "npt_free_ps": 500},
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 72},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "16GB", "cores": 1},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_parse_off_unit_exposes_required_metadata(tmp_path: Path) -> None:
    off = tmp_path / "structure.off"
    off.write_text(OFF_TEXT, encoding="utf-8")
    assert list_off_units(off) == ["CONDENSEDSEQUENCE"]
    unit = parse_off_unit(off, "CONDENSEDSEQUENCE")
    assert len(unit.atoms) == 3
    assert unit.heavy_atom_count == 2
    assert [r["name"] for r in unit.residues] == ["ROH", "0Mr"]
    assert unit.atoms[0]["name"] == "C1"
    assert unit.atoms[0]["type"] == "Cg"
    assert unit.atoms[0]["charge"] == pytest.approx(0.1)
    assert unit.atoms[2]["residue_index"] == 2
    assert unit.atoms[2]["x"] == pytest.approx(1.0)
    assert unit.connectivity == [{"atom1": 1, "atom2": 2, "flags": 1}, {"atom1": 1, "atom2": 3, "flags": 1}]


def test_parse_off_wrong_unit_fails_with_available_units(tmp_path: Path) -> None:
    off = tmp_path / "structure.off"
    off.write_text(OFF_TEXT, encoding="utf-8")
    with pytest.raises(ValueError, match="available units: CONDENSEDSEQUENCE"):
        parse_off_unit(off, "WRONG")


def test_safe_extract_ignores_macos_metadata(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_bundle(bundle)
    out = tmp_path / "out"
    extracted = safe_extract_zip(bundle, out)
    assert any(p.name == "structure.off" for p in extracted)
    assert not (out / "__MACOSX").exists()
    assert not (out / ".DS_Store").exists()


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="unsafe ZIP member path"):
        safe_extract_zip(bundle, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_inspect_bundle_writes_summary_and_done(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_bundle(bundle)
    cfg = load_config(_write_config(tmp_path, bundle))
    workspace = tmp_path / "workspace"
    (workspace / "02_prepare").mkdir(parents=True)

    result = inspect_glycam_bundle(cfg, workspace=workspace)
    assert result.structure_off.is_file()
    assert result.bacterial_frcmod.is_file()
    assert result.acid_frcmod.is_file()
    assert result.sentinel_path.is_file()
    summary = json.loads(result.summary_path.read_text())
    assert summary["counts"] == {"atoms": 3, "bonds": 2, "heavy_atoms": 2, "residues": 2}
    assert summary["residues"][0]["name"] == "ROH"
    assert summary["validation"]["passed"] is True


def test_inspect_bundle_fails_closed_on_expected_count_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_bundle(bundle)
    cfg_path = _write_config(tmp_path, bundle)
    data = yaml.safe_load(cfg_path.read_text())
    data["glycam"]["expected_heavy_atoms"] = 99
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    cfg = load_config(cfg_path)
    workspace = tmp_path / "workspace"
    (workspace / "02_prepare").mkdir(parents=True)

    with pytest.raises(ValueError, match="heavy-atom count"):
        inspect_glycam_bundle(cfg, workspace=workspace)
    assert not (workspace / "02_prepare" / "glycam" / ".done").exists()
