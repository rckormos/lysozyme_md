from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.protein import DISULFIDE_CUTOFF_ANGSTROM, prepare_protein


def _config(tmp_path: Path, residues: int = 6) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\n" + "A" * residues + "\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "phase6_test",
        "protein": {"fasta": str(fasta), "expected_residues": residues},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCCC"},
        "glycam": {"bundle": str(bundle), "unit_name": "CONDENSEDSEQUENCE", "expected_heavy_atoms": 4, "expected_residues": 1},
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


def _atom(serial: int, name: str, resname: str, resid: int, x: float, y: float, z: float, element: str) -> str:
    return f"ATOM  {serial:5d} {name:<4s} {resname:>3s} A{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 70.00          {element:>2s}"


def _workspace(tmp_path: Path, *, ambiguous: bool = False, missing_sg: bool = False) -> Path:
    ws = tmp_path / "workspace"
    chai = ws / "01_chai"
    chai.mkdir(parents=True)
    lines = []
    serial = 1
    cysteine_sg = {1: (0.0, 0.0, 0.0), 2: (2.03, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 4: (12.03, 0.0, 0.0)}
    for resid in range(1, 7):
        resname = "CYS" if resid <= 4 else "ALA"
        lines.append(_atom(serial, "CA", resname, resid, float(resid), 2.0, 0.0, "C")); serial += 1
        if resname == "CYS" and not (missing_sg and resid == 1):
            x, y, z = cysteine_sg[resid]
            if ambiguous and resid == 2:
                x, y = 2.03, 0.0
            lines.append(_atom(serial, "SG", resname, resid, x, y, z, "S")); serial += 1
        if resid == 5:
            lines.append(_atom(serial, "HA", resname, resid, float(resid), 2.2, 0.0, "H")); serial += 1
    # Put residue 1 and 2 at a disulfide distance, and 3 and 4 likewise.
    if ambiguous:
        # Make residue 2 close to both 1 and 3.
        pass
    lines.append("HETATM" + " " * 70 + " C")
    (chai / "pred.model_idx_0.pdb").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (chai / ".done").write_text("{}\n", encoding="utf-8")
    return ws


def test_prepare_protein_removes_hydrogens_detects_disulfides_and_writes_leap(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = prepare_protein(cfg, workspace=ws)
    assert result.sentinel_path.is_file()
    text = result.protein_pdb.read_text()
    assert " H " not in text
    assert "HETATM" not in text
    assert text.endswith("TER\nEND\n")
    assert "CYX" in text
    bonds = result.leap_bonds_path.read_text().splitlines()
    assert bonds == ["bond protein.1.SG protein.2.SG", "bond protein.3.SG protein.4.SG"]
    rows = result.disulfides_tsv.read_text().splitlines()
    assert len(rows) == 3
    validation = json.loads(result.validation_path.read_text())
    assert validation["counts"]["protein_residues"] == 6
    assert validation["counts"]["hydrogen_records_removed"] == 1
    assert validation["counts"]["disulfide_pairs"] == 2
    assert validation["checks"]["passed"] is True


def test_prepare_protein_requires_sg_for_every_cysteine(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path, missing_sg=True)
    with pytest.raises(ValueError, match="missing SG"):
        prepare_protein(cfg, workspace=ws)
    assert not (ws / "02_prepare/protein/.done").exists()


def test_prepare_protein_fails_on_ambiguous_disulfide_candidates(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    pdb = ws / "01_chai/pred.model_idx_0.pdb"
    lines = pdb.read_text().splitlines()
    # Move residue 3 SG to the same point as residue 1, making residue 1 a candidate with both 2 and 3.
    for i, line in enumerate(lines):
        if line.startswith("ATOM") and line[12:16].strip() == "SG" and line[22:26].strip() == "3":
            lines[i] = line[:30] + f"{0.000:8.3f}{0.000:8.3f}{0.000:8.3f}" + line[54:]
    pdb.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="ambiguous disulfide"):
        prepare_protein(cfg, workspace=ws)
    assert not (ws / "02_prepare/protein/.done").exists()


def test_disulfide_cutoff_is_conservative() -> None:
    assert DISULFIDE_CUTOFF_ANGSTROM == pytest.approx(2.4)
