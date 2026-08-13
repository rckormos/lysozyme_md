from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.leap import assemble_dry_complex


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "leap_test",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {"bundle": str(bundle), "unit_name": "CONDENSEDSEQUENCE", "expected_heavy_atoms": 3, "expected_residues": 1},
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


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "02_prepare/glycam/extracted/structure").mkdir(parents=True)
    (ws / "02_prepare/protein").mkdir(parents=True)
    (ws / "02_prepare/coordinate_transfer").mkdir(parents=True)
    (ws / "03_dry_relax").mkdir(parents=True)
    (ws / "02_prepare/glycam/.done").write_text("{}\n")
    (ws / "02_prepare/protein/.done").write_text("{}\n")
    (ws / "02_prepare/coordinate_transfer/.done").write_text("{}\n")
    (ws / "02_prepare/glycam/extracted/structure/structure.off").write_text("OFF\n")
    (ws / "02_prepare/glycam/extracted/frcmod.glycam06_bacterial_K3O").write_text("params\n")
    (ws / "02_prepare/glycam/extracted/frcmod.glycam06_intraring_doublebond_protonatedacids").write_text("params\n")
    (ws / "02_prepare/protein/protein_chai.pdb").write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 70.00           C\nEND\n")
    (ws / "02_prepare/protein/disulfide_bonds.leap").write_text("bond protein.1.SG protein.2.SG\n")
    (ws / "02_prepare/coordinate_transfer/glycan_aligned.off").write_text("OFF\n")
    return ws


def _fake_tleap(path: Path, *, charge: str = "0.0000", atoms: int = 2, fatal: str = "") -> Path:
    script = path / "tleap"
    script.write_text(
        "#!/bin/sh\n"
        "echo '" + (fatal or "") + "'\n"
        "echo 'Total unperturbed charge: " + charge + "'\n"
        f"cat > complex_dry.pdb <<'EOF'\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 70.00           C\n"
        + ("ATOM      2  O   ALA A   1       1.000   0.000   0.000  1.00 70.00           O\n" if atoms >= 2 else "")
        + "END\nEOF\n"
        f"cat > complex_dry.parm7 <<'EOF'\n%FLAG POINTERS\n%FORMAT(10I8)\n{atoms:8d}\nEOF\n"
        f"cat > complex_dry.rst7 <<'EOF'\nLEaP test\n{atoms}\n  0.000000  0.000000  0.000000\nEOF\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_dry_run_renders_leap_input_without_running(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = assemble_dry_complex(cfg, workspace=ws, dry_run=True)
    assert result.dry_run is True
    text = result.input_path.read_text()
    assert "source leaprc.protein.ff19SB" in text
    assert "source leaprc.GLYCAM_06j-1" in text
    assert "loadamberparams frcmod.glycam06_bacterial_K3O" in text
    assert "loadamberparams frcmod.glycam06_intraring_doublebond_protonatedacids" in text
    assert "bond protein.1.SG protein.2.SG" in text
    assert "saveamberparm complex complex_dry.parm7 complex_dry.rst7" in text
    assert not result.sentinel_path.exists()


def test_assemble_dry_complex_validates_outputs_and_writes_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_tleap(tmp_path, atoms=2)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    result = assemble_dry_complex(cfg, workspace=ws)
    assert result.sentinel_path.is_file()
    validation = json.loads(result.validation_path.read_text())
    assert validation["charge"]["integral"] is True
    assert validation["counts"] == {"pdb_atoms": 2, "parm7_atoms": 2, "rst7_atoms": 2}
    assert validation["checks"]["passed"] is True


def test_assemble_fails_closed_on_fatal_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_tleap(tmp_path, fatal="FATAL: Atom has no type")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(ValueError, match="fatal/parameter errors"):
        assemble_dry_complex(cfg, workspace=ws)
    assert not (ws / "03_dry_relax/.done").exists()


def test_assemble_fails_on_nonintegral_charge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_tleap(tmp_path, charge="0.2500")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(ValueError, match="non-integral"):
        assemble_dry_complex(cfg, workspace=ws)
    assert not (ws / "03_dry_relax/.done").exists()


def test_assemble_requires_all_prior_checkpoints(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    (ws / "02_prepare/protein/.done").unlink()
    with pytest.raises(ValueError, match="Phase 6"):
        assemble_dry_complex(cfg, workspace=ws, dry_run=True)
