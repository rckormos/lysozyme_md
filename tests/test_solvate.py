from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.solvate import _box_volume, calculate_kcl_counts, solvate_and_ionize


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "solvate_test",
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


def _pdb(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(f"ATOM  {i+1:5d}  CA  ALA A   1       {float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 70.00           C\n")
        f.write("END\n")


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    for d in ("02_prepare/glycam/extracted/structure", "02_prepare/protein", "02_prepare/coordinate_transfer", "03_dry_relax/hydrogen_relax", "04_solvate"):
        (ws / d).mkdir(parents=True)
    for marker in (ws / "02_prepare/glycam/.done", ws / "02_prepare/protein/.done", ws / "02_prepare/coordinate_transfer/.done", ws / "03_dry_relax/.done", ws / "03_dry_relax/hydrogen_relax/.done"):
        marker.write_text("{}\n")
    (ws / "02_prepare/coordinate_transfer/glycan_aligned.off").write_text("OFF\n")
    _pdb(ws / "02_prepare/protein/protein_chai.pdb", 1)
    (ws / "02_prepare/protein/disulfide_bonds.leap").write_text("\n")
    (ws / "02_prepare/glycam/extracted/frcmod.glycam06_bacterial_K3O").write_text("params\n")
    (ws / "02_prepare/glycam/extracted/frcmod.glycam06_intraring_doublebond_protonatedacids").write_text("params\n")
    _pdb(ws / "03_dry_relax/complex_dry.pdb", 2)
    (ws / "03_dry_relax/complex_dry.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n")
    (ws / "03_dry_relax/validation.json").write_text(json.dumps({"charge": {"total_unperturbed": 6.0}}))
    (ws / "03_dry_relax/hydrogen_relax/complex_hrelaxed.rst7").write_text("relaxed\n2\n    1.0000000    2.0000000    3.0000000    4.0000000    5.0000000    6.0000000\n")
    return ws


def test_box_volume() -> None:
    assert _box_volume((20, 20, 20, 109.47122063449069, 109.47122063449069, 109.47122063449069)) == pytest.approx(6158.402871, rel=1e-6)


def test_kcl_count_formula_and_neutralization() -> None:
    assert calculate_kcl_counts(0.05, 10000.0, 6.0) == (0, 0, 6)
    assert calculate_kcl_counts(0.05, 10000.0, -6.0) == (0, 6, 0)
    assert calculate_kcl_counts(0.05, 10000.0, 0.0) == (0, 0, 0)


def test_dry_run_writes_probe_only_and_never_guesses_ion_counts(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = solvate_and_ionize(cfg, workspace=ws, dry_run=True)
    assert result.dry_run
    probe = (result.stage / "solvate_probe.in").read_text()
    assert "solvateOct complex OPCBOX 12.0" in probe
    assert not (result.stage / "solvate_ionize.in").exists()
    assert not result.sentinel_path.exists()


def test_full_run_uses_two_source_based_leap_sessions_and_no_loadamberparm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    fake = tmp_path / "tleap"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = \"solvate_probe.in\" ]; then\n"
        "  echo 'Total unperturbed charge: 6.0000'\n"
        "  printf '%s\\n' '%FLAG POINTERS' '%FORMAT(10I8)' '       3' > solvated_probe.parm7\n"
        "  printf '%s\\n' 'probe' '3' '    1.0000000    2.0000000    3.0000000    4.0000000    5.0000000    6.0000000' '    7.0000000    8.0000000    9.0000000' '   20.0000000   20.0000000   20.0000000  109.4712206  109.4712206  109.4712206' > solvated_probe.rst7\n"
        "  printf '%s\\n' 'ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 70.00           C' 'ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00 70.00           C' 'ATOM      3  O   HOH W   1       7.000   8.000   9.000  1.00 70.00           O' 'END' > solvated_probe.pdb\n"
        "else\n"
        "  echo 'Total unperturbed charge: 0.0000'\n"
        "  printf '%s\\n' '%FLAG POINTERS' '%FORMAT(10I8)' '       8' > complex_solvated.parm7\n"
        "  printf '%s\\n' 'final' '8' '    1.0000000    2.0000000    3.0000000    4.0000000    5.0000000    6.0000000' '    7.0000000    8.0000000    9.0000000   10.0000000   11.0000000   12.0000000' '   13.0000000   14.0000000   15.0000000   16.0000000   17.0000000   18.0000000' '   19.0000000   20.0000000   21.0000000   22.0000000   23.0000000   24.0000000' '   20.0000000   20.0000000   20.0000000  109.4712206  109.4712206  109.4712206' > complex_solvated.rst7\n"
        "  { printf 'ATOM  %5d  CA  ALA A   1       0.000   0.000   0.000  1.00 70.00           C\\n' 1; printf 'ATOM  %5d  CA  ALA A   1       0.000   0.000   0.000  1.00 70.00           C\\n' 2; for i in 3 4 5 6 7 8; do printf 'HETATM%5d  Cl  Cl- X   1       0.000   0.000   0.000  1.00 70.00          Cl\\n' $i; done; printf 'END\\n'; } > complex_solvated.pdb\n"
        "fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + __import__("os").environ.get("PATH", ""))
    result = solvate_and_ionize(cfg, workspace=ws)
    text = result.input_path.read_text()
    assert "addIonsRand complex K+ 0" in text
    assert "addIonsRand complex Cl- 6" in text
    assert "loadAmberParm" not in text
    assert "saveAmberParm complex complex_solvated.parm7 complex_solvated.rst7" in text
    assert "savePdb complex complex_solvated.pdb" in text
    validation = json.loads(result.validation_path.read_text())
    assert validation["salt"]["pairs"] == 0
    assert validation["salt"]["potassium"] == 0
    assert validation["salt"]["chloride"] == 6
    assert validation["checks"]["no_loadAmberParm"] is True
