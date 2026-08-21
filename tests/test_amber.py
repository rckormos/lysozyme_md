from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from lyso_md.amber import relax_hydrogens
from lyso_md.config import load_config


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "amber_test",
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
    stage = ws / "03_dry_relax"
    stage.mkdir(parents=True)
    (stage / ".done").write_text("{}\n")
    (stage / "complex_dry.parm7").write_text("%FLAG POINTERS\n%FORMAT(10I8)\n       2\n", encoding="utf-8")
    (stage / "complex_dry.rst7").write_text("dry\n2\n  0.000000  0.000000  0.000000  1.000000  1.000000  1.000000\n", encoding="utf-8")
    return ws


def _fake_pmemd(path: Path, *, nonzero: bool = False, complete: bool = True) -> Path:
    script = path / "pmemd"
    marker = "FINAL RESULTS\n NSTEP ENERGY RMS GMAX NAME NUMBER\n 1000 -12.5000 0.0005 0.4567 TEST 1\n5.  TIMINGS\n" if complete else "NSTEP 500\n"
    script.write_text(
        "#!/bin/sh\n"
        "cat > hrelax.out <<'EOF'\n" + marker + "EOF\n"
        "cat > complex_hrelaxed.rst7 <<'EOF'\n"
        "relaxed\n2\n  0.100000  0.200000  0.300000  1.100000  1.200000  1.300000\n"
        "EOF\n"
        + ("echo 'Floating point exception' 1>&2\n" if nonzero else "")
        + ("exit 1\n" if nonzero else "exit 0\n"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_dry_run_generates_cpu_pmemd_input(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = relax_hydrogens(cfg, workspace=ws, dry_run=True)
    text = result.input_path.read_text()
    assert "imin=1" in text
    assert "maxcyc=1000" in text
    assert "ncyc=500" in text
    assert "ntb=0" in text
    assert "igb=0" in text
    assert "cut=1000.0" in text
    assert "restraint_wt=100.0" in text
    assert "restraintmask='!@H='" in text
    assert not result.sentinel_path.exists()


def test_relax_validates_restart_and_writes_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_pmemd(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr("lyso_md.amber._xh_bond_geometry", lambda *args, **kwargs: {"bond_count": 0, "max_abs_change_from_initial": 0.0, "max_abs_change_from_equilibrium": 0.0, "warnings": [], "failures": [], "passed": True})
    result = relax_hydrogens(cfg, workspace=ws)
    assert result.sentinel_path.is_file()
    validation = json.loads(result.validation_path.read_text())
    assert validation["results"]["step"] == 1000
    assert validation["checks"]["finite_energy_gradient"] is True
    assert validation["checks"]["matching_atom_counts"] is True
    assert result.output_path.is_file()


def test_nonzero_pmemd_is_warning_if_normal_completion_is_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_pmemd(tmp_path, nonzero=True)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr("lyso_md.amber._xh_bond_geometry", lambda *args, **kwargs: {"bond_count": 0, "max_abs_change_from_initial": 0.0, "max_abs_change_from_equilibrium": 0.0, "warnings": [], "failures": [], "passed": True})
    result = relax_hydrogens(cfg, workspace=ws)
    validation = json.loads(result.validation_path.read_text())
    assert validation["process"]["returncode"] == 1
    assert validation["warnings"]
    assert result.sentinel_path.is_file()


def test_incomplete_pmemd_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    _fake_pmemd(tmp_path, complete=False)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises((RuntimeError, ValueError), match="FINAL RESULTS|normal-completion|normal completion"):
        relax_hydrogens(cfg, workspace=ws)
    assert not (ws / "03_dry_relax/hydrogen_relax/.done").exists()


def test_missing_phase7_checkpoint_fails(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    (ws / "03_dry_relax/.done").unlink()
    with pytest.raises(ValueError, match="Phase 7"):
        relax_hydrogens(cfg, workspace=ws, dry_run=True)


def test_netcdf_restart_parser_reads_amber22_format(tmp_path: Path) -> None:
    from scipy.io import netcdf_file
    from lyso_md.amber import _parse_restart_atom_count, _parse_restart_coordinates

    path = tmp_path / "complex_hrelaxed.rst7"
    with netcdf_file(str(path), mode="w") as nc:
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", 2)
        nc.createVariable("coordinates", "d", ("atom", "spatial"))[:] = [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
        ]
    assert _parse_restart_atom_count(path) == 2
    assert _parse_restart_coordinates(path, 2) == [0.1, 0.2, 0.3, 1.1, 1.2, 1.3]


def test_netcdf_restart_parser_rejects_nonfinite_coordinates(tmp_path: Path) -> None:
    from scipy.io import netcdf_file
    from lyso_md.amber import _parse_restart_coordinates

    path = tmp_path / "complex_hrelaxed.rst7"
    with netcdf_file(str(path), mode="w") as nc:
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", 1)
        nc.createVariable("coordinates", "d", ("atom", "spatial"))[:] = [[float("nan"), 0.0, 0.0]]
    with pytest.raises(ValueError, match="non-finite"):
        _parse_restart_coordinates(path, 1)


def test_hydrogen_relaxation_uses_full_xh_force_terms(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    result = relax_hydrogens(cfg, workspace=ws, dry_run=True)
    text = result.input_path.read_text()
    assert "ntc=1" in text
    assert "ntf=1" in text
    assert "ntmin=1" in text
    assert "drms=1.0e-03" in text
    assert "ntxo=2" in text
    assert "ntc=2" not in text
    assert "ntf=2" not in text


def test_final_results_parser_uses_final_section() -> None:
    from lyso_md.amber import _parse_final_results

    text = """
   100 -100.0 12.0 50.0 O 1

 FINAL RESULTS

   1000 -200.0 0.0005 0.02 H 2

 5.  TIMINGS
"""
    result = _parse_final_results(text)
    assert result["step"] == 1000
    assert result["rms"] == pytest.approx(0.0005)
    assert result["gmax"] == pytest.approx(0.02)


def test_final_results_parser_requires_convergence_in_validation(tmp_path: Path) -> None:
    from lyso_md.amber import _parse_final_results, _DRMS_DEFAULT

    result = _parse_final_results(
        "FINAL RESULTS\n  1000 -10.0 0.01 0.5 H 1\n 5.  TIMINGS\n"
    )
    assert result["rms"] > _DRMS_DEFAULT


def test_xh_bond_geometry_flags_detached_hydrogen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types
    from lyso_md.amber import _xh_bond_geometry

    class Atom:
        def __init__(self, idx: int, name: str, element: int):
            self.idx = idx
            self.name = name
            self.element = element
            self.residue = types.SimpleNamespace(number=1)

    class BondType:
        req = 1.01

    a = Atom(0, "NZ", 7)
    h = Atom(1, "HZ3", 1)
    bond = types.SimpleNamespace(atom1=a, atom2=h, type=BondType())
    structure = types.SimpleNamespace(atoms=[a, h], bonds=[bond])
    fake_parmed = types.SimpleNamespace(load_file=lambda _: structure)
    monkeypatch.setitem(sys.modules, "parmed", fake_parmed)

    parm7 = tmp_path / "complex.parm7"
    parm7.write_text("placeholder\n", encoding="utf-8")
    initial = [0.0, 0.0, 0.0, 1.01, 0.0, 0.0]
    detached = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    result = _xh_bond_geometry(parm7, initial, detached)
    assert result["bond_count"] == 1
    assert result["passed"] is False
    assert result["failures"][0]["atom2"] == "1:HZ3"
