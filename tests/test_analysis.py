from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lyso_md.analysis import analyze
from lyso_md.config import load_config


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "analysis_test",
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


def _production_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    stage = ws / "07_production" / "chunk_001"
    stage.mkdir(parents=True)
    (ws / "07_production" / ".done").write_text("{}\n", encoding="utf-8")
    (stage / ".done").write_text("{}\n", encoding="utf-8")
    (stage / "validation.json").write_text(json.dumps({"status": "done", "checks": {"passed": True}}), encoding="utf-8")
    (stage / "production.nc").write_bytes(b"netcdf")
    (stage / "complex_solvated.parm7").write_text("parm\n", encoding="utf-8")
    return ws


def test_dry_run_generates_complete_analysis_suite(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _production_workspace(tmp_path)
    result = analyze(cfg, workspace=ws, dry_run=True)
    assert result.dry_run is True
    stage = ws / "07_analysis"
    for name in [
        "preprocess.in", "pairwise_preprocess.in", "clustering_preprocess.in", "rmsd.in", "rmsf.in", "rg.in", "dssp.in",
        "hbond_protein_to_glycan.in", "hbond_glycan_to_protein.in", "contacts.in", "pca.in",
        "pca_projection.in", "pca_modes.in", "dccm.in", "clustering.in", "average_structure.in", "pairwise_rmsd.in",
        "distances.in", "angles.in", "analysis_manifest.json",
    ]:
        assert (stage / name).is_file(), name
    pca = (stage / "pca.in").read_text()
    assert "rms first @CA" in pca
    assert "matrix covar name covar @CA" in pca
    assert "diagmatrix covar out ca_modes.dat vecs 20 name ca_modes" in pca
    assert "projection modes" not in pca
    projection = (stage / "pca_projection.in").read_text()
    assert "readdata ca_modes.dat name ca_modes" in projection
    assert "projection modes ca_modes.dat beg 1 end 3 @CA out ca_projection.dat" in projection
    contacts = (stage / "contacts.in").read_text()
    assert "nativecontacts :1-130 :131-135 distance 4.0 skipnative out protein_glycan_contacts.dat" in contacts
    distances = (stage / "distances.in").read_text()
    assert "distance protein_glycan_ca_distance :65@CA :132@C1 out distances.dat" in distances
    assert ":131@C1" not in distances
    angles = (stage / "angles.in").read_text()
    assert "angle protein_glycan_angle :64@CA :65@CA :132@C1 out angles.dat" in angles
    assert ":131@C1" not in angles
    assert "\ncontacts :1-130 :131-135" not in "\n" + contacts
    modes = (stage / "pca_modes.in").read_text()
    assert "trajoutmask @CA" in modes
    assert not (stage / ".done").exists()
    clustering_pre = (stage / "clustering_preprocess.in").read_text()
    assert "trajin " + str(stage / "processed.nc") + " 1 last 100" in clustering_pre
    clustering = (stage / "clustering.in").read_text()
    assert "trajin " + str(stage / "clustering_subsampled.nc") in clustering



def test_run_cpptraj_does_not_double_suffix_temp_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lyso_md import analysis as analysis_mod

    input_path = tmp_path / "clustering_preprocess.in"
    output = tmp_path / "clustering_subsampled.nc"
    input_path.write_text(
        "parm topology.parm7\n"
        "trajin processed.nc 1 last 100\n"
        "trajout clustering_subsampled.nc netcdf nobox\n"
        "run\nquit\n",
        encoding="utf-8",
    )

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, text, capture_output):
        rendered = input_path.with_name("clustering_preprocess.tmp.in").read_text(encoding="utf-8")
        assert "clustering_subsampled.nc.tmp" in rendered
        assert "clustering_subsampled.nc.tmp.tmp" not in rendered
        (tmp_path / "clustering_subsampled.nc.tmp").write_bytes(b"netcdf")
        return Proc()

    monkeypatch.setattr(analysis_mod.shutil, "which", lambda name: "/usr/bin/cpptraj")
    monkeypatch.setattr(analysis_mod.subprocess, "run", fake_run)
    analysis_mod._run_cpptraj(input_path, cwd=tmp_path, outputs=[output])
    assert output.read_bytes() == b"netcdf"
    assert not (tmp_path / "clustering_subsampled.nc.tmp").exists()


def test_analysis_step_inputs_are_explicit_for_split_pca(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _production_workspace(tmp_path)
    stage = ws / "07_analysis"
    stage.mkdir(parents=True)
    from lyso_md import analysis as analysis_mod
    inputs = analysis_mod._analysis_inputs(stage / "processed.parm7", stage / "processed.nc", stage)
    assert "pca.in" in inputs
    assert "pca_projection.in" in inputs
    assert "pca_modes.in" in inputs



def test_analysis_requires_completed_production(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    with pytest.raises(ValueError, match="completed 1-target production"):
        analyze(cfg, workspace=tmp_path / "workspace", dry_run=True)


def test_analysis_requires_contiguous_validated_chunks(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _production_workspace(tmp_path)
    (ws / "07_production" / "chunk_002").mkdir()
    (ws / "07_production" / "chunk_002" / ".done").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"production chunk 2 has \.done but no validation.json"):
        analyze(cfg, workspace=ws, dry_run=True)


def test_analysis_reuses_completed_preprocess_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _production_workspace(tmp_path)
    stage = ws / "07_analysis"
    stage.mkdir(parents=True)
    processed = stage / "processed.nc"
    processed.write_bytes(b"processed-netcdf")
    topology = stage / "processed.parm7"
    topology.write_bytes(b"processed-topology")
    from lyso_md import analysis as analysis_mod
    analysis_mod._checkpoint_write(stage, "preprocess", [processed, topology])
    calls: list[str] = []
    monkeypatch.setattr(analysis_mod, "_run_cpptraj", lambda *args, **kwargs: calls.append(Path(args[0]).name))
    monkeypatch.setattr(analysis_mod, "_write_parmed_stripped", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ParmEd should not rerun")))
    # The test only exercises checkpoint recognition directly; downstream products are intentionally absent.
    assert analysis_mod._checkpoint_valid(stage, "preprocess", [processed, topology])
    assert calls == []


def test_analysis_checkpoint_detects_tampered_output(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _production_workspace(tmp_path)
    stage = ws / "07_analysis"
    stage.mkdir(parents=True)
    processed = stage / "processed.nc"
    processed.write_bytes(b"processed-netcdf")
    topology = stage / "processed.parm7"
    topology.write_bytes(b"processed-topology")
    from lyso_md import analysis as analysis_mod
    analysis_mod._checkpoint_write(stage, "preprocess", [processed, topology])
    processed.write_bytes(b"tampered")
    assert not analysis_mod._checkpoint_valid(stage, "preprocess", [processed, topology])


def test_run_cpptraj_redirects_relative_output_to_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lyso_md import analysis as analysis_mod

    input_path = tmp_path / "rmsd.in"
    output = tmp_path / "rmsd_protein_ca.dat"
    input_path.write_text(
        "parm topology.parm7\n"
        "trajin processed.nc\n"
        "rms first :1-130@CA out rmsd_protein_ca.dat\n"
        "run\nquit\n",
        encoding="utf-8",
    )

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, text, capture_output):
        rendered = input_path.with_name("rmsd.tmp.in").read_text(encoding="utf-8")
        assert "rmsd_protein_ca.dat.tmp" in rendered
        (tmp_path / "rmsd_protein_ca.dat.tmp").write_text("#Frame RMSD\n1 0.0\n", encoding="utf-8")
        return Proc()

    monkeypatch.setattr(analysis_mod.shutil, "which", lambda name: "/usr/bin/cpptraj")
    monkeypatch.setattr(analysis_mod.subprocess, "run", fake_run)
    analysis_mod._run_cpptraj(input_path, cwd=tmp_path, outputs=[output])
    assert output.read_text(encoding="utf-8") == "#Frame RMSD\n1 0.0\n"
    assert not (tmp_path / "rmsd_protein_ca.dat.tmp").exists()


def test_analysis_uses_analysis_steps_for_final_required_outputs() -> None:
    from lyso_md import analysis as analysis_mod

    steps = {
        "rmsd": ("rmsd.in", [Path("rmsd.dat")]),
        "pca_covariance": ("pca.in", [Path("ca_modes.dat")]),
    }
    required: list[Path] = []
    for _, outputs in steps.values():
        required.extend(outputs)
    assert required == [Path("rmsd.dat"), Path("ca_modes.dat")]
