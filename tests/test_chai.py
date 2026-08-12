from __future__ import annotations

from pathlib import Path

import pytest

from lyso_md.chai import (
    build_chai_lsf_script,
    build_chai_shell_command,
    ligand_heavy_atom_count,
    submit_chai,
    validate_chai_pdb,
    write_chai_input,
)
from lyso_md.config import load_config


def _write_config(tmp_path: Path) -> Path:
    (tmp_path / "seq.fasta").write_text(">x\nAC\n")
    (tmp_path / "bundle.zip").write_bytes(b"PK")
    p = tmp_path / "c.yaml"
    p.write_text('''
name: test_design
protein: {fasta: seq.fasta, expected_residues: 2}
chai:
  enabled: true
  model_index: 0
  ligand_resname: LIG
  glycan_smiles: "CCO"
  command: chai-lab fold
  mamba_init: /opt/mamba.sh
  mamba_env: env_chai
  walltime: "06:00"
glycam:
  bundle: bundle.zip
  expected_heavy_atoms: 3
  expected_residues: 1
forcefield: {protein: ff19SB, glycan: GLYCAM_06j-1, water: OPC}
solvent: {buffer_angstrom: 12, salt: KCl, concentration_molar: 0.05}
md: {temperature_k: 300, pressure_bar: 1, cutoff_angstrom: 9, production_timestep_fs: 2}
equilibration: {hydrogen_relax_steps: 1000, solvent_min_steps: 5000, all_min_steps: 5000, heat_ps: 100, npt_5_ps: 250, npt_1_ps: 250, npt_free_ps: 500}
production: {target_ns: 1000, chunk_ns: 250, walltime_hours: 72}
scheduler: {type: lsf, project: p, gpu_queue: gpu, gpu_resource: num=1/host, memory: 32GB, cores: 1}
''')
    return p


def test_heavy_atom_count():
    assert ligand_heavy_atom_count("CCO") == 3


def test_write_input_and_command(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    inp = tmp_path / "chai.fasta"
    write_chai_input(cfg, inp)
    text = inp.read_text()
    assert ">protein|name=test_design" in text
    assert "\nAC\n" in text
    assert ">ligand|name=LIG" in text
    assert text.rstrip().endswith("CCO")
    cmd = build_chai_shell_command(cfg, inp, tmp_path / "out")
    assert "source /opt/mamba.sh" in cmd
    assert "mamba activate env_chai" in cmd
    assert "chai-lab fold" in cmd


def test_validate_pdb(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  CA  CYS A   2       1.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    3  C1  LIG B   1       2.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    4  C2  LIG B   1       3.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    5  O1  LIG B   1       4.000   0.000   0.000  1.00  0.00           O  \nEND\n"
    )
    v = validate_chai_pdb(pdb, expected_residues=2, ligand_resname="LIG", expected_ligand_heavy_atoms=3)
    assert v["passed"] is True
    assert v["ligand_heavy_atoms"] == 3


def test_invalid_smiles_fails():
    with pytest.raises(ValueError):
        ligand_heavy_atom_count("not a smiles (((")



def test_build_lsf_script(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "logs").mkdir()
    (workspace / "config.yaml").write_text("# fixture\n")
    script = build_chai_lsf_script(cfg, workspace=workspace, pipeline_executable="/opt/lyso/bin/lyso-md")
    assert "#BSUB -P p" in script
    assert "#BSUB -J test_design_chai" in script
    assert "#BSUB -q gpu" in script
    assert "#BSUB -W 06:00" in script
    assert '#BSUB -gpu "num=1/host"' in script
    assert '#BSUB -R "rusage[mem=32GB]"' in script
    assert f"#BSUB -oo {workspace.resolve()}/logs/chai.%J.out" in script
    assert f"#BSUB -eo {workspace.resolve()}/logs/chai.%J.err" in script
    assert "set -euo pipefail" in script
    assert "/opt/lyso/bin/lyso-md _chai-worker" in script


def test_submit_chai_uses_bsub_and_records_job_id(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "logs").mkdir()
    (workspace / "config.yaml").write_text(cfg_path.read_text())

    monkeypatch.setattr("lyso_md.chai.shutil.which", lambda name: "/opt/lyso/bin/lyso-md")

    class Proc:
        returncode = 0
        stdout = "Job <123456> is submitted to queue <gpu>.\n"

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Proc()

    monkeypatch.setattr("lyso_md.chai.subprocess.run", fake_run)
    result = submit_chai(cfg, workspace=workspace, dry_run=False)
    assert result.job_id == "123456"
    assert calls[0][0] == ["bsub"]
    assert "#BSUB -q gpu" in calls[0][1]["input"]
    metadata = result.submission_path.read_text()
    assert '"job_id": "123456"' in metadata
    assert '"status": "submitted"' in metadata
    assert not (workspace / "01_chai" / ".done").exists()


def test_submit_chai_dry_run_does_not_call_bsub(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "logs").mkdir()
    (workspace / "config.yaml").write_text(cfg_path.read_text())
    monkeypatch.setattr("lyso_md.chai.shutil.which", lambda name: "/opt/lyso/bin/lyso-md")

    def fail_run(*args, **kwargs):
        raise AssertionError("bsub must not be called during --dry-run")

    monkeypatch.setattr("lyso_md.chai.subprocess.run", fail_run)
    result = submit_chai(cfg, workspace=workspace, dry_run=True)
    assert result.dry_run is True
    assert result.job_id is None
    assert (workspace / "01_chai" / "chai.lsf").is_file()
    assert not (workspace / "01_chai" / ".done").exists()
