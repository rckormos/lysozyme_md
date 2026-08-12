from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.mapping import map_chai_to_glycam, read_chai_ligand_atoms


OFF_TEXT = '''!index array str
 "CONDENSEDSEQUENCE"
!entry.CONDENSEDSEQUENCE.unit.atoms table str name str type int typex int resx int flags int seq int elmnt dbl chg
 "C1" "Cg" 0 1 0 1 6 0.100000
 "C2" "Cg" 0 1 0 2 6 0.100000
 "O1" "Oh" 0 1 0 3 8 -0.200000
!entry.CONDENSEDSEQUENCE.unit.residues table str name int seq int childseq int startatomx str restype int imagingx
 "LIG" 1 0 1 "?" 0
!entry.CONDENSEDSEQUENCE.unit.positions table dbl x dbl y dbl z
 0.0 0.0 0.0
 1.5 0.0 0.0
 2.9 0.0 0.0
!entry.CONDENSEDSEQUENCE.unit.connectivity table int atom1x int atom2x int flags
 1 2 1
 2 3 1
'''


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">x\nAC\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")
    data = {
        "name": "map_test",
        "protein": {"fasta": str(fasta), "expected_residues": 2},
        "chai": {"enabled": True, "model_index": 0, "ligand_resname": "LIG", "glycan_smiles": "CCO"},
        "glycam": {
            "bundle": str(bundle),
            "unit_name": "CONDENSEDSEQUENCE",
            "expected_heavy_atoms": 3,
            "expected_residues": 1,
        },
        "forcefield": {"protein": "ff19SB", "glycan": "GLYCAM_06j-1", "water": "OPC"},
        "solvent": {"buffer_angstrom": 12, "salt": "KCl", "concentration_molar": 0.05},
        "md": {"temperature_k": 300, "pressure_bar": 1, "cutoff_angstrom": 9, "production_timestep_fs": 2},
        "equilibration": {"hydrogen_relax_steps": 1000, "solvent_min_steps": 5000, "all_min_steps": 5000, "heat_ps": 100, "npt_5_ps": 250, "npt_1_ps": 250, "npt_free_ps": 500},
        "production": {"target_ns": 1000, "chunk_ns": 250, "walltime_hours": 72},
        "scheduler": {"type": "lsf", "project": "p", "gpu_queue": "gpu", "gpu_resource": "num=1/host", "memory": "16GB", "cores": 1},
    }
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    chai = workspace / "01_chai"
    chai.mkdir(parents=True)
    # Deliberately include a five-character atom name to exercise the tolerant
    # whitespace parser used for ProDy/Chai PDBs.
    (chai / "pred.model_idx_0.pdb").write_text(
        "HETATM 1001 C1_1 LIG B   1       0.000   0.000   0.000  1.00 70.00         B C  \n"
        "HETATM 1002 C10_1 LIG B   1       1.500   0.000   0.000  1.00 70.00         B C  \n"
        "HETATM 1003 O1_1 LIG B   1       2.900   0.000   0.000  1.00 70.00         B O  \n",
        encoding="utf-8",
    )
    (chai / ".done").write_text("{}\n", encoding="utf-8")
    glycam = workspace / "02_prepare" / "glycam"
    off_dir = glycam / "extracted" / "structure"
    off_dir.mkdir(parents=True)
    (off_dir / "structure.off").write_text(OFF_TEXT, encoding="utf-8")
    (glycam / ".done").write_text("{}\n", encoding="utf-8")
    return workspace


def test_read_chai_ligand_tolerates_long_atom_names(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    atoms = read_chai_ligand_atoms(workspace / "01_chai" / "pred.model_idx_0.pdb", "LIG")
    assert [a.name for a in atoms] == ["C1_1", "C10_1", "O1_1"]
    assert [a.element for a in atoms] == ["C", "C", "O"]


def test_map_chai_to_glycam_writes_auditable_bijection(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    workspace = _workspace(tmp_path)
    result = map_chai_to_glycam(cfg, workspace=workspace)

    with result.mapping_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 3
    assert [row["smiles_index_0based"] for row in rows] == ["0", "1", "2"]
    assert [row["glycam_atom"] for row in rows] == ["C1", "C2", "O1"]
    assert result.sentinel_path.is_file()

    validation = json.loads(result.validation_path.read_text())
    assert validation["checks"]["passed"] is True
    assert validation["checks"]["mapping_bijective"] is True
    assert validation["graph"]["isomorphism_candidates"] == 1


def test_map_fails_if_chai_order_does_not_match_smiles(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    workspace = _workspace(tmp_path)
    pdb = workspace / "01_chai" / "pred.model_idx_0.pdb"
    lines = pdb.read_text().splitlines()
    lines[1] = "HETATM 1002 O10_1 LIG B   1       1.500   0.000   0.000  1.00 70.00         B O  "
    lines[2] = "HETATM 1003 C3_1 LIG B   1       2.900   0.000   0.000  1.00 70.00         B C  "
    pdb.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="atom order no longer matches"):
        map_chai_to_glycam(cfg, workspace=workspace)
    assert not (workspace / "02_prepare" / "mapping" / ".done").exists()
