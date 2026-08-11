from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from . import __version__
from .config import PipelineConfig
from .workspace import sha256_file, utc_now


@dataclass(frozen=True)
class ChaiResult:
    stage_dir: Path
    selected_pdb: Path | None
    validation_path: Path
    dry_run: bool


def normalize_smiles(smiles: str) -> str:
    """Remove YAML folding whitespace without altering SMILES punctuation."""
    return "".join(smiles.split())


def ligand_heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    if mol is None:
        raise ValueError("chai.glycan_smiles is not a valid RDKit SMILES")
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _read_fasta_sequence(path: Path) -> str:
    seq = "".join(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith(">"))
    if not seq:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return seq


def write_chai_input(cfg: PipelineConfig, path: Path) -> None:
    seq = _read_fasta_sequence(cfg.protein.fasta)
    text = f">protein|name={cfg.name}\n{seq}\n>ligand|name={cfg.chai.ligand_resname}\n{normalize_smiles(cfg.chai.glycan_smiles)}\n"
    path.write_text(text, encoding="utf-8")


def build_chai_shell_command(cfg: PipelineConfig, input_fasta: Path, output_dir: Path) -> str:
    c = cfg.chai
    parts = []
    if c.mamba_init:
        parts.append(f"source {shlex.quote(c.mamba_init)}")
    if c.mamba_env:
        parts.append(f"mamba activate {shlex.quote(c.mamba_env)}")
    command = f"{c.command} {shlex.quote(str(input_fasta))} {shlex.quote(str(output_dir))}"
    parts.append(command)
    return " && ".join(parts)


def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path, ligand_resname: str) -> None:
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError("gemmi is required to convert Chai CIF output to PDB") from exc
    st = gemmi.read_structure(str(cif_path))
    aa = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name.upper() not in aa and residue.name.upper() not in {"HOH","WAT"}:
                    residue.name = ligand_resname[:3]
    st.write_pdb(str(pdb_path))


def _element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element and element[0].isalpha():
        return element.upper()
    name = line[12:16].strip()
    letters = "".join(ch for ch in name if ch.isalpha())
    return (letters[:1] or "?").upper()


def validate_chai_pdb(pdb_path: Path, *, expected_residues: int, ligand_resname: str, expected_ligand_heavy_atoms: int) -> dict[str, Any]:
    serials: list[int] = []
    protein_residues: set[tuple[str, str, str]] = set()
    ligand_heavy = 0
    finite = True
    ligand_present = False
    for raw in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (raw.startswith("ATOM  ") or raw.startswith("HETATM")):
            continue
        try:
            serials.append(int(raw[6:11]))
            x, y, z = float(raw[30:38]), float(raw[38:46]), float(raw[46:54])
            finite = finite and all(math.isfinite(v) for v in (x, y, z))
        except (ValueError, IndexError):
            finite = False
        resname = raw[17:20].strip()
        if raw.startswith("ATOM  "):
            protein_residues.add((raw[21:22], raw[22:26].strip(), raw[26:27]))
        elif resname == ligand_resname:
            ligand_present = True
            if _element_from_pdb_line(raw) != "H":
                ligand_heavy += 1
    checks = {
        "selected_model_exists": pdb_path.is_file() and pdb_path.stat().st_size > 0,
        "protein_residue_count": len(protein_residues),
        "protein_residue_count_ok": len(protein_residues) == expected_residues,
        "ligand_present": ligand_present,
        "ligand_heavy_atoms": ligand_heavy,
        "ligand_heavy_atom_count_ok": ligand_heavy == expected_ligand_heavy_atoms,
        "finite_coordinates": finite,
        "unique_atom_serials": len(serials) == len(set(serials)),
    }
    checks["passed"] = all(v for k, v in checks.items() if k.endswith("_ok") or k in {"selected_model_exists","ligand_present","finite_coordinates","unique_atom_serials"})
    return checks


def run_chai(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> ChaiResult:
    stage = workspace / "01_chai"
    stage.mkdir(parents=True, exist_ok=True)
    input_fasta = stage / "chai_input.fasta"
    output_dir = stage / "chai_output"
    log_path = stage / "chai.log"
    validation_path = stage / "validation.json"
    command_path = stage / "expected_command.sh"
    sentinel = stage / ".done"
    if sentinel.exists():
        sentinel.unlink()
    write_chai_input(cfg, input_fasta)
    command = build_chai_shell_command(cfg, input_fasta, output_dir)
    command_path.write_text("#!/bin/bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
    expected = ligand_heavy_atom_count(cfg.chai.glycan_smiles)
    if cfg.glycam.expected_heavy_atoms is not None and expected != cfg.glycam.expected_heavy_atoms:
        raise ValueError(f"SMILES heavy-atom count {expected} does not match glycam.expected_heavy_atoms {cfg.glycam.expected_heavy_atoms}")
    if dry_run:
        payload = {"stage":"chai","status":"dry_run","command":command,"expected_ligand_heavy_atoms":expected,"passed":False}
        validation_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
        log_path.write_text("DRY RUN: Chai was not executed.\n"+command+"\n")
        return ChaiResult(stage, None, validation_path, True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    proc = subprocess.run(["bash","-lc",command], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Chai command failed with exit code {proc.returncode}; see {log_path}")
    selected_cif = output_dir / f"pred.model_idx_{cfg.chai.model_index}.cif"
    selected_pdb = stage / f"pred.model_idx_{cfg.chai.model_index}.pdb"
    if not selected_cif.is_file():
        candidates = list(output_dir.rglob(f"pred.model_idx_{cfg.chai.model_index}.cif"))
        if len(candidates) != 1:
            raise RuntimeError(f"selected Chai CIF not found uniquely for model index {cfg.chai.model_index}")
        selected_cif = candidates[0]
    _convert_cif_to_pdb(selected_cif, selected_pdb, cfg.chai.ligand_resname)
    checks = validate_chai_pdb(selected_pdb, expected_residues=cfg.protein.expected_residues, ligand_resname=cfg.chai.ligand_resname, expected_ligand_heavy_atoms=expected)
    payload = {"stage":"chai","status":"done" if checks["passed"] else "failed","selected_cif":str(selected_cif),"selected_pdb":str(selected_pdb),"expected_ligand_heavy_atoms":expected,"checks":checks}
    validation_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    if not checks["passed"]:
        raise RuntimeError(f"Chai output validation failed; see {validation_path}")
    done = {"stage":"chai","status":"done","completed_at":utc_now(),"pipeline_version":__version__,"input_sha256":sha256_file(input_fasta),"selected_cif_sha256":sha256_file(selected_cif),"selected_pdb_sha256":sha256_file(selected_pdb),"validation_sha256":sha256_file(validation_path),"validation":checks}
    sentinel.write_text(json.dumps(done, indent=2, sort_keys=True)+"\n")
    return ChaiResult(stage, selected_pdb, validation_path, False)
