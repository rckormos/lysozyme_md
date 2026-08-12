from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import PipelineConfig
from .glycam import OffUnit, parse_off_unit
from .mapping import read_chai_ligand_atoms

_SECTION_RE = re.compile(r"^!entry\.([^.]+)\.unit\.([^\s]+)\s+(.*)$")


@dataclass(frozen=True)
class CoordinateTransferResult:
    stage_dir: Path
    aligned_off_path: Path
    mapping_copy_path: Path
    repair_path: Path
    validation_path: Path
    sentinel_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _vec(atom: dict[str, Any]) -> np.ndarray:
    return np.array([float(atom['x']), float(atom['y']), float(atom['z'])], dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, *, label: str) -> np.ndarray:
    n = _norm(v)
    if not math.isfinite(n) or n < 1e-10:
        raise ValueError(f'cannot normalize degenerate vector for {label}')
    return v / n


def _rotation_align_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a proper rotation mapping unit vector a onto unit vector b."""
    a = _unit(a, label='single-neighbor old axis')
    b = _unit(b, label='single-neighbor new axis')
    cross = np.cross(a, b)
    s = _norm(cross)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degree turn around any stable axis orthogonal to a.
        basis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, basis), label='antiparallel rotation axis')
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = cross / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + kx * s + (kx @ kx) * (1.0 - c)


def _kabsch_rotation(old_vectors: list[np.ndarray], new_vectors: list[np.ndarray]) -> np.ndarray:
    if len(old_vectors) != len(new_vectors) or len(old_vectors) < 2:
        raise ValueError('Kabsch local-frame rotation requires at least two paired vectors')
    p = np.stack([_unit(v, label='old local-frame vector') for v in old_vectors], axis=0)
    q = np.stack([_unit(v, label='new local-frame vector') for v in new_vectors], axis=0)
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return r


def _adjacency(unit: OffUnit) -> dict[int, list[int]]:
    adj = {int(atom['index']): [] for atom in unit.atoms}
    for edge in unit.connectivity:
        a, b = int(edge['atom1']), int(edge['atom2'])
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _read_mapping(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    required = {'chai_serial', 'chai_atom', 'smiles_index_0based', 'glycam_atom', 'glycam_resname', 'glycam_resid', 'element'}
    if not rows or set(rows[0]) != required:
        missing = required.difference(rows[0] if rows else set())
        raise ValueError(f'Phase 5 mapping TSV is malformed; missing columns: {sorted(missing)}')
    return rows


def _locate_structure_off(workspace: Path) -> Path:
    root = workspace / '02_prepare' / 'glycam' / 'extracted'
    matches = list(root.rglob('structure.off')) if root.is_dir() else []
    if len(matches) != 1:
        raise ValueError(f'Phase 5 requires exactly one extracted structure.off; found {len(matches)}')
    return matches[0]


def _replace_off_positions(source: Path, destination: Path, unit_name: str, coordinates: list[np.ndarray]) -> None:
    lines = Path(source).read_text(encoding='utf-8', errors='strict').splitlines()
    output: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        raw = lines[i]
        match = _SECTION_RE.match(raw.strip())
        if match and match.group(1) == unit_name and match.group(2) == 'positions':
            if replaced:
                raise ValueError(f'OFF unit {unit_name!r} contains multiple positions tables')
            output.append(raw)
            i += 1
            old_rows = 0
            while i < len(lines) and not lines[i].lstrip().startswith('!'):
                if lines[i].strip():
                    old_rows += 1
                i += 1
            if old_rows != len(coordinates):
                raise ValueError(f'OFF positions row count mismatch: {old_rows} vs {len(coordinates)} atoms')
            for xyz in coordinates:
                if not np.all(np.isfinite(xyz)):
                    raise ValueError('refusing to write non-finite OFF coordinates')
                output.append(f' {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}')
            replaced = True
            continue
        output.append(raw)
        i += 1
    if not replaced:
        raise ValueError(f'OFF unit {unit_name!r} has no positions table')
    Path(destination).write_text('\n'.join(output) + '\n', encoding='utf-8')


def _topology_signature(unit: OffUnit) -> dict[str, Any]:
    return {
        'atoms': [
            (
                int(a['index']), str(a['name']), str(a['type']), int(a['residue_index']),
                int(a['atomic_number']), round(float(a['charge']), 10),
            )
            for a in unit.atoms
        ],
        'residues': [(int(r['index']), str(r['name']), int(r['start_atom_index']), str(r['residue_type'])) for r in unit.residues],
        'connectivity': [(int(e['atom1']), int(e['atom2']), int(e['flags'])) for e in unit.connectivity],
    }


def transfer_glycan_coordinates(cfg: PipelineConfig, *, workspace: Path) -> CoordinateTransferResult:
    """Phase 5: transfer Chai heavy coordinates into GLYCAM and rebuild local H geometry."""
    workspace = Path(workspace).resolve()
    mapping_stage = workspace / '02_prepare' / 'mapping'
    mapping_source = mapping_stage / 'atom_mapping.tsv'
    if not (mapping_stage / '.done').is_file() or not mapping_source.is_file():
        raise ValueError('Phase 5 requires completed Phase 4 Chai-to-GLYCAM mapping')
    chai_pdb = workspace / '01_chai' / f'pred.model_idx_{cfg.chai.model_index}.pdb'
    if not chai_pdb.is_file():
        raise ValueError(f'Phase 5 requires Chai PDB: {chai_pdb}')
    structure_off = _locate_structure_off(workspace)

    stage = workspace / '02_prepare' / 'coordinate_transfer'
    stage.mkdir(parents=True, exist_ok=True)
    aligned_off = stage / 'glycan_aligned.off'
    mapping_copy = stage / 'atom_mapping.tsv'
    repairs_tsv = stage / 'tetrahedral_hydrogen_repairs.tsv'
    validation_path = stage / 'hydrogen_validation.json'
    sentinel_path = stage / '.done'
    if sentinel_path.exists():
        sentinel_path.unlink()

    source_unit = parse_off_unit(structure_off, cfg.glycam.unit_name)
    atoms = {int(a['index']): a for a in source_unit.atoms}
    residues = {int(r['index']): str(r['name']) for r in source_unit.residues}
    adj = _adjacency(source_unit)
    mapping_rows = _read_mapping(mapping_source)
    chai_atoms = {a.serial: a for a in read_chai_ligand_atoms(chai_pdb, cfg.chai.ligand_resname)}

    by_res_atom: dict[tuple[int, str], int] = {}
    for atom in source_unit.atoms:
        key = (int(atom['residue_index']), str(atom['name']))
        if key in by_res_atom:
            raise ValueError(f'duplicate GLYCAM atom name within residue: {key}')
        by_res_atom[key] = int(atom['index'])

    new_coords = {idx: _vec(atom).copy() for idx, atom in atoms.items()}
    mapped_indices: set[int] = set()
    for row in mapping_rows:
        resid = int(row['glycam_resid'])
        key = (resid, row['glycam_atom'])
        if key not in by_res_atom:
            raise ValueError(f'mapping references unknown GLYCAM atom {key}')
        off_idx = by_res_atom[key]
        serial = int(row['chai_serial'])
        if serial not in chai_atoms:
            raise ValueError(f'mapping references unknown Chai ligand serial {serial}')
        off_atom = atoms[off_idx]
        chai_atom = chai_atoms[serial]
        if int(off_atom['atomic_number']) == 1:
            raise ValueError(f'Phase 4 mapping unexpectedly references hydrogen OFF atom {key}')
        if row['element'].upper() != chai_atom.element.upper():
            raise ValueError(f'mapping element mismatch for Chai serial {serial}')
        if off_idx in mapped_indices:
            raise ValueError(f'mapping is not bijective: duplicate GLYCAM atom index {off_idx}')
        mapped_indices.add(off_idx)
        new_coords[off_idx] = np.array([chai_atom.x, chai_atom.y, chai_atom.z], dtype=float)

    heavy_indices = {idx for idx, atom in atoms.items() if int(atom['atomic_number']) != 1}
    if mapped_indices != heavy_indices:
        missing = sorted(heavy_indices.difference(mapped_indices))
        extra = sorted(mapped_indices.difference(heavy_indices))
        raise ValueError(f'Phase 5 requires all and only GLYCAM heavy atoms mapped; missing={missing}, extra={extra}')

    repair_rows: list[dict[str, Any]] = []
    ordinary_hydrogens = 0
    bond_errors: list[float] = []
    for h_idx, h_atom in atoms.items():
        if int(h_atom['atomic_number']) != 1:
            continue
        neighbors = adj[h_idx]
        if len(neighbors) != 1:
            raise ValueError(f'GLYCAM hydrogen atom {h_idx} must have exactly one bonded neighbor; found {len(neighbors)}')
        parent_idx = neighbors[0]
        parent = atoms[parent_idx]
        if int(parent['atomic_number']) == 1:
            raise ValueError(f'GLYCAM hydrogen atom {h_idx} is bonded to another hydrogen')
        old_parent = _vec(parent)
        old_h = _vec(h_atom)
        r_xh = _norm(old_h - old_parent)
        if not math.isfinite(r_xh) or r_xh < 0.1:
            raise ValueError(f'invalid original GLYCAM X-H bond length at atom {h_idx}: {r_xh}')
        heavy_neighbors = [n for n in adj[parent_idx] if int(atoms[n]['atomic_number']) != 1]
        hydrogen_neighbors = [n for n in adj[parent_idx] if int(atoms[n]['atomic_number']) == 1]
        new_parent = new_coords[parent_idx]

        is_tetrahedral_ch = (
            int(parent['atomic_number']) == 6
            and len(heavy_neighbors) == 3
            and len(hydrogen_neighbors) == 1
        )
        if is_tetrahedral_ch:
            u = [_unit(new_coords[n] - new_parent, label=f'tetrahedral neighbor {n}') for n in heavy_neighbors]
            direction = -_unit(u[0] + u[1] + u[2], label=f'tetrahedral missing direction at atom {parent_idx}')
            new_h = new_parent + r_xh * direction
            repair_rows.append({
                'hydrogen_index': h_idx,
                'hydrogen_atom': h_atom['name'],
                'parent_index': parent_idx,
                'parent_atom': parent['name'],
                'residue_index': int(parent['residue_index']),
                'residue_name': residues[int(parent['residue_index'])],
                'original_bond_length_angstrom': f'{r_xh:.6f}',
                'repaired_bond_length_angstrom': f'{_norm(new_h - new_parent):.6f}',
            })
        else:
            if len(heavy_neighbors) == 0:
                raise ValueError(f'cannot construct local frame for hydrogen {h_idx}: parent has no heavy-atom neighbor')
            old_vectors = [_vec(atoms[n]) - old_parent for n in heavy_neighbors]
            new_vectors = [new_coords[n] - new_parent for n in heavy_neighbors]
            if len(heavy_neighbors) == 1:
                rotation = _rotation_align_vector(old_vectors[0], new_vectors[0])
            else:
                rotation = _kabsch_rotation(old_vectors, new_vectors)
            direction = _unit(rotation @ (old_h - old_parent), label=f'rotated hydrogen direction {h_idx}')
            new_h = new_parent + r_xh * direction
            ordinary_hydrogens += 1
        new_coords[h_idx] = new_h
        bond_errors.append(abs(_norm(new_h - new_parent) - r_xh))

    mapping_copy.write_bytes(mapping_source.read_bytes())
    fieldnames = [
        'hydrogen_index', 'hydrogen_atom', 'parent_index', 'parent_atom',
        'residue_index', 'residue_name', 'original_bond_length_angstrom', 'repaired_bond_length_angstrom',
    ]
    with repairs_tsv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(repair_rows)

    ordered_coords = [new_coords[int(a['index'])] for a in source_unit.atoms]
    _replace_off_positions(structure_off, aligned_off, cfg.glycam.unit_name, ordered_coords)
    aligned_unit = parse_off_unit(aligned_off, cfg.glycam.unit_name)
    topology_preserved = _topology_signature(source_unit) == _topology_signature(aligned_unit)
    if not topology_preserved:
        raise ValueError('glycan_aligned.off changed GLYCAM topology, atom metadata, residue metadata, or charges')

    # Confirm every mapped heavy atom survived OFF serialization at the Chai coordinate.
    heavy_max_error = 0.0
    aligned_atoms = {int(a['index']): a for a in aligned_unit.atoms}
    for idx in mapped_indices:
        heavy_max_error = max(heavy_max_error, _norm(_vec(aligned_atoms[idx]) - new_coords[idx]))
    finite = all(np.all(np.isfinite(new_coords[idx])) for idx in new_coords)
    if not finite:
        raise ValueError('non-finite coordinates remain after Phase 5 hydrogen repair')
    if heavy_max_error > 1e-6:
        raise ValueError(f'aligned OFF heavy coordinates differ from transferred Chai coordinates by {heavy_max_error:.6g} A')
    max_bond_error = max(bond_errors, default=0.0)
    if max_bond_error > 1e-6:
        raise ValueError(f'hydrogen bond-length preservation error exceeds tolerance: {max_bond_error:.6g} A')

    validation = {
        'stage': 'coordinate_transfer_and_hydrogen_repair',
        'status': 'done',
        'unit_name': cfg.glycam.unit_name,
        'counts': {
            'atoms': len(source_unit.atoms),
            'heavy_atoms_transferred': len(mapped_indices),
            'hydrogens_total': len(source_unit.atoms) - len(mapped_indices),
            'tetrahedral_ch_repairs': len(repair_rows),
            'ordinary_hydrogens_local_frame_moved': ordinary_hydrogens,
        },
        'geometry': {
            'max_serialization_heavy_coordinate_error_angstrom': heavy_max_error,
            'max_hydrogen_bond_length_error_angstrom': max_bond_error,
        },
        'checks': {
            'all_heavy_atoms_mapped': mapped_indices == heavy_indices,
            'topology_and_parameters_preserved': topology_preserved,
            'finite_coordinates': finite,
            'hydrogen_bond_lengths_preserved': max_bond_error <= 1e-6,
            'passed': True,
        },
        'inputs': {
            'structure_off': str(structure_off.relative_to(workspace)),
            'chai_pdb': str(chai_pdb.relative_to(workspace)),
            'mapping': str(mapping_source.relative_to(workspace)),
        },
        'outputs': {
            'glycan_aligned_off': str(aligned_off.relative_to(workspace)),
            'mapping_copy': str(mapping_copy.relative_to(workspace)),
            'tetrahedral_hydrogen_repairs': str(repairs_tsv.relative_to(workspace)),
        },
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    sentinel = {
        'stage': 'coordinate_transfer_and_hydrogen_repair',
        'status': 'done',
        'completed_at': _utc_now(),
        'pipeline_version': __version__,
        'structure_off_sha256': _sha256(structure_off),
        'chai_pdb_sha256': _sha256(chai_pdb),
        'mapping_sha256': _sha256(mapping_source),
        'glycan_aligned_off_sha256': _sha256(aligned_off),
        'repair_audit_sha256': _sha256(repairs_tsv),
        'validation_sha256': _sha256(validation_path),
        'heavy_atoms_transferred': len(mapped_indices),
        'tetrahedral_ch_repairs': len(repair_rows),
    }
    sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return CoordinateTransferResult(stage, aligned_off, mapping_copy, repairs_tsv, validation_path, sentinel_path)
