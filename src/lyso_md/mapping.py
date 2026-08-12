from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from rdkit import Chem

from . import __version__
from .chai import normalize_smiles
from .config import PipelineConfig
from .glycam import OffUnit, parse_off_unit


@dataclass(frozen=True)
class LigandAtom:
    serial: int
    name: str
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MappingResult:
    stage_dir: Path
    mapping_path: Path
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


def _pdb_atom_tokens(line: str) -> tuple[int, str, str, float, float, float, str]:
    """Parse a Chai/ProDy HETATM line, tolerating >4-character atom names.

    ProDy-generated Chai PDBs can shift fixed columns when atom names such as
    C10_1 exceed the PDB atom-name field. The whitespace representation remains
    unambiguous for the Chai ligand records, so use it as a guarded fallback.
    """
    parts = line.split()
    if len(parts) < 10:
        raise ValueError(f'malformed HETATM record: {line!r}')
    try:
        serial = int(parts[1])
        name = parts[2]
        resname = parts[3]
        x, y, z = map(float, parts[6:9])
    except (ValueError, IndexError) as exc:
        raise ValueError(f'malformed HETATM record: {line!r}') from exc
    element = parts[-1].upper()
    if not element.isalpha() or len(element) > 2:
        letters = ''.join(ch for ch in name if ch.isalpha())
        element = (letters[:1] or '?').upper()
    return serial, name, resname, x, y, z, element


def read_chai_ligand_atoms(pdb_path: Path, ligand_resname: str) -> list[LigandAtom]:
    atoms: list[LigandAtom] = []
    for raw in Path(pdb_path).read_text(encoding='utf-8', errors='replace').splitlines():
        if not raw.startswith('HETATM'):
            continue
        serial, name, resname, x, y, z, element = _pdb_atom_tokens(raw)
        if resname != ligand_resname:
            continue
        if element == 'H':
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError(f'non-finite Chai ligand coordinate at atom serial {serial}')
        atoms.append(LigandAtom(serial, name, element, x, y, z))
    if not atoms:
        raise ValueError(f'Chai ligand residue {ligand_resname!r} was not found in {pdb_path}')
    serials = [a.serial for a in atoms]
    if len(serials) != len(set(serials)):
        raise ValueError('Chai ligand atom serials are not unique')
    return atoms


def _smiles_graph(smiles: str) -> tuple[Chem.Mol, nx.Graph]:
    mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    if mol is None:
        raise ValueError('chai.glycan_smiles is not a valid RDKit SMILES')
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_number=atom.GetAtomicNum(), symbol=atom.GetSymbol().upper())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return mol, graph


def _off_heavy_graph(unit: OffUnit) -> tuple[nx.Graph, dict[int, dict[str, Any]]]:
    heavy = {int(atom['index']): atom for atom in unit.atoms if int(atom['atomic_number']) != 1}
    graph = nx.Graph()
    for index, atom in heavy.items():
        graph.add_node(index, atomic_number=int(atom['atomic_number']))
    for edge in unit.connectivity:
        a, b = int(edge['atom1']), int(edge['atom2'])
        if a in heavy and b in heavy:
            graph.add_edge(a, b)
    return graph, heavy


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _geometry_score(
    smiles_graph: nx.Graph,
    mapping: dict[int, int],
    chai_atoms: list[LigandAtom],
    off_atoms: dict[int, dict[str, Any]],
) -> float:
    """Local-distance RMS disagreement for graph-neighbor pairs within two bonds."""
    sq: list[float] = []
    all_lengths = dict(nx.all_pairs_shortest_path_length(smiles_graph, cutoff=2))
    for i, distances in all_lengths.items():
        for j, graph_distance in distances.items():
            if i >= j or graph_distance not in (1, 2):
                continue
            ca = chai_atoms[i]
            cb = chai_atoms[j]
            oa = off_atoms[mapping[i]]
            ob = off_atoms[mapping[j]]
            dc = _distance((ca.x, ca.y, ca.z), (cb.x, cb.y, cb.z))
            do = _distance((float(oa['x']), float(oa['y']), float(oa['z'])), (float(ob['x']), float(ob['y']), float(ob['z'])))
            sq.append((dc - do) ** 2)
    if not sq:
        raise ValueError('cannot score mapping geometry: no local heavy-atom pairs')
    return math.sqrt(sum(sq) / len(sq))


def _equivalent_off_atoms(a: dict[str, Any], b: dict[str, Any], off_graph: nx.Graph) -> bool:
    if int(a['atomic_number']) != int(b['atomic_number']):
        return False
    if int(a['residue_index']) != int(b['residue_index']):
        return False
    if a.get('type') != b.get('type'):
        return False
    if not math.isclose(float(a.get('charge', 0.0)), float(b.get('charge', 0.0)), abs_tol=1e-8):
        return False
    ai, bi = int(a['index']), int(b['index'])
    if off_graph.degree(ai) != off_graph.degree(bi):
        return False
    # Require the same neighbor set. This intentionally recognizes cases such
    # as the GLYCAM carboxylate O3A/O3B pair, which are parameter-identical and
    # bonded to the same carbon.
    return set(off_graph.neighbors(ai)) == set(off_graph.neighbors(bi))


def _canonicalize_near_tie(
    candidates: list[tuple[float, dict[int, int]]],
    off_atoms: dict[int, dict[str, Any]],
    off_graph: nx.Graph,
    *,
    tolerance: float = 0.001,
) -> tuple[float, dict[int, int], bool, list[dict[str, Any]]]:
    """Resolve only scientifically equivalent graph automorphisms deterministically."""
    candidates = sorted(candidates, key=lambda item: item[0])
    best_score = candidates[0][0]
    tied = [(score, mapping) for score, mapping in candidates if score - best_score <= tolerance]
    if len(tied) == 1:
        return tied[0][0], tied[0][1], False, []

    reference = tied[0][1]
    equivalent_swaps: list[dict[str, Any]] = []
    for _, other in tied[1:]:
        differing = [idx for idx in reference if reference[idx] != other[idx]]
        for idx in differing:
            a = off_atoms[reference[idx]]
            b = off_atoms[other[idx]]
            if not _equivalent_off_atoms(a, b, off_graph):
                raise ValueError(
                    'mapping ambiguity remains after geometric scoring and changes non-equivalent GLYCAM atoms; '
                    f'SMILES atom {idx} could map to OFF atoms {a["index"]} ({a["name"]}) or {b["index"]} ({b["name"]})'
                )
            equivalent_swaps.append(
                {
                    'smiles_index_0based': idx,
                    'off_atom_a': int(a['index']),
                    'off_name_a': a['name'],
                    'off_atom_b': int(b['index']),
                    'off_name_b': b['name'],
                    'residue_index': int(a['residue_index']),
                }
            )

    # Equivalent parameter-identical atoms are scientifically interchangeable.
    # Canonical OFF-index order makes the audit table deterministic and matches
    # the historical reference mapping while preserving fail-closed behavior for
    # non-equivalent ambiguity.
    chosen = min(tied, key=lambda item: tuple(item[1][i] for i in sorted(item[1])))
    unique_swaps = {json.dumps(item, sort_keys=True): item for item in equivalent_swaps}
    return chosen[0], chosen[1], True, list(unique_swaps.values())


def map_chai_to_glycam(cfg: PipelineConfig, *, workspace: Path) -> MappingResult:
    workspace = Path(workspace).resolve()
    chai_pdb = workspace / '01_chai' / f'pred.model_idx_{cfg.chai.model_index}.pdb'
    glycam_stage = workspace / '02_prepare' / 'glycam'
    structure_off = glycam_stage / 'extracted' / 'structure' / 'structure.off'
    if not structure_off.is_file():
        matches = list((glycam_stage / 'extracted').rglob('structure.off')) if (glycam_stage / 'extracted').is_dir() else []
        if len(matches) == 1:
            structure_off = matches[0]
    if not chai_pdb.is_file():
        raise ValueError(f'Phase 4 requires completed Chai PDB: {chai_pdb}')
    if not (workspace / '01_chai' / '.done').is_file():
        raise ValueError('Phase 4 requires a validated Phase 2 .done sentinel')
    if not structure_off.is_file() or not (glycam_stage / '.done').is_file():
        raise ValueError('Phase 4 requires completed Phase 3 GLYCAM inspection')

    stage = workspace / '02_prepare' / 'mapping'
    stage.mkdir(parents=True, exist_ok=True)
    mapping_path = stage / 'atom_mapping.tsv'
    validation_path = stage / 'validation.json'
    sentinel_path = stage / '.done'
    if sentinel_path.exists():
        sentinel_path.unlink()

    chai_atoms = read_chai_ligand_atoms(chai_pdb, cfg.chai.ligand_resname)
    mol, smiles_graph = _smiles_graph(cfg.chai.glycan_smiles)
    unit = parse_off_unit(structure_off, cfg.glycam.unit_name)
    off_graph, off_atoms = _off_heavy_graph(unit)

    if len(chai_atoms) != mol.GetNumAtoms():
        raise ValueError(f'Chai/SMILES heavy-atom count mismatch: {len(chai_atoms)} vs {mol.GetNumAtoms()}')
    if len(off_atoms) != mol.GetNumAtoms():
        raise ValueError(f'GLYCAM/SMILES heavy-atom count mismatch: {len(off_atoms)} vs {mol.GetNumAtoms()}')

    smiles_elements = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
    chai_elements = [atom.element.upper() for atom in chai_atoms]
    if Counter(smiles_elements) != Counter(chai_elements):
        raise ValueError(f'Chai/SMILES element counts differ: {Counter(chai_elements)} vs {Counter(smiles_elements)}')
    off_elements = Counter(Chem.GetPeriodicTable().GetElementSymbol(int(a['atomic_number'])).upper() for a in off_atoms.values())
    if Counter(smiles_elements) != off_elements:
        raise ValueError(f'GLYCAM/SMILES element counts differ: {off_elements} vs {Counter(smiles_elements)}')

    # Chai preserves the input ligand atom ordering. Validate that assumption
    # element-by-element before using the stereospecific SMILES graph as the
    # Chai ligand graph; fail rather than guessing if an upstream converter has
    # reordered atoms.
    order_mismatches = [i for i, (a, b) in enumerate(zip(chai_elements, smiles_elements)) if a != b]
    if order_mismatches:
        raise ValueError(
            'Chai ligand atom order no longer matches the input SMILES element sequence; '
            f'first mismatched indices: {order_mismatches[:10]}'
        )

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        smiles_graph,
        off_graph,
        node_match=lambda a, b: int(a['atomic_number']) == int(b['atomic_number']),
    )
    candidates: list[tuple[float, dict[int, int]]] = []
    max_candidates = 10000
    for idx, candidate in enumerate(matcher.isomorphisms_iter(), start=1):
        if idx > max_candidates:
            raise ValueError(f'graph mapping produced more than {max_candidates} isomorphisms; refusing ambiguous search')
        candidates.append((_geometry_score(smiles_graph, candidate, chai_atoms, off_atoms), dict(candidate)))
    if not candidates:
        raise ValueError('Chai/SMILES and GLYCAM heavy-atom molecular graphs are not isomorphic')

    score, selected, used_equivalent_tiebreak, equivalent_swaps = _canonicalize_near_tie(candidates, off_atoms, off_graph)

    residues = {int(r['index']): str(r['name']) for r in unit.residues}
    rows: list[dict[str, Any]] = []
    for smiles_index in range(mol.GetNumAtoms()):
        chai = chai_atoms[smiles_index]
        off = off_atoms[selected[smiles_index]]
        rows.append(
            {
                'chai_serial': chai.serial,
                'chai_atom': chai.name,
                'smiles_index_0based': smiles_index,
                'glycam_atom': off['name'],
                'glycam_resname': residues[int(off['residue_index'])],
                'glycam_resid': int(off['residue_index']),
                'element': chai.element,
            }
        )

    fieldnames = ['chai_serial', 'chai_atom', 'smiles_index_0based', 'glycam_atom', 'glycam_resname', 'glycam_resid', 'element']
    with mapping_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    candidate_scores = sorted(score for score, _ in candidates)
    validation = {
        'stage': 'chai_to_glycam_mapping',
        'status': 'done',
        'chai_pdb': str(chai_pdb.relative_to(workspace)),
        'structure_off': str(structure_off.relative_to(workspace)),
        'unit_name': unit.name,
        'counts': {
            'chai_heavy_atoms': len(chai_atoms),
            'smiles_heavy_atoms': mol.GetNumAtoms(),
            'glycam_heavy_atoms': len(off_atoms),
            'mapping_rows': len(rows),
        },
        'element_counts': dict(sorted(Counter(smiles_elements).items())),
        'graph': {
            'smiles_edges': smiles_graph.number_of_edges(),
            'glycam_heavy_edges': off_graph.number_of_edges(),
            'isomorphic': True,
            'isomorphism_candidates': len(candidates),
            'candidate_geometry_scores_angstrom': candidate_scores,
            'selected_geometry_score_angstrom': score,
            'equivalent_atom_tiebreak_used': used_equivalent_tiebreak,
            'equivalent_swaps': equivalent_swaps,
        },
        'checks': {
            'heavy_atom_counts_match': len(chai_atoms) == mol.GetNumAtoms() == len(off_atoms),
            'element_counts_match': True,
            'chai_smiles_atom_order_matches': True,
            'graphs_isomorphic': True,
            'mapping_bijective': len(set(selected.values())) == len(selected) == mol.GetNumAtoms(),
            'passed': True,
        },
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    sentinel = {
        'stage': 'chai_to_glycam_mapping',
        'status': 'done',
        'completed_at': _utc_now(),
        'pipeline_version': __version__,
        'chai_pdb_sha256': _sha256(chai_pdb),
        'structure_off_sha256': _sha256(structure_off),
        'mapping_sha256': _sha256(mapping_path),
        'validation_sha256': _sha256(validation_path),
        'mapping_rows': len(rows),
    }
    sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return MappingResult(stage, mapping_path, validation_path, sentinel_path)
