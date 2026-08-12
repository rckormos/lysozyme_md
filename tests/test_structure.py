from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lyso_md.config import load_config
from lyso_md.glycam import parse_off_unit
from lyso_md.structure import transfer_glycan_coordinates


OFF_TEXT = '''!index array str
 "CONDENSEDSEQUENCE"
!entry.CONDENSEDSEQUENCE.unit.atoms table str name str type int typex int resx int flags int seq int elmnt dbl chg
 "C0" "Cg" 0 1 0 1 6 0.000000
 "C1" "Cg" 0 1 0 2 6 0.000000
 "C2" "Cg" 0 1 0 3 6 0.000000
 "C3" "Cg" 0 1 0 4 6 0.000000
 "H0" "H1" 0 1 0 5 1 0.000000
 "H1" "H1" 0 1 0 6 1 0.000000
!entry.CONDENSEDSEQUENCE.unit.residues table str name int seq int childseq int startatomx str restype int imagingx
 "LIG" 1 0 1 "?" 0
!entry.CONDENSEDSEQUENCE.unit.positions table dbl x dbl y dbl z
 0.000000 0.000000 0.000000
 1.000000 0.000000 0.000000
 0.000000 1.000000 0.000000
 0.000000 0.000000 1.000000
 -0.577350 -0.577350 -0.577350
 2.000000 0.000000 0.000000
!entry.CONDENSEDSEQUENCE.unit.connectivity table int atom1x int atom2x int flags
 1 2 1
 1 3 1
 1 4 1
 1 5 1
 2 6 1
'''


def _config(tmp_path: Path) -> Path:
    fasta = tmp_path / 'sequence.fasta'
    fasta.write_text('>x\nAC\n', encoding='utf-8')
    bundle = tmp_path / 'bundle.zip'
    bundle.write_bytes(b'PK')
    data = {
        'name': 'phase5_test',
        'protein': {'fasta': str(fasta), 'expected_residues': 2},
        'chai': {'enabled': True, 'model_index': 0, 'ligand_resname': 'LIG', 'glycan_smiles': 'CCCC'},
        'glycam': {'bundle': str(bundle), 'unit_name': 'CONDENSEDSEQUENCE', 'expected_heavy_atoms': 4, 'expected_residues': 1},
        'forcefield': {'protein': 'ff19SB', 'glycan': 'GLYCAM_06j-1', 'water': 'OPC'},
        'solvent': {'buffer_angstrom': 12, 'salt': 'KCl', 'concentration_molar': 0.05},
        'md': {'temperature_k': 300, 'pressure_bar': 1, 'cutoff_angstrom': 9, 'production_timestep_fs': 2},
        'equilibration': {'hydrogen_relax_steps': 1000, 'solvent_min_steps': 5000, 'all_min_steps': 5000, 'heat_ps': 100, 'npt_5_ps': 250, 'npt_1_ps': 250, 'npt_free_ps': 500},
        'production': {'target_ns': 1000, 'chunk_ns': 250, 'walltime_hours': 72},
        'scheduler': {'type': 'lsf', 'project': 'p', 'gpu_queue': 'gpu', 'gpu_resource': 'num=1/host', 'memory': '16GB', 'cores': 1},
    }
    path = tmp_path / 'source.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / 'workspace'
    chai = ws / '01_chai'
    chai.mkdir(parents=True)
    # Translate/rotate the heavy atoms relative to OFF. C0 has exactly three
    # heavy neighbors and one H, so H0 must use the tetrahedral repair rule.
    coords = [(10, 10, 10), (10, 11, 10), (9, 10, 10), (10, 10, 11)]
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(f'HETATM {1000+i} C{i}_1 LIG B   1       {x:7.3f} {y:7.3f} {z:7.3f}  1.00 70.00         B C  ')
    (chai / 'pred.model_idx_0.pdb').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (chai / '.done').write_text('{}\n', encoding='utf-8')

    glycam = ws / '02_prepare' / 'glycam'
    offdir = glycam / 'extracted' / 'structure'
    offdir.mkdir(parents=True)
    (offdir / 'structure.off').write_text(OFF_TEXT, encoding='utf-8')
    (glycam / '.done').write_text('{}\n', encoding='utf-8')

    mapping = ws / '02_prepare' / 'mapping'
    mapping.mkdir(parents=True)
    rows = [
        {'chai_serial': 1001+i, 'chai_atom': f'C{i+1}_1', 'smiles_index_0based': i, 'glycam_atom': f'C{i}', 'glycam_resname': 'LIG', 'glycam_resid': 1, 'element': 'C'}
        for i in range(4)
    ]
    with (mapping / 'atom_mapping.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (mapping / '.done').write_text('{}\n', encoding='utf-8')
    return ws


def test_coordinate_transfer_preserves_topology_and_repairs_tetrahedral_h(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    source = parse_off_unit(ws / '02_prepare/glycam/extracted/structure/structure.off', 'CONDENSEDSEQUENCE')
    result = transfer_glycan_coordinates(cfg, workspace=ws)
    aligned = parse_off_unit(result.aligned_off_path, 'CONDENSEDSEQUENCE')

    assert result.sentinel_path.is_file()
    assert [a['name'] for a in source.atoms] == [a['name'] for a in aligned.atoms]
    assert [a['type'] for a in source.atoms] == [a['type'] for a in aligned.atoms]
    assert [a['charge'] for a in source.atoms] == [a['charge'] for a in aligned.atoms]
    assert source.connectivity == aligned.connectivity

    by_name = {a['name']: a for a in aligned.atoms}
    assert np.allclose([by_name['C0']['x'], by_name['C0']['y'], by_name['C0']['z']], [10, 10, 10])
    # New heavy-neighbor unit vectors at C0 are +y, -x, +z. Missing direction is -normalize(sum).
    expected_direction = -np.array([-1.0, 1.0, 1.0]) / np.sqrt(3.0)
    old_bond = np.sqrt(3 * 0.577350**2)
    h0 = np.array([by_name['H0']['x'], by_name['H0']['y'], by_name['H0']['z']])
    assert np.allclose(h0, np.array([10, 10, 10]) + old_bond * expected_direction, atol=2e-6)

    with result.repair_path.open() as handle:
        repairs = list(csv.DictReader(handle, delimiter='\t'))
    assert len(repairs) == 1
    assert repairs[0]['hydrogen_atom'] == 'H0'

    validation = json.loads(result.validation_path.read_text())
    assert validation['counts']['heavy_atoms_transferred'] == 4
    assert validation['counts']['tetrahedral_ch_repairs'] == 1
    assert validation['counts']['ordinary_hydrogens_local_frame_moved'] == 1
    assert validation['checks']['passed'] is True


def test_coordinate_transfer_preserves_all_original_xh_bond_lengths(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    source = parse_off_unit(ws / '02_prepare/glycam/extracted/structure/structure.off', 'CONDENSEDSEQUENCE')
    result = transfer_glycan_coordinates(cfg, workspace=ws)
    aligned = parse_off_unit(result.aligned_off_path, 'CONDENSEDSEQUENCE')
    old = {int(a['index']): np.array([a['x'], a['y'], a['z']]) for a in source.atoms}
    new = {int(a['index']): np.array([a['x'], a['y'], a['z']]) for a in aligned.atoms}
    for edge in source.connectivity:
        a, b = int(edge['atom1']), int(edge['atom2'])
        aa, bb = source.atoms[a-1], source.atoms[b-1]
        if int(aa['atomic_number']) == 1 or int(bb['atomic_number']) == 1:
            assert np.linalg.norm(old[a] - old[b]) == pytest.approx(np.linalg.norm(new[a] - new[b]), abs=2e-6)


def test_coordinate_transfer_fails_without_complete_mapping(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    ws = _workspace(tmp_path)
    mapping = ws / '02_prepare/mapping/atom_mapping.tsv'
    lines = mapping.read_text().splitlines()
    mapping.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='all and only GLYCAM heavy atoms mapped'):
        transfer_glycan_coordinates(cfg, workspace=ws)
    assert not (ws / '02_prepare/coordinate_transfer/.done').exists()
