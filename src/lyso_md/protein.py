from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import PipelineConfig

DISULFIDE_CUTOFF_ANGSTROM = 2.4


@dataclass(frozen=True)
class ProteinPreparationResult:
    stage_dir: Path
    protein_pdb: Path
    disulfides_tsv: Path
    leap_bonds_path: Path
    validation_path: Path
    sentinel_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdb_element(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element and element[0].isalpha():
        return element.upper()
    name = line[12:16].strip()
    letters = "".join(ch for ch in name if ch.isalpha())
    return (letters[:1] or "?").upper()


def _pdb_record(line: str) -> tuple[int, str, str, str, str, str, str, float, float, float]:
    try:
        serial = int(line[6:11])
        atom_name = line[12:16].strip()
        altloc = line[16:17]
        resname = line[17:20].strip()
        chain = line[21:22]
        resid = line[22:26].strip()
        icode = line[26:27]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"malformed protein PDB ATOM record: {line!r}") from exc
    if not resid:
        raise ValueError(f"protein PDB atom has no residue number: {line!r}")
    if not atom_name:
        raise ValueError(f"protein PDB atom has no atom name: {line!r}")
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(f"non-finite protein coordinate at serial {serial}")
    return serial, atom_name, altloc, resname, chain, resid, icode, x, y, z


def _residue_key(line: str) -> tuple[str, str, str]:
    return (line[21:22], line[22:26].strip(), line[26:27])


def _residue_number(line: str) -> str:
    return line[22:26].strip()


def _is_hydrogen(line: str) -> bool:
    element = _pdb_element(line)
    if element == "H":
        return True
    name = line[12:16].strip().upper()
    return bool(name) and name[0] == "H"


def _replace_resname(line: str, new_name: str) -> str:
    return f"{line[:17]}{new_name:>3s}{line[20:]}"


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _format_leap_atom(resid: str, atom: str) -> str:
    return f"protein.{resid}.{atom}"


def prepare_protein(cfg: PipelineConfig, *, workspace: Path) -> ProteinPreparationResult:
    """Phase 6: extract a hydrogen-free protein PDB and detect disulfides."""
    workspace = Path(workspace).resolve()
    chai_done = workspace / "01_chai" / ".done"
    chai_pdb = workspace / "01_chai" / f"pred.model_idx_{cfg.chai.model_index}.pdb"
    if not chai_done.is_file():
        raise ValueError("Phase 6 requires a validated Phase 2 .done sentinel")
    if not chai_pdb.is_file():
        raise ValueError(f"Phase 6 requires Chai PDB: {chai_pdb}")

    stage = workspace / "02_prepare" / "protein"
    stage.mkdir(parents=True, exist_ok=True)
    protein_pdb = stage / "protein_chai.pdb"
    disulfides_tsv = stage / "disulfides.tsv"
    leap_bonds = stage / "disulfide_bonds.leap"
    validation_path = stage / "validation.json"
    sentinel = stage / ".done"
    if sentinel.exists():
        sentinel.unlink()

    source_lines = Path(chai_pdb).read_text(encoding="utf-8", errors="replace").splitlines()
    atom_lines: list[str] = []
    residues: dict[tuple[str, str, str], dict[str, Any]] = {}
    serials: set[int] = set()
    removed_hydrogens = 0

    for raw in source_lines:
        if not raw.startswith("ATOM  "):
            continue
        serial, atom_name, altloc, resname, chain, resid_num, _icode_num, x, y, z = _pdb_record(raw)
        if serial in serials:
            raise ValueError(f"duplicate protein atom serial: {serial}")
        serials.add(serial)
        if _is_hydrogen(raw):
            removed_hydrogens += 1
            continue
        key = _residue_key(raw)
        residue = residues.setdefault(
            key,
            {"chain": chain, "resid": _residue_number(raw), "icode": key[2], "resname": resname, "atoms": {}},
        )
        if residue["resname"] != resname:
            raise ValueError(f"residue {key} changes residue name within the Chai PDB")
        if atom_name in residue["atoms"]:
            raise ValueError(f"duplicate atom {atom_name} in protein residue {key}")
        residue["atoms"][atom_name] = (x, y, z)
        atom_lines.append(raw)

    expected = cfg.protein.expected_residues
    if len(residues) != expected:
        raise ValueError(f"protein residue count mismatch: found {len(residues)}, expected {expected}")
    if not atom_lines:
        raise ValueError("Chai PDB contains no protein ATOM records")

    # Residue numbers must uniquely identify LEaP bond endpoints. The target
    # system is a single protein chain; fail closed rather than guessing when
    # multiple chains reuse the same residue number.
    number_to_keys: dict[str, list[tuple[str, str, str]]] = {}
    for key, residue in residues.items():
        number_to_keys.setdefault(residue["resid"], []).append(key)
    duplicated_numbers = {n: keys for n, keys in number_to_keys.items() if len(keys) > 1}
    if duplicated_numbers:
        raise ValueError(f"protein residue numbers are not globally unique for LEaP bond commands: {duplicated_numbers}")

    cysteines = [
        (key, residue)
        for key, residue in residues.items()
        if residue["resname"] == "CYS"
    ]
    sg_missing = [key for key, residue in cysteines if "SG" not in residue["atoms"]]
    if sg_missing:
        raise ValueError(f"CYS residue(s) missing SG atom: {sg_missing}")

    candidates: list[dict[str, Any]] = []
    for i, (key_a, residue_a) in enumerate(cysteines):
        for key_b, residue_b in cysteines[i + 1 :]:
            distance = _distance(residue_a["atoms"]["SG"], residue_b["atoms"]["SG"])
            if distance <= DISULFIDE_CUTOFF_ANGSTROM:
                candidates.append(
                    {
                        "chain_a": residue_a["chain"],
                        "resid_a": residue_a["resid"],
                        "icode_a": residue_a["icode"],
                        "chain_b": residue_b["chain"],
                        "resid_b": residue_b["resid"],
                        "icode_b": residue_b["icode"],
                        "distance_angstrom": distance,
                        "key_a": key_a,
                        "key_b": key_b,
                    }
                )

    degree: dict[tuple[str, str, str], int] = {key: 0 for key, _ in cysteines}
    for candidate in candidates:
        degree[candidate["key_a"]] += 1
        degree[candidate["key_b"]] += 1
    ambiguous = [key for key, count in degree.items() if count > 1]
    if ambiguous:
        raise ValueError(
            "ambiguous disulfide detection: a CYS SG lies within the disulfide cutoff of multiple CYS residues: "
            f"{ambiguous}"
        )

    pairs = sorted(candidates, key=lambda row: (int(row["resid_a"]), int(row["resid_b"])))
    participating = {candidate["key_a"] for candidate in pairs} | {candidate["key_b"] for candidate in pairs}

    output_lines: list[str] = []
    for raw in source_lines:
        if not raw.startswith("ATOM  "):
            continue
        if _is_hydrogen(raw):
            continue
        key = _residue_key(raw)
        if key in participating and raw[17:20].strip() == "CYS":
            raw = _replace_resname(raw, "CYX")
        output_lines.append(raw.rstrip())
    if not output_lines:
        raise ValueError("protein preparation produced no ATOM records")
    protein_pdb.write_text("\n".join(output_lines) + "\nTER\nEND\n", encoding="utf-8")

    fieldnames = [
        "chain_a", "resid_a", "icode_a", "chain_b", "resid_b", "icode_b", "sg_distance_angstrom",
    ]
    with disulfides_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow({k: pair[k] for k in fieldnames[:-1]} | {"sg_distance_angstrom": f"{pair['distance_angstrom']:.4f}"})

    leap_bonds.write_text(
        "".join(f"bond {_format_leap_atom(pair['resid_a'], 'SG')} {_format_leap_atom(pair['resid_b'], 'SG')}\n" for pair in pairs),
        encoding="utf-8",
    )

    validation = {
        "stage": "protein",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "source": {"pdb": str(chai_pdb), "sha256": _sha256(chai_pdb)},
        "outputs": {
            "protein_pdb": str(protein_pdb),
            "protein_pdb_sha256": _sha256(protein_pdb),
            "disulfides_tsv": str(disulfides_tsv),
            "leap_bonds": str(leap_bonds),
        },
        "counts": {
            "protein_residues": len(residues),
            "expected_protein_residues": expected,
            "heavy_atom_records": len(atom_lines),
            "hydrogen_records_removed": removed_hydrogens,
            "cysteine_residues": len(cysteines),
            "disulfide_candidates": len(candidates),
            "disulfide_pairs": len(pairs),
        },
        "disulfides": [
            {
                **{k: pair[k] for k in fieldnames[:-1]},
                "sg_distance_angstrom": pair["distance_angstrom"],
            }
            for pair in pairs
        ],
        "checks": {
            "protein_residue_count_ok": len(residues) == expected,
            "all_cys_have_sg": not sg_missing,
            "no_ambiguous_disulfides": not ambiguous,
            "finite_coordinates": True,
            "hydrogens_removed": True,
            "passed": True,
        },
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sentinel.write_text(json.dumps({
        "stage": "protein",
        "status": "complete",
        "completed_at": validation["completed_at"],
        "pipeline_version": __version__,
        "validation": str(validation_path),
        "outputs": [str(protein_pdb), str(disulfides_tsv), str(leap_bonds)],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ProteinPreparationResult(stage, protein_pdb, disulfides_tsv, leap_bonds, validation_path, sentinel)
