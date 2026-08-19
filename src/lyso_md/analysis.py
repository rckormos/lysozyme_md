from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import __version__
from .config import PipelineConfig

PROTEIN_MASK = ":1-130"
GLYCAN_MASK = ":131-135"
SOLVENT_IONS_MASK = ":WAT,K+,Cl-"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_chunks(workspace: Path) -> list[Path]:
    root = workspace / "07_production"
    chunks: list[Path] = []
    number = 1
    while True:
        stage = root / f"chunk_{number:03d}"
        done = stage / ".done"
        validation = stage / "validation.json"
        trajectory = stage / "production.nc"
        if not done.is_file():
            break
        if not validation.is_file():
            raise ValueError(f"production chunk {number} has .done but no validation.json")
        payload = json.loads(validation.read_text(encoding="utf-8"))
        if payload.get("status") != "done" or not payload.get("checks", {}).get("passed", False):
            raise ValueError(f"production chunk {number} is not backed by a passing validation")
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            raise ValueError(f"production chunk {number} has no usable production.nc")
        chunks.append(stage)
        number += 1
    if not chunks:
        raise ValueError("Phase 16 requires at least one completed production chunk")
    return chunks


def _find_topology(workspace: Path, chunks: Iterable[Path]) -> Path:
    for chunk in chunks:
        candidate = chunk / "complex_solvated.parm7"
        if candidate.is_file():
            return candidate.resolve()
    candidate = workspace / "04_solvate" / "complex_solvated.parm7"
    if candidate.is_file():
        return candidate.resolve()
    raise ValueError("cannot locate the completed solvated topology complex_solvated.parm7")


def _cpptraj_header(topology: Path, trajectory: Path) -> str:
    return f"parm {topology}\ntrajin {trajectory}\n"


def _render_preprocess(topology: Path, chunks: list[Path], output: Path) -> str:
    lines = [f"parm {topology}"]
    for chunk in chunks:
        lines.append(f"trajin {chunk / 'production.nc'}")
    lines.extend([
        "autoimage",
        f"rms first {PROTEIN_MASK}@N,CA,C",
        f"strip {SOLVENT_IONS_MASK}",
        f"trajout {output} netcdf nobox",
        "run",
        "quit",
        "",
    ])
    return "\n".join(lines)


def _analysis_inputs(processed_topology: Path, processed_trajectory: Path, root: Path) -> dict[str, str]:
    header = _cpptraj_header(processed_topology, processed_trajectory)
    return {
        "rmsd.in": header + "rms first :1-130@CA out rmsd_protein_ca.dat\n"
        + "rms first :131-135 out rmsd_glycan.dat\nrun\nquit\n",
        "rmsf.in": header + "atomicfluct :1-130@CA byres out rmsf_ca.dat\nrun\nquit\n",
        "rg.in": header + f"radgyr {PROTEIN_MASK} out rg_protein.dat\nradgyr {GLYCAN_MASK} out rg_glycan.dat\nrun\nquit\n",
        "dssp.in": header + "secstruct :1-130 out dssp.dat\nrun\nquit\n",
        "hbond_protein_to_glycan.in": header
        + f"hbond HB_P2G out hbond_protein_to_glycan.dat donormask {PROTEIN_MASK} acceptormask {GLYCAN_MASK}\nrun\nquit\n",
        "hbond_glycan_to_protein.in": header
        + f"hbond HB_G2P out hbond_glycan_to_protein.dat donormask {GLYCAN_MASK} acceptormask {PROTEIN_MASK}\nrun\nquit\n",
        "contacts.in": header
        + f"nativecontacts {PROTEIN_MASK} {GLYCAN_MASK} distance 4.0 skipnative out protein_glycan_contacts.dat\nrun\nquit\n",
        "pca.in": header
        + "rms first @CA\n"
        + "matrix covar name covar @CA\n"
        + "diagmatrix covar out ca_modes.dat vecs 20 name ca_modes\n"
        + "projection modes ca_modes.dat beg 1 end 3 @CA out ca_projection.dat\n"
        + "run\nquit\n",
        "pca_modes.in": header
        + "readdata ca_modes.dat name ca_modes\n"
        + "runanalysis modes name ca_modes trajout mode1.nc trajoutmask @CA pcmin -10 pcmax 10 tmode 1\n"
        + "runanalysis modes name ca_modes trajout mode2.nc trajoutmask @CA pcmin -10 pcmax 10 tmode 2\n"
        + "runanalysis modes name ca_modes trajout mode3.nc trajoutmask @CA pcmin -10 pcmax 10 tmode 3\n"
        + "quit\n",
        "dccm.in": header + "rms first @CA\nmatrix correl name dccm @CA out dccm.dat\nrun\nquit\n",
        "clustering.in": header
        + "cluster hieragglo epsilon 2.0 rms @CA sieve 10 out cluster.dat summary cluster_summary.dat info cluster_info.dat\nrun\nquit\n",
        "average_structure.in": header + "average average_structure.pdb pdb\nrun\nquit\n",
        "pairwise_rmsd.in": header + "rms @CA pairwise out pairwise_rmsd.dat\nrun\nquit\n",
        "distances.in": header + "distance protein_glycan_ca_distance :65@CA :131@C1 out distances.dat\nrun\nquit\n",
        "angles.in": header + "angle protein_glycan_angle :64@CA :65@CA :131@C1 out angles.dat\nrun\nquit\n",
    }


def _render_pairwise_preprocess(topology: Path, processed_trajectory: Path, output: Path) -> str:
    return "\n".join([
        f"parm {topology}",
        f"trajin {processed_trajectory} 1 last 10",
        f"trajout {output} netcdf nobox",
        "run",
        "quit",
        "",
    ])


def _write_parmed_stripped(topology: Path, output: Path) -> None:
    try:
        import parmed as pmd
    except ImportError as exc:
        raise RuntimeError("ParmEd is required for Phase 16 stripped-topology generation") from exc
    structure = pmd.load_file(str(topology))
    structure.strip(SOLVENT_IONS_MASK)
    structure.save(str(output), overwrite=True)


def _run_cpptraj(input_path: Path, *, cwd: Path) -> None:
    executable = shutil.which("cpptraj")
    if executable is None:
        raise RuntimeError("cpptraj was not found on PATH; load Amber 22 before running Phase 16")
    proc = subprocess.run([executable, "-i", str(input_path)], cwd=cwd, text=True, capture_output=True)
    log = cwd / f"{input_path.stem}.cpptraj.log"
    log.write_text((proc.stdout or "") + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"cpptraj failed for {input_path.name} with exit code {proc.returncode}; see {log}")


@dataclass(frozen=True)
class AnalysisResult:
    stage: Path
    dry_run: bool
    processed_trajectory: Path
    processed_topology: Path


def analyze(cfg: PipelineConfig, *, workspace: Path, dry_run: bool = False) -> AnalysisResult:
    workspace = Path(workspace).resolve()
    production_done = workspace / "07_production" / ".done"
    if not production_done.is_file():
        raise ValueError("Phase 16 requires a completed 1-target production checkpoint (07_production/.done)")
    chunks = _completed_chunks(workspace)
    topology = _find_topology(workspace, chunks)
    stage = workspace / "07_analysis"
    stage.mkdir(parents=True, exist_ok=True)
    processed_trajectory = stage / "processed.nc"
    processed_topology = stage / "processed.parm7"

    preprocess = stage / "preprocess.in"
    preprocess.write_text(_render_preprocess(topology, chunks, processed_trajectory), encoding="utf-8")

    inputs = _analysis_inputs(processed_topology, processed_trajectory, stage)
    for name, text in inputs.items():
        (stage / name).write_text(text, encoding="utf-8")

    pairwise_input = stage / "pairwise_preprocess.in"
    pairwise_trajectory = stage / "pairwise_subsampled.nc"
    pairwise_input.write_text(_render_pairwise_preprocess(processed_topology, processed_trajectory, pairwise_trajectory), encoding="utf-8")

    manifest = {
        "stage": "analysis",
        "pipeline_version": __version__,
        "created_at": _utc_now(),
        "production_chunks": [str(chunk) for chunk in chunks],
        "topology": str(topology),
        "processed_trajectory": str(processed_trajectory),
        "processed_topology": str(processed_topology),
        "inputs": sorted([preprocess.name, pairwise_input.name, *inputs.keys()]),
        "protein_residues": "1-130",
        "glycan_residues": "131-135",
        "pca": {
            "rms": "@CA",
            "covariance": "matrix covar name covar @CA",
            "eigenvectors": 20,
            "projection": "PC1-PC3",
            "mode_mask": "@CA",
        },
    }
    (stage / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if dry_run:
        return AnalysisResult(stage, True, processed_trajectory, processed_topology)

    _run_cpptraj(preprocess, cwd=stage)
    if not processed_trajectory.is_file() or processed_trajectory.stat().st_size == 0:
        raise RuntimeError("CPPTRAJ preprocessing did not create processed.nc")
    _write_parmed_stripped(topology, processed_topology)
    if not processed_topology.is_file() or processed_topology.stat().st_size == 0:
        raise RuntimeError("ParmEd did not create processed.parm7")
    _run_cpptraj(pairwise_input, cwd=stage)
    for name in inputs:
        _run_cpptraj(stage / name, cwd=stage)

    required = [
        processed_trajectory,
        processed_topology,
        stage / "rmsd_protein_ca.dat",
        stage / "rmsd_glycan.dat",
        stage / "rmsf_ca.dat",
        stage / "rg_protein.dat",
        stage / "rg_glycan.dat",
        stage / "dssp.dat",
        stage / "hbond_protein_to_glycan.dat",
        stage / "hbond_glycan_to_protein.dat",
        stage / "protein_glycan_contacts.dat",
        stage / "ca_modes.dat",
        stage / "ca_projection.dat",
        stage / "mode1.nc",
        stage / "mode2.nc",
        stage / "mode3.nc",
        stage / "dccm.dat",
        stage / "cluster.dat",
        stage / "cluster_summary.dat",
        stage / "average_structure.pdb",
        stage / "pairwise_subsampled.nc",
        stage / "pairwise_rmsd.dat",
        stage / "distances.dat",
        stage / "angles.dat",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Phase 16 analysis did not produce required output(s): " + ", ".join(missing))

    validation = {
        "stage": "analysis",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "checks": {"production_complete": True, "processed_trajectory_exists": True, "processed_topology_exists": True, "required_outputs_exist": True, "passed": True},
        "inputs": {"topology": str(topology), "production_chunks": [str(chunk) for chunk in chunks]},
        "outputs": {"processed_trajectory": str(processed_trajectory), "processed_topology": str(processed_topology)},
        "sha256": {path.name: _sha256(path) for path in required},
    }
    (stage / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / ".done").write_text(json.dumps({"stage": "analysis", "status": "done", "pipeline_version": __version__, "completed_at": validation["completed_at"], "validation": str(stage / "validation.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AnalysisResult(stage, False, processed_trajectory, processed_topology)
