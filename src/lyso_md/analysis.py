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
        + "run\nquit\n",
        "pca_projection.in": header
        + "readdata ca_modes.dat name ca_modes\n"
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


def _checkpoint_path(stage: Path, name: str) -> Path:
    return stage / f"{name}.done"


def _checkpoint_write(stage: Path, name: str, outputs: list[Path]) -> None:
    payload = {
        "stage": f"analysis_{name}",
        "status": "done",
        "pipeline_version": __version__,
        "completed_at": _utc_now(),
        "outputs": [str(path) for path in outputs],
        "sha256": {path.name: _sha256(path) for path in outputs},
    }
    _checkpoint_path(stage, name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checkpoint_valid(stage: Path, name: str, outputs: list[Path]) -> bool:
    checkpoint = _checkpoint_path(stage, name)
    if not checkpoint.is_file():
        return False
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "done":
            return False
        for path in outputs:
            if not path.is_file() or path.stat().st_size == 0:
                return False
            recorded = payload.get("sha256", {}).get(path.name)
            if recorded and recorded != _sha256(path):
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _outputs_exist(outputs: list[Path]) -> bool:
    return bool(outputs) and all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def _run_cpptraj(input_path: Path, *, cwd: Path, outputs: list[Path] | None = None) -> None:
    executable = shutil.which("cpptraj")
    if executable is None:
        raise RuntimeError("cpptraj was not found on PATH; load Amber 22 before running Phase 16")
    output_paths = outputs or []
    tmp_paths: list[tuple[Path, Path]] = []
    rendered = input_path.read_text(encoding="utf-8")
    if output_paths:
        replacements: list[tuple[Path, Path]] = []
        for output in output_paths:
            tmp = output.with_name(output.name + ".tmp")
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            replacements.append((output, tmp))
            tmp_paths.append((output, tmp))
        for final, tmp in replacements:
            rendered = rendered.replace(str(final), str(tmp))
            rendered = rendered.replace(final.name, tmp.name)
        input_path = input_path.with_name(input_path.stem + ".tmp.in")
        input_path.write_text(rendered, encoding="utf-8")
    proc = subprocess.run([executable, "-i", str(input_path)], cwd=cwd, text=True, capture_output=True)
    log = cwd / f"{input_path.stem.replace('.tmp', '')}.cpptraj.log"
    log.write_text((proc.stdout or "") + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        for _, tmp in tmp_paths:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        if input_path.name.endswith(".tmp.in"):
            input_path.unlink(missing_ok=True)
        raise RuntimeError(f"cpptraj failed for {input_path.name} with exit code {proc.returncode}; see {log}")
    for final, tmp in tmp_paths:
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"CPPTRAJ did not create usable output: {final}")
        tmp.replace(final)
    if input_path.name.endswith(".tmp.in"):
        input_path.unlink(missing_ok=True)


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

    preprocess_outputs = [processed_trajectory]
    topology_outputs = [processed_topology]
    if not _checkpoint_valid(stage, "preprocess", preprocess_outputs):
        if not _outputs_exist(preprocess_outputs):
            _run_cpptraj(preprocess, cwd=stage, outputs=preprocess_outputs)
        _write_parmed_stripped(topology, processed_topology)
        if not _outputs_exist(topology_outputs):
            raise RuntimeError("ParmEd did not create processed.parm7")
        _checkpoint_write(stage, "preprocess", [processed_trajectory, processed_topology])
    elif not _outputs_exist(topology_outputs):
        _write_parmed_stripped(topology, processed_topology)
        _checkpoint_write(stage, "preprocess", [processed_trajectory, processed_topology])

    pairwise_trajectory = stage / "pairwise_subsampled.nc"
    if not _checkpoint_valid(stage, "pairwise_preprocess", [pairwise_trajectory]):
        if not _outputs_exist([pairwise_trajectory]):
            _run_cpptraj(pairwise_input, cwd=stage, outputs=[pairwise_trajectory])
        _checkpoint_write(stage, "pairwise_preprocess", [pairwise_trajectory])

    analysis_outputs = {
        "rmsd": [stage / "rmsd_protein_ca.dat", stage / "rmsd_glycan.dat"],
        "rmsf": [stage / "rmsf_ca.dat"],
        "rg": [stage / "rg_protein.dat", stage / "rg_glycan.dat"],
        "dssp": [stage / "dssp.dat"],
        "hbond_protein_to_glycan": [stage / "hbond_protein_to_glycan.dat"],
        "hbond_glycan_to_protein": [stage / "hbond_glycan_to_protein.dat"],
        "contacts": [stage / "protein_glycan_contacts.dat"],
        "pca_covariance": [stage / "ca_modes.dat"],
        "pca_projection": [stage / "ca_projection.dat"],
        "pca_modes": [stage / "mode1.nc", stage / "mode2.nc", stage / "mode3.nc"],
        "dccm": [stage / "dccm.dat"],
        "clustering": [stage / "cluster.dat", stage / "cluster_summary.dat"],
        "average_structure": [stage / "average_structure.pdb"],
        "pairwise_rmsd": [stage / "pairwise_rmsd.dat"],
        "distances": [stage / "distances.dat"],
        "angles": [stage / "angles.dat"],
    }
    for name, outputs in analysis_outputs.items():
        if _checkpoint_valid(stage, name, outputs):
            continue
        if _outputs_exist(outputs):
            _checkpoint_write(stage, name, outputs)
            continue
        _run_cpptraj(stage / next(k for k in inputs if k.startswith(name + ".")), cwd=stage, outputs=outputs)
        _checkpoint_write(stage, name, outputs)

    required = [processed_trajectory, processed_topology, pairwise_trajectory]
    for outputs in analysis_outputs.values():
        required.extend(outputs)
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
