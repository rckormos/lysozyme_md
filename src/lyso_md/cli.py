from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from .config import load_config
from .chai import run_chai, submit_chai
from .glycam import inspect_glycam_bundle
from .mapping import map_chai_to_glycam
from .protein import prepare_protein
from .structure import transfer_glycan_coordinates
from .leap import assemble_dry_complex
from .amber import relax_hydrogens
from .solvate import solvate_and_ionize
from .minimize import prepare_minimization, run_minimization_worker
from .heating import prepare_heating, run_heating_worker
from .npt import prepare_npt_smoke, run_npt_smoke_worker
from .equilibration import prepare_npt_equilibration, run_npt_equilibration_worker
from .production import prepare_production, run_production_worker, reconcile_production_checkpoint
from .orchestration import format_status
from .analysis import analyze as run_analysis
from .qc import write_qc_report
from .logging_utils import configure_logging
from .workspace import initialize_workspace

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Restartable FASTA-to-MD pipeline for lysozyme designs.")


def _fail(exc: Exception) -> None:
    typer.echo(f"ERROR: {exc}", err=True)
    raise typer.Exit(code=2)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-lines logs."),
) -> None:
    configure_logging(verbose=verbose, json_logs=json_logs)


@app.command("validate-config")
def validate_config(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Validate schema, scientific bounds, and required input-file existence."""
    try:
        cfg = load_config(config, check_files=True)
    except (ValueError, ValidationError, OSError) as exc:
        _fail(exc)
    typer.echo(f"Configuration valid: {cfg.name}")


@app.command()
def init(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    workspace_root: Optional[Path] = typer.Option(
        None,
        "--workspace-root",
        help="Parent directory for the design workspace. Defaults to the source config's directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Preserve an existing workspace as a timestamped backup, then initialize a fresh workspace.",
    ),
) -> None:
    """Initialize a per-design workspace outside the source repository."""
    try:
        cfg = load_config(config, check_files=True)
        workspace, backup = initialize_workspace(
            cfg,
            source_config=config,
            workspace_root=workspace_root,
            force=force,
        )
    except (ValueError, ValidationError, OSError) as exc:
        _fail(exc)
    if backup:
        typer.echo(f"Preserved previous workspace: {backup}")
    typer.echo(f"Initialized workspace: {workspace}")


def _stub(name: str, config: Path, from_stage: Optional[str], through: Optional[str], dry_run: bool) -> None:
    try:
        cfg = load_config(config, check_files=True)
    except (ValueError, ValidationError, OSError) as exc:
        _fail(exc)
    payload = {
        "command": name,
        "design": cfg.name,
        "from": from_stage,
        "through": through,
        "dry_run": dry_run,
        "implemented": False,
    }
    typer.echo(json.dumps(payload, sort_keys=True))
    typer.echo(f"{name} is a Phase 0 stub; execution is implemented in later phases.", err=True)


def _validate_prepare_stage_bounds(from_stage: Optional[str], through: Optional[str]) -> None:
    allowed = (None, "chai", "glycam", "mapping", "coordinates", "protein", "leap", "hydrogen-relax", "solvate", "minimize", "heat", "npt-smoke", "npt-equilibrate", "production")
    if from_stage not in allowed:
        raise ValueError("implemented prepare stages are: chai, glycam, mapping, coordinates, protein, leap, hydrogen-relax, solvate, minimize, heat, npt-smoke, npt-equilibrate, production")
    if through not in allowed:
        raise ValueError("implemented prepare stages are: chai, glycam, mapping, coordinates, protein, leap, hydrogen-relax, solvate, minimize, heat, npt-smoke, npt-equilibrate, production")
    order = {"chai": 1, "glycam": 2, "mapping": 3, "coordinates": 4, "protein": 5, "leap": 6, "hydrogen-relax": 7, "solvate": 8, "minimize": 9, "heat": 10, "npt-smoke": 11, "npt-equilibrate": 12, "production": 13}
    if from_stage is not None and through is not None and order[from_stage] > order[through]:
        raise ValueError("--from stage must not come after --through stage")


def _validate_submit_stage_bounds(from_stage: Optional[str], through: Optional[str]) -> None:
    if from_stage not in (None, "chai") or through not in (None, "chai"):
        raise ValueError("Phase 3 submit currently supports the LSF Chai stage only; GLYCAM inspection is local")


def _phase2_workspace(config: Path) -> Path:
    workspace = config.parent.resolve()
    if not (workspace / "manifest.json").is_file():
        raise ValueError("Phase 2 must be run with the initialized workspace config.yaml")
    return workspace


@app.command()
def prepare(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    through: Optional[str] = typer.Option(None, "--through"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    local: bool = typer.Option(False, "--local", help="Run Chai directly instead of submitting through LSF."),
) -> None:
    """Prepare implemented stages through glycan coordinate transfer and hydrogen repair."""
    try:
        cfg = load_config(config, check_files=True)
        _validate_prepare_stage_bounds(from_stage, through)
        workspace = _phase2_workspace(config)

        start_stage = from_stage or "chai"
        end_stage = through or "coordinates"

        if start_stage == "chai":
            chai_done = workspace / "01_chai" / ".done"
            if not chai_done.is_file():
                if not cfg.chai.enabled:
                    typer.echo("Chai stage disabled by configuration")
                elif local:
                    result = run_chai(cfg, workspace=workspace, dry_run=dry_run)
                    if result.dry_run:
                        typer.echo(f"Prepared local Chai dry run: {result.stage_dir}")
                    else:
                        typer.echo(f"Chai stage complete: {result.selected_pdb}")
                else:
                    submission = submit_chai(cfg, workspace=workspace, dry_run=dry_run)
                    if submission.dry_run:
                        typer.echo(f"Prepared Chai LSF dry run: {submission.script_path}")
                    else:
                        typer.echo(f"Chai stage submitted: job {submission.job_id}")
                        typer.echo(f"Submission metadata: {submission.submission_path}")
                    return
            elif end_stage == "chai":
                typer.echo(f"Chai stage already complete: {chai_done}")
                return

        stage_order = {"chai": 1, "glycam": 2, "mapping": 3, "coordinates": 4, "protein": 5, "leap": 6, "hydrogen-relax": 7, "solvate": 8, "minimize": 9, "heat": 10, "npt-smoke": 11, "npt-equilibrate": 12, "production": 13}
        start_num = stage_order[start_stage]
        end_num = stage_order[end_stage]

        if start_num <= stage_order["glycam"] <= end_num:
            glycam_done = workspace / "02_prepare" / "glycam" / ".done"
            if not glycam_done.is_file():
                if dry_run:
                    typer.echo("GLYCAM inspection is a local deterministic stage; --dry-run does not execute it.")
                    return
                result = inspect_glycam_bundle(cfg, workspace=workspace)
                typer.echo(f"GLYCAM inspection complete: {result.summary_path}")
            elif end_stage == "glycam":
                typer.echo(f"GLYCAM inspection already complete: {glycam_done}")
                return

        if start_num <= stage_order["mapping"] <= end_num:
            mapping_done = workspace / "02_prepare" / "mapping" / ".done"
            if not mapping_done.is_file():
                if dry_run:
                    typer.echo("Chai-to-GLYCAM mapping is a local deterministic stage; --dry-run does not execute it.")
                    return
                result = map_chai_to_glycam(cfg, workspace=workspace)
                typer.echo(f"Chai-to-GLYCAM mapping complete: {result.mapping_path}")
            elif end_stage == "mapping":
                typer.echo(f"Chai-to-GLYCAM mapping already complete: {mapping_done}")
                return

        if start_num <= stage_order["coordinates"] <= end_num:
            coordinate_done = workspace / "02_prepare" / "coordinate_transfer" / ".done"
            if not coordinate_done.is_file():
                if dry_run:
                    typer.echo("Coordinate transfer is a local deterministic stage; --dry-run does not execute it.")
                    return
                result = transfer_glycan_coordinates(cfg, workspace=workspace)
                typer.echo(f"Glycan coordinate transfer complete: {result.aligned_off_path}")
            elif end_stage == "coordinates":
                typer.echo(f"Coordinate transfer already complete: {coordinate_done}")
                return

        if start_num <= stage_order["protein"] <= end_num:
            protein_done = workspace / "02_prepare" / "protein" / ".done"
            if not protein_done.is_file():
                if dry_run:
                    typer.echo("Protein preparation is a local deterministic stage; --dry-run does not execute it.")
                    return
                result = prepare_protein(cfg, workspace=workspace)
                typer.echo(f"Protein preparation complete: {result.protein_pdb}")
            elif end_stage == "protein":
                typer.echo(f"Protein preparation already complete: {protein_done}")
                return

        if start_num <= stage_order["leap"] <= end_num:
            leap_done = workspace / "03_dry_relax" / ".done"
            if not leap_done.is_file():
                if dry_run:
                    result = assemble_dry_complex(cfg, workspace=workspace, dry_run=True)
                    typer.echo(f"Prepared dry LEaP input: {result.input_path}")
                    return
                result = assemble_dry_complex(cfg, workspace=workspace, dry_run=False)
                typer.echo(f"Dry LEaP assembly complete: {result.complex_pdb}")
            elif end_stage == "leap":
                typer.echo(f"Dry LEaP assembly already complete: {leap_done}")
                return

        if start_num <= stage_order["hydrogen-relax"] <= end_num:
            relax_done = workspace / "03_dry_relax" / "hydrogen_relax" / ".done"
            if not relax_done.is_file():
                result = relax_hydrogens(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared CPU pmemd hydrogen-relaxation input: {result.input_path}")
                    return
                typer.echo(f"Hydrogen relaxation complete: {result.output_path}")
            elif end_stage == "hydrogen-relax":
                typer.echo(f"Hydrogen relaxation already complete: {relax_done}")
                return

        if end_stage == "solvate":
            solvate_done = workspace / "04_solvate" / ".done"
            if solvate_done.is_file():
                typer.echo(f"Solvation/ionization already complete: {solvate_done}")
            else:
                result = solvate_and_ionize(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared Phase 9 LEaP geometry probe: {result.stage / 'solvate_probe.in'}")
                else:
                    typer.echo(f"Solvation/ionization complete: {result.rst7}")
                if end_stage == "solvate":
                    return
            if end_stage == "solvate":
                return

        if start_num <= stage_order["minimize"] <= end_num:
            minimize_done = workspace / "05_minimize" / ".done"
            if not minimize_done.is_file():
                result = prepare_minimization(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared Phase 10 LSF minimization scripts: {result.stage}")
                else:
                    typer.echo(f"Phase 10 minimization submitted: solvent job {result.solvent_job_id}, all-system job {result.all_job_id}")
                    typer.echo(f"Submission metadata: {result.submission_path}")
            else:
                typer.echo(f"Periodic minimization already complete: {minimize_done}")
            if end_stage == "minimize":
                return

        if start_num <= stage_order["heat"] <= end_num:
            heat_done = workspace / "06_equilibrate" / "heat" / ".done"
            if not heat_done.is_file():
                result = prepare_heating(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared Phase 11 LSF heating script: {result.script_path}")
                else:
                    typer.echo(f"Phase 11 heating submitted: job {result.job_id}")
                    typer.echo(f"Submission metadata: {result.submission_path}")
            else:
                typer.echo(f"Heating already complete: {heat_done}")
            return

        if start_num <= stage_order["npt-smoke"] <= end_num:
            smoke_done = workspace / "06_equilibrate" / "npt_smoke" / ".done"
            if not smoke_done.is_file():
                result = prepare_npt_smoke(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared Phase 12 LSF NPT smoke-test script: {result.script_path}")
                else:
                    typer.echo(f"Phase 12 NPT smoke test submitted: job {result.job_id}")
                    typer.echo(f"Submission metadata: {result.submission_path}")
                if end_stage == "npt-smoke":
                    return
            elif end_stage == "npt-smoke":
                typer.echo(f"NPT smoke test already complete: {smoke_done}")
                return

        if start_num <= stage_order["npt-equilibrate"] <= end_num:
            equil_done = workspace / "06_equilibrate" / "npt_equilibrate" / ".done"
            if not equil_done.is_file():
                result = prepare_npt_equilibration(cfg, workspace=workspace, dry_run=dry_run)
                if result.dry_run:
                    typer.echo(f"Prepared Phase 13 LSF NPT equilibration scripts: {result.stage}")
                else:
                    typer.echo(f"Phase 13 NPT equilibration submitted: jobs {', '.join(j for j in result.job_ids if j)}")
                    typer.echo(f"Submission metadata: {result.submission_path}")
            else:
                typer.echo(f"NPT equilibration already complete: {equil_done}")

        if start_num <= stage_order["production"] <= end_num:
            production_done = workspace / "07_production" / ".done"
            result = prepare_production(cfg, workspace=workspace, dry_run=dry_run)
            if result.completed:
                typer.echo(f"Production target already complete: {production_done}")
            elif result.dry_run:
                typer.echo(f"Prepared Phase 14 production chunk {result.chunk_number}: {result.script_path}")
            else:
                typer.echo(f"Phase 14 production chunk {result.chunk_number} submitted: job {result.job_id}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)


@app.command()
def submit(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    through: Optional[str] = typer.Option(None, "--through"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Submit implemented LSF stages; Phase 3 GLYCAM inspection runs locally via prepare."""
    try:
        cfg = load_config(config, check_files=True)
        if from_stage == "minimize" or through == "minimize":
            if from_stage != "minimize" or through != "minimize":
                raise ValueError("Phase 10 submit requires --from minimize --through minimize")
            workspace = _phase2_workspace(config)
            result = prepare_minimization(cfg, workspace=workspace, dry_run=dry_run)
            if result.dry_run:
                typer.echo(f"Prepared Phase 10 LSF minimization scripts: {result.stage}")
            else:
                typer.echo(f"Phase 10 minimization submitted: solvent job {result.solvent_job_id}, all-system job {result.all_job_id}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
        if from_stage == "heat" or through == "heat":
            if from_stage != "heat" or through != "heat":
                raise ValueError("Phase 11 submit requires --from heat --through heat")
            workspace = _phase2_workspace(config)
            result = prepare_heating(cfg, workspace=workspace, dry_run=dry_run)
            if result.dry_run:
                typer.echo(f"Prepared Phase 11 LSF heating script: {result.script_path}")
            else:
                typer.echo(f"Phase 11 heating submitted: job {result.job_id}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
        if from_stage == "npt-smoke" or through == "npt-smoke":
            if from_stage != "npt-smoke" or through != "npt-smoke":
                raise ValueError("Phase 12 submit requires --from npt-smoke --through npt-smoke")
            workspace = _phase2_workspace(config)
            result = prepare_npt_smoke(cfg, workspace=workspace, dry_run=dry_run)
            if result.dry_run:
                typer.echo(f"Prepared Phase 12 LSF NPT smoke-test script: {result.script_path}")
            else:
                typer.echo(f"Phase 12 NPT smoke test submitted: job {result.job_id}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
        if from_stage == "npt-equilibrate" or through == "npt-equilibrate":
            if from_stage != "npt-equilibrate" or through != "npt-equilibrate":
                raise ValueError("Phase 13 submit requires --from npt-equilibrate --through npt-equilibrate")
            workspace = _phase2_workspace(config)
            result = prepare_npt_equilibration(cfg, workspace=workspace, dry_run=dry_run)
            if result.dry_run:
                typer.echo(f"Prepared Phase 13 LSF NPT equilibration scripts: {result.stage}")
            else:
                typer.echo(f"Phase 13 NPT equilibration submitted: jobs {', '.join(j for j in result.job_ids if j)}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
        if from_stage == "production" or through == "production":
            if from_stage != "production" or through != "production":
                raise ValueError("Phase 14 submit requires --from production --through production")
            workspace = _phase2_workspace(config)
            result = prepare_production(cfg, workspace=workspace, dry_run=dry_run)
            if result.completed:
                typer.echo("Phase 14 production target already complete")
            elif result.dry_run:
                typer.echo(f"Prepared Phase 14 production chunk {result.chunk_number}: {result.script_path}")
            else:
                typer.echo(f"Phase 14 production chunk {result.chunk_number} submitted: job {result.job_id}")
                typer.echo(f"Submission metadata: {result.submission_path}")
            return
        _validate_submit_stage_bounds(from_stage, through)
        workspace = _phase2_workspace(config)
        if not cfg.chai.enabled:
            typer.echo("Chai stage disabled by configuration")
            return
        submission = submit_chai(cfg, workspace=workspace, dry_run=dry_run)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    if submission.dry_run:
        typer.echo(f"Prepared Chai LSF dry run: {submission.script_path}")
    else:
        typer.echo(f"Chai stage submitted: job {submission.job_id}")
        typer.echo(f"Submission metadata: {submission.submission_path}")


@app.command("_minimize-worker", hidden=True)
def minimize_worker(
    worker: str = typer.Argument(...),
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker for Phase 10 GPU minimization."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        run_minimization_worker(cfg, workspace=workspace, worker=worker)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(f"Phase 10 {worker} minimization complete")


@app.command("_heat-worker", hidden=True)
def heat_worker(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker for Phase 11 GPU NVT heating."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        run_heating_worker(cfg, workspace=workspace)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo("Phase 11 heating complete")

@app.command("_npt-smoke-worker", hidden=True)
def npt_smoke_worker(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker for Phase 12 GPU NPT smoke test."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        run_npt_smoke_worker(cfg, workspace=workspace)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo("Phase 12 NPT smoke test complete")


@app.command("_npt-equilibrate-worker", hidden=True)
def npt_equilibrate_worker(
    stage: str = typer.Argument(...),
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker for Phase 13 staged GPU NPT equilibration."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        run_npt_equilibration_worker(cfg, workspace=workspace, stage_name=stage)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(f"Phase 13 {stage} NPT equilibration complete")


@app.command("_production-worker", hidden=True)
def production_worker(
    chunk: int = typer.Argument(..., min=1),
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker for Phase 14 chunked GPU production MD."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        run_production_worker(cfg, workspace=workspace, chunk_number=chunk)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(f"Phase 14 production chunk {chunk} complete")


@app.command("_chai-worker", hidden=True)
def chai_worker(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Internal LSF worker: execute Chai, validate output, and write .done."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        result = run_chai(cfg, workspace=workspace, dry_run=False)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(f"Chai stage complete: {result.selected_pdb}")


@app.command("reconcile-production")
def reconcile_production(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Reconcile validated production chunks into the aggregate production checkpoint."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        result = reconcile_production_checkpoint(cfg, workspace=workspace)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(
        f"Production checkpoint reconciled: {workspace / '07_production/.done'} "
        f"({result['results']['completed_ns']:g} ns across {result['results']['chunks']} chunks)"
    )


@app.command()
def status(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Summarize pipeline checkpoints and recorded LSF job IDs."""
    try:
        load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
    except (ValueError, ValidationError, OSError) as exc:
        _fail(exc)
    typer.echo(format_status(workspace))


@app.command()
def report(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Aggregate pipeline validation and analysis metrics into a QC report."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        result = write_qc_report(cfg, workspace=workspace)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    typer.echo(f"QC report complete: {workspace / '07_analysis/qc_report.md'}")


@app.command()
def analyze(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate CPPTRAJ/analysis inputs without executing external tools."),
) -> None:
    """Run the standard CPPTRAJ analysis suite on completed production."""
    try:
        cfg = load_config(config, check_files=True)
        workspace = _phase2_workspace(config)
        result = run_analysis(cfg, workspace=workspace, dry_run=dry_run)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    if result.dry_run:
        typer.echo(f"Prepared Phase 16 analysis inputs: {result.stage}")
    else:
        typer.echo(f"Phase 16 analysis complete: {result.stage}")
