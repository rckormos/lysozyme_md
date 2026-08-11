from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from .config import load_config
from .chai import run_chai
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


@app.command()
def prepare(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    through: Optional[str] = typer.Option(None, "--through"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run implemented preparation stages; Phase 2 currently implements Chai only."""
    try:
        cfg = load_config(config, check_files=True)
        if from_stage not in (None, "chai"):
            raise ValueError("Phase 2 supports --from chai only")
        if through not in (None, "chai"):
            raise ValueError("Phase 2 supports --through chai only")
        workspace = config.parent.resolve()
        if not (workspace / "manifest.json").is_file():
            raise ValueError("prepare must be run with the initialized workspace config.yaml")
        if not cfg.chai.enabled:
            typer.echo("Chai stage disabled by configuration")
            return
        result = run_chai(cfg, workspace=workspace, dry_run=dry_run)
    except (ValueError, ValidationError, OSError, RuntimeError) as exc:
        _fail(exc)
    if result.dry_run:
        typer.echo(f"Prepared Chai dry run: {result.stage_dir}")
    else:
        typer.echo(f"Chai stage complete: {result.selected_pdb}")


@app.command()
def submit(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    through: Optional[str] = typer.Option(None, "--through"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Submit LSF stages (stub in Phases 0-1)."""
    _stub("submit", config, from_stage, through, dry_run)


@app.command()
def status(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Summarize pipeline state (stub in Phases 0-1)."""
    _stub("status", config, None, None, False)


@app.command()
def analyze(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
) -> None:
    """Run standard analysis (stub in Phases 0-1)."""
    _stub("analyze", config, None, None, False)
