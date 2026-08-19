from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGES: tuple[tuple[str, Path], ...] = (
    ("chai", Path("01_chai/.done")),
    ("glycam", Path("02_prepare/glycam/.done")),
    ("mapping", Path("02_prepare/mapping/.done")),
    ("coordinates", Path("02_prepare/coordinate_transfer/.done")),
    ("protein", Path("02_prepare/protein/.done")),
    ("leap", Path("03_dry_relax/.done")),
    ("hydrogen-relax", Path("03_dry_relax/hydrogen_relax/.done")),
    ("solvate", Path("04_solvate/.done")),
    ("minimize", Path("05_minimize/.done")),
    ("heat", Path("06_equilibrate/heat/.done")),
    ("npt-smoke", Path("06_equilibrate/npt_smoke/.done")),
    ("npt-equilibrate", Path("06_equilibrate/.done")),
    ("production", Path("07_production/.done")),
    ("analysis", Path("07_analysis/.done")),
)


@dataclass(frozen=True)
class StageStatus:
    name: str
    done: bool
    sentinel: str
    job_ids: tuple[str, ...] = ()


def _job_ids_from_json(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    values: list[str] = []
    for key in ("job_id", "solvent_job_id", "all_job_id"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if value:
            values.append(str(value))
    job_ids = payload.get("job_ids") if isinstance(payload, dict) else None
    if isinstance(job_ids, list):
        values.extend(str(value) for value in job_ids if value)
    return tuple(dict.fromkeys(values))


def collect_status(workspace: Path) -> tuple[StageStatus, ...]:
    workspace = Path(workspace).resolve()
    statuses: list[StageStatus] = []
    submission_files = {
        "chai": workspace / "01_chai/submission.json",
        "minimize": workspace / "05_minimize/submission.json",
        "heat": workspace / "06_equilibrate/heat/submission.json",
        "npt-smoke": workspace / "06_equilibrate/npt_smoke/submission.json",
        "npt-equilibrate": workspace / "06_equilibrate/npt_equilibrate/submission.json",
        "production": workspace / "07_production/submission.json",
    }
    for name, relative in STAGES:
        sentinel = workspace / relative
        statuses.append(StageStatus(name, sentinel.is_file(), str(sentinel), _job_ids_from_json(submission_files.get(name, Path()))) )
    return tuple(statuses)


def format_status(workspace: Path) -> str:
    statuses = collect_status(workspace)
    lines = ["Stage status:", ""]
    for status in statuses:
        state = "DONE" if status.done else "PENDING"
        jobs = f"  jobs={','.join(status.job_ids)}" if status.job_ids else ""
        lines.append(f"{status.name:16s} {state:7s}{jobs}")
    return "\n".join(lines)
