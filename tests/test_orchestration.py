from pathlib import Path
import json

from lyso_md.orchestration import collect_status, format_status


def test_collect_status_reports_done_and_job_ids(tmp_path: Path) -> None:
    (tmp_path / "01_chai").mkdir(parents=True)
    (tmp_path / "01_chai/.done").write_text("{}\n")
    (tmp_path / "01_chai/submission.json").write_text(json.dumps({"job_id": "12345"}))
    (tmp_path / "07_production").mkdir(parents=True)
    (tmp_path / "07_production/.done").write_text("{}\n")
    (tmp_path / "07_production/submission.json").write_text(json.dumps({"job_id": "999", "job_ids": ["999"]}))

    statuses = {item.name: item for item in collect_status(tmp_path)}
    assert statuses["chai"].done is True
    assert statuses["chai"].job_ids == ("12345",)
    assert statuses["production"].done is True
    assert statuses["production"].job_ids == ("999",)
    assert statuses["npt-equilibrate"].done is False


def test_format_status_is_human_readable(tmp_path: Path) -> None:
    (tmp_path / "06_equilibrate").mkdir(parents=True)
    (tmp_path / "06_equilibrate/.done").write_text("{}\n")
    text = format_status(tmp_path)
    assert "npt-equilibrate  DONE" in text
    assert "production       PENDING" in text


def test_reconcile_production_creates_aggregate_checkpoint(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from lyso_md.production import reconcile_production_checkpoint

    root = tmp_path / "07_production"
    root.mkdir(parents=True)
    for number, completed_ns in ((1, 250.0), (2, 500.0), (3, 750.0), (4, 1000.0)):
        stage = root / f"chunk_{number:03d}"
        stage.mkdir()
        (stage / "production.rst7").write_text("restart\n")
        (stage / ".done").write_text("{}\n")
        (stage / "validation.json").write_text(json.dumps({
            "status": "done",
            "checks": {"passed": True},
            "results": {"completed_ns": completed_ns},
        }) + "\n")

    cfg = SimpleNamespace(production=SimpleNamespace(target_ns=1000.0))
    result = reconcile_production_checkpoint(cfg, workspace=tmp_path)

    assert result["checks"]["passed"] is True
    assert result["results"]["completed_ns"] == 1000.0
    assert (root / ".done").is_file()
    assert (root / "production_validation.json").is_file()


def test_reconcile_production_refuses_incomplete_target(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from lyso_md.production import reconcile_production_checkpoint

    root = tmp_path / "07_production" / "chunk_001"
    root.mkdir(parents=True)
    (root / "production.rst7").write_text("restart\n")
    (root / ".done").write_text("{}\n")
    (root / "validation.json").write_text(json.dumps({
        "status": "done",
        "checks": {"passed": True},
        "results": {"completed_ns": 250.0},
    }) + "\n")

    cfg = SimpleNamespace(production=SimpleNamespace(target_ns=1000.0))
    import pytest
    with pytest.raises(ValueError, match="below the configured target"):
        reconcile_production_checkpoint(cfg, workspace=tmp_path)
    assert not (tmp_path / "07_production/.done").exists()
