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
