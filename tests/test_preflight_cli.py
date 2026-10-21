"""CLI integration tests: no access to downloaded benchmark test partitions."""
import json
from pathlib import Path

import pandas as pd
import pytest

from splitshiftlab.cli import main


def demo_csv(tmp_path: Path) -> Path:
    # 4 whole groups with 10 rows each. 50% validation => 2 groups.
    df = pd.DataFrame({
        "subject": [f"subject_{g}" for g in range(4) for _ in range(10)],
        "label": [r % 2 for g in range(4) for r in range(10)],
        "feature": [g + r / 10 for g in range(4) for r in range(10)],
    })
    file = tmp_path / "development.csv"
    df.to_csv(file, index=False)
    return file


def preflight_argv(csv, path, *additional):
    return ["preflight", "--csv", str(csv), "--target", "label",
            "--group", "subject", "--test-size", "0.5", "--output", str(path),
            *additional]


def test_feasible_cli_writes_valid_json(tmp_path, capsys):
    csv = demo_csv(tmp_path)
    report_file = tmp_path / "feasible.json"
    main(preflight_argv(csv, report_file))
    report = json.loads(report_file.read_text())
    assert report["status"] == "feasible"
    assert report["certificate"]["subject_overlap"] == 0
    assert report["certificate"]["training_rows"] == 20
    assert report["certificate"]["validation_rows"] == 20
    assert "PREFLIGHT: FEASIBLE" in capsys.readouterr().out


def test_infeasible_cli_writes_reason(tmp_path, capsys):
    csv = demo_csv(tmp_path)
    report_file = tmp_path / "infeasible.json"
    main(preflight_argv(csv, report_file, "--min-validation-rows", "38"))
    report = json.loads(report_file.read_text())
    assert report["status"] == "infeasible"
    assert report["certificate"] is None
    assert report["blockers"][0]["code"] == "NO_VALID_GROUP_ALLOCATION"
    assert "PREFLIGHT: INFEASIBLE" in capsys.readouterr().out


def test_resource_caps_stay_undetermined(tmp_path, capsys):
    csv = demo_csv(tmp_path)
    out = tmp_path / "undetermined.json"
    main(preflight_argv(csv, out, "--max-groups", "2"))
    report = json.loads(out.read_text())
    assert report["status"] == "undetermined"
    assert report["blockers"] == []
    assert "PREFLIGHT: UNDETERMINED" in capsys.readouterr().out


def test_existing_demo_command_preserved(tmp_path):
    csv = tmp_path / "demo.csv"
    out = tmp_path / "demo.json"
    main(["demo", "--csv-output", str(csv), "--output", str(out)])
    assert csv.exists() and out.exists()
    data = json.loads(out.read_text())
    assert data["split"]["audit"]["valid"] is True
