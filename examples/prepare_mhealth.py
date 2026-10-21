"""Step 2.46: Lossless, streaming conversion of the UCI MHEALTH log files.

Expected project layout:
    data/mhealth/**/mHealth_subject1.log ... mHealth_subject10.log

Output files:
    out/mhealth/mhealth_all_subjects.csv
    out/mhealth/mhealth_preparation_report.json

This script includes activity 0 (background). It does not train models, make
train/test splits, or access the protected UCI HAR test dataset.

Run from the SplitShiftLab project root:
    python examples/prepare_mhealth.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import sys
import tempfile


NUM_FEATURES = 23
EXPECTED_SUBJECTS = tuple(range(1, 11))
ALLOWED_LABELS = set(range(13))  # background=0, activity classes=1..12
FILENAME_RE = re.compile(r"mHealth_subject([0-9]+)\.log", re.IGNORECASE)
CSV_HEADER = [f"sensor_{i:02d}" for i in range(1, NUM_FEATURES + 1)] + ["subject", "label"]


def discover_subjects(data_dir: Path) -> list[tuple[int, Path]]:
    if not data_dir.is_dir():
        raise ValueError(f"MHEALTH data folder does not exist: {data_dir}")

    found: dict[int, Path] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        match = FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        subject_no = int(match.group(1))
        if subject_no not in EXPECTED_SUBJECTS:
            raise ValueError(f"Unexpected subject ID in filename: {path}")
        if subject_no in found:
            raise ValueError(f"Duplicate subject_{subject_no:02d} logs: {found[subject_no]} and {path}")
        found[subject_no] = path

    missing = sorted(set(EXPECTED_SUBJECTS) - set(found))
    if missing:
        raise ValueError(f"Missing MHEALTH log files for subject IDs: {missing}")
    return [(sid, found[sid]) for sid in EXPECTED_SUBJECTS]


def convert(data_dir: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    subjects = discover_subjects(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mhealth_all_subjects.csv"
    report_path = output_dir / "mhealth_preparation_report.json"

    # A temporary CSV is used to avoid publishing an incomplete dataset on errors.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv.tmp",
        prefix="mhealth_", dir=output_dir, delete=False,
    ) as out_file:
        temp_path = Path(out_file.name)
        writer = csv.writer(out_file)
        writer.writerow(CSV_HEADER)
        total_counts: Counter[int] = Counter()
        per_subject: dict[str, dict] = {}
        total_rows = 0
        total_blank_lines = 0
        try:
            for sid, path in subjects:
                name = f"subject_{sid:02d}"
                counts: Counter[int] = Counter()
                subject_rows = 0
                blank_lines = 0
                with path.open("r", encoding="utf-8-sig") as in_file:
                    for line_no, line in enumerate(in_file, start=1):
                        fields = line.split()
                        if not fields:
                            blank_lines += 1
                            continue
                        if len(fields) != NUM_FEATURES + 1:
                            raise ValueError(
                                f"{path.name}:{line_no}: expected 24 whitespace-separated "
                                f"values, found {len(fields)}"
                            )
                        for col, value in enumerate(fields[:NUM_FEATURES], start=1):
                            try:
                                number = float(value)
                            except ValueError as exc:
                                raise ValueError(
                                    f"{path.name}:{line_no}: sensor column {col} "
                                    f"is not numeric: {value!r}"
                                ) from exc
                            if not math.isfinite(number):
                                raise ValueError(
                                    f"{path.name}:{line_no}: sensor column {col} "
                                    f"is not finite: {value!r}"
                                )
                        try:
                            label = int(fields[-1])
                        except ValueError as exc:
                            raise ValueError(
                                f"{path.name}:{line_no}: invalid integer activity "
                                f"label {fields[-1]!r}"
                            ) from exc
                        if label not in ALLOWED_LABELS:
                            raise ValueError(
                                f"{path.name}:{line_no}: unexpected activity "
                                f"label {label}; expected 0..12"
                            )
                        # Keep original feature strings (no rounding/reformatting).
                        writer.writerow([*fields[:NUM_FEATURES], name, label])
                        subject_rows += 1
                        counts[label] += 1
                if subject_rows == 0:
                    raise ValueError(f"{path} contains no usable records")
                total_rows += subject_rows
                total_blank_lines += blank_lines
                total_counts.update(counts)
                per_subject[name] = {
                    "source_file": str(path.as_posix()),
                    "rows": subject_rows,
                    "blank_lines_skipped": blank_lines,
                    "label_counts": {str(i): counts[i] for i in range(13)},
                }
                print(f"Converted {name}: {subject_rows:,} rows (background={counts[0]:,})", flush=True)
        except BaseException:
            out_file.flush()
            temp_path.unlink(missing_ok=True)
            raise

    temp_path.replace(csv_path)
    digest = hashlib.sha256()
    with csv_path.open("rb") as csv_file:
        for block in iter(lambda: csv_file.read(1024 * 1024), b""):
            digest.update(block)

    report = {
        "study": "Step 2.46 UCI MHEALTH lossless log-to-CSV preparation",
        "source_directory": str(data_dir.as_posix()),
        "output_csv": str(csv_path.as_posix()),
        "output_csv_sha256": digest.hexdigest(),
        "records": total_rows,
        "subjects": len(subjects),
        "features": NUM_FEATURES,
        "columns": len(CSV_HEADER),
        "feature_columns": CSV_HEADER[:NUM_FEATURES],
        "subject_column": "subject",
        "target_column": "label",
        "labels_preserved": list(range(13)),
        "background_label_zero_retained": True,
        "background_rows": total_counts[0],
        "non_background_rows": total_rows - total_counts[0],
        "total_blank_lines_skipped": total_blank_lines,
        "label_counts": {str(i): total_counts[i] for i in range(13)},
        "per_subject": per_subject,
        "source_files": [str(path.as_posix()) for _, path in subjects],
        "official_uci_har_test_used": False,
        "notes": [
            "Sensor features are named generically; consult MHEALTH documentation for physical meanings.",
            "All activity-0 background rows are retained. No data split or ML training is performed.",
            "Original numeric strings are preserved in the CSV, except whitespace delimiters.",
        ],
    }
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    return csv_path, report_path, report


def main() -> int:
    data_dir = Path("data/mhealth")
    output_dir = Path("out/mhealth")
    try:
        csv_path, report_path, report = convert(data_dir, output_dir)
    except (OSError, ValueError) as exc:
        print(f"MHEALTH PREPARATION: FAILED — {exc}", file=sys.stderr)
        return 1
    print("MHEALTH PREPARATION: COMPLETE")
    print(f"Subjects: {report['subjects']}")
    print(f"Records: {report['records']:,}")
    print(f"Features: {report['features']}; columns (including subject and label): {report['columns']}")
    print(f"Background label 0 retained: {report['background_rows']:,} records")
    print(f"Activity labels present: {[i for i in report['labels_preserved'] if report['label_counts'][str(i)] > 0]}")
    print(f"CSV saved: {csv_path}")
    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
