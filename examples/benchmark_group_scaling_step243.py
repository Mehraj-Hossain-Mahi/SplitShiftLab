"""Step 2.43: independently validated synthetic group-count scaling benchmark.

Usage (from SplitShiftLab project root):
    python examples/benchmark_group_scaling_step243.py

Place this file alongside examples/legacy_protocol_compiler_step232.py from Step 2.42.
The installed SplitShiftLab source and datasets are not changed.
Only deterministically generated synthetic data is used; no official UCI HAR test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from itertools import combinations
import json
from math import ceil, comb
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol as batched_preflight


def import_legacy(file: Path):
    if not file.is_file():
        raise FileNotFoundError(f"Step 2.42 archived solver missing: {file}")
    module_name = "splitshiftlab._benchmark_previous_step243"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the legacy solver module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module.preflight_group_protocol


def make_grouped_dataset(n_groups: int, *, seed: int = 8243):
    rng = np.random.default_rng(seed + 101 * n_groups)
    labels, subjects = [], []
    count_matrix = np.zeros((n_groups, 6), dtype=np.int64)
    for g in range(n_groups):
        size = int(rng.integers(36, 82))
        weights = rng.dirichlet(np.full(6, 1.4))
        counts = rng.multinomial(size - 6, weights) + 1
        count_matrix[g] = counts
        for cls, count in enumerate(counts, start=1):
            labels.extend([cls] * int(count))
            subjects.extend([f"synthetic_subject_{g:02d}"] * int(count))
    frame = pd.DataFrame({"subject": subjects, "label": labels})
    assert len(frame) == int(count_matrix.sum())
    return frame, count_matrix


def independently_find_exact_boundary(count_matrix: np.ndarray, test_size: float):
    """Independent brute-force reference, not a call to SplitShiftLab."""
    groups = len(count_matrix)
    k = ceil(groups * test_size)
    total = count_matrix.sum(axis=0)
    highest = -1
    matching = 0
    first_group_witness = None
    eligible_allocations = 0
    for selection in combinations(range(groups), k):
        validation = count_matrix[list(selection)].sum(axis=0)
        training = total - validation
        if validation.sum() < 2 or training.sum() < 2:
            continue
        eligible_allocations += 1
        threshold = int(min(int(validation.min()), int(training.min())))
        if threshold > highest:
            highest = threshold
            matching = 1
            first_group_witness = [f"synthetic_subject_{i:02d}" for i in selection]
        elif threshold == highest:
            matching += 1
    if highest < 1 or eligible_allocations == 0:
        raise RuntimeError("Synthetic data unexpectedly lacks positive feasible class support")
    # If +1 violates the global necessary 2*m bound, the solver rightly rejects early;
    # that would not exercise the exact group search, so reject the fixture.
    if int(total.min()) < 2 * (highest + 1):
        raise RuntimeError("Fixture's impossible boundary fails the global support necessary bound")
    return {
        "maximum_feasible_class_support": highest,
        "first_infeasible_class_support": highest + 1,
        "allocations_at_maximum": matching,
        "first_witness_group_ids": first_group_witness,
        "allocations_checked": eligible_allocations,
        "required_validation_groups": k,
        "class_totals": list(map(int, total)),
    }


def report_signature(report: dict, threshold: int, expected_status: str, nchoosek: int):
    if report["status"] != expected_status:
        raise RuntimeError(f"Expected {expected_status}, received {report['status']} for {threshold=}")
    if expected_status == "feasible":
        cert = report.get("certificate")
        if not cert or cert["subject_overlap"] != 0:
            raise RuntimeError("Missing or overlapping feasible certificate")
        if min(cert["per_class_train_counts"].values()) < threshold:
            raise RuntimeError("Insufficient training-class support")
        if min(cert["per_class_validation_counts"].values()) < threshold:
            raise RuntimeError("Insufficient validation-class support")
    else:
        if report.get("certificate") is not None:
            raise RuntimeError("Infeasible report has a witness")
        if not report.get("class_support_search_exhausted"):
            raise RuntimeError("Infeasibility reported without complete search")
        if report.get("class_support_allocations_examined") != nchoosek:
            raise RuntimeError("Exhaustive allocation count is incorrect")
    return {
        "status": report["status"],
        "allocations_examined": report.get("class_support_allocations_examined"),
        "search_exhausted": report.get("class_support_search_exhausted"),
        "certificate": None if report.get("certificate") is None else {
            "groups": sorted(report["certificate"]["validation_group_ids"]),
            "training_rows": report["certificate"]["training_rows"],
            "validation_rows": report["certificate"]["validation_rows"],
            "per_class_train_counts": report["certificate"]["per_class_train_counts"],
            "per_class_validation_counts": report["certificate"]["per_class_validation_counts"],
        },
    }


def describe(vals):
    return {"runs": len(vals),
            "mean_seconds": round(statistics.mean(vals), 6),
            "median_seconds": round(statistics.median(vals), 6),
            "min_seconds": round(min(vals), 6),
            "max_seconds": round(max(vals), 6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("out/uci_har/benchmark_synthetic_group_scaling.json"))
    args = parser.parse_args()
    if args.runs < 2 or args.warmups < 0:
        parser.error("Require --runs >= 2 and --warmups >= 0")
    previous_path = Path(__file__).resolve().parent / "legacy_protocol_compiler_step232.py"
    previous = import_legacy(previous_path)
    solvers = {"previous": previous, "batched": batched_preflight}
    configurations = [10, 14, 18, 21]
    rows = []
    summary = []
    print("STEP 2.43 — SYNTHETIC GROUP-SCALING BENCHMARK")
    print("Independent exhaustive oracle for every synthetic configuration")
    print("Previous vs batched; excludes Python startup and CSV reading")
    for n in configurations:
        df, count_matrix = make_grouped_dataset(n)
        oracle = independently_find_exact_boundary(count_matrix, 0.25)
        k = oracle["required_validation_groups"]
        nchoosek = comb(n, k)
        if oracle["allocations_checked"] != nchoosek:
            raise RuntimeError("Oracle did not enumerate all group combinations")
        assert oracle["maximum_feasible_class_support"] > 0
        thresholds = [oracle["maximum_feasible_class_support"],
                      oracle["first_infeasible_class_support"]]
        specs = {t: ProtocolSpec(target="label", mode="group", group_col="subject",
                                 test_size=0.25, seed=42, min_train_rows=2,
                                 min_test_rows=2, min_class_support=t)
                 for t in thresholds}

        def execute(ver, threshold):
            start = perf_counter()
            report = solvers[ver](df, specs[threshold])
            elapsed = perf_counter() - start
            expected = "feasible" if threshold == thresholds[0] else "infeasible"
            signature = report_signature(report, threshold, expected, nchoosek)
            return elapsed, signature

        for _ in range(args.warmups):
            for t in thresholds:
                for ver in ("previous", "batched"):
                    execute(ver, t)
        for r in range(1, args.runs + 1):
            order_thresholds = thresholds if r % 2 else list(reversed(thresholds))
            order_versions = ("previous", "batched") if r % 2 else ("batched", "previous")
            for t in order_thresholds:
                signatures = {}
                for ver in order_versions:
                    seconds, sig = execute(ver, t)
                    signatures[ver] = sig
                    rows.append({"groups": n, "validation_groups": k,
                                 "possible_allocations": nchoosek, "round": r,
                                 "threshold": t, "status": sig["status"],
                                 "solver": ver, "seconds": round(seconds, 7),
                                 "allocations_examined": sig["allocations_examined"]})
                if signatures["previous"] != signatures["batched"]:
                    raise RuntimeError(f"Different solver results: {n=} {t=}")
        for t in thresholds:
            by_version = {ver: [row["seconds"] for row in rows
                                if row["groups"] == n and row["threshold"] == t
                                and row["solver"] == ver]
                          for ver in solvers}
            old_stat, new_stat = describe(by_version["previous"]), describe(by_version["batched"])
            faster_pairs = sum(old > new for old, new in zip(by_version["previous"], by_version["batched"]))
            stats = {"groups": n, "dataset_rows": len(df),
                     "validation_groups": k, "allocations": nchoosek,
                     "synthetic_seed": 8243 + 101 * n,
                     "boundary": oracle,
                     "scenario": "feasible" if t == thresholds[0] else "infeasible",
                     "threshold": t,
                     "previous": old_stat, "batched": new_stat,
                     "mean_speedup": round(old_stat["mean_seconds"] / new_stat["mean_seconds"], 3),
                     "batched_faster_pairs": faster_pairs}
            summary.append(stats)
            print(f"{n:2d} groups | {nchoosek:6d} allocations | {stats['scenario']:10s} "
                  f"threshold={t:3d} | previous={old_stat['median_seconds']:.4f}s "
                  f"batched={new_stat['median_seconds']:.4f}s | faster {faster_pairs}/{args.runs}")
    result = {
        "study": "Step 2.43 synthetic group-count scaling benchmark",
        "source": "Deterministically seeded synthetic grouped class counts",
        "independent_oracle": "Exhaustive evaluation of all exact-k subject combinations",
        "official_test_used": False,
        "uci_har_data_used": False,
        "measure_scope": "Whole preflight function; excludes Python startup and CSV loading",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "previous_source_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "batched_source_sha256": hashlib.sha256(Path(batched_preflight.__code__.co_filename).read_bytes()).hexdigest(),
        "runs_per_case_version": args.runs,
        "warmups_per_case_version": args.warmups,
        "summary": summary,
        "records": rows,
        "limitations": [
            "Synthetic configurations are designed for correctness and scaling diagnostics, not realistic application-performance estimates.",
            "Only one deterministic synthetic dataset per group count is evaluated.",
            "Paired order alternation reduces but cannot eliminate OS scheduler and thermal variability.",
            "Timings include hashing, row subset-sum DP and class support checks, not only the group combination loop.",
            "No superiority claim for all datasets, platforms, and workloads.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Old/new signatures match independent feasibility boundaries: PASSED")
    print("Official UCI HAR test set: NOT ACCESSED")
    print(f"JSON: {args.output}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
