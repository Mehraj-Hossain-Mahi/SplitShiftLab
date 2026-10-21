"""Command-line interface for SplitShiftLab research prototype."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import ProtocolSpec, evaluate_classifier, generate_split, shift_stress_split
from .protocol_compiler import preflight_group_protocol


def demo_data(n_groups: int = 36, rows_per_group: int = 12, seed: int = 9) -> pd.DataFrame:
    """Create purely synthetic group-correlated data for software demonstrations."""
    rng = np.random.default_rng(seed)
    records = []
    for group in range(n_groups):
        offset = group / n_groups * 2 - 1
        for row in range(rows_per_group):
            f1 = rng.normal(offset, 0.65)
            f2 = rng.normal(0.25 * offset, 1.1)
            target = int(f1 + 0.8 * f2 + rng.normal(0, 0.55) > 0.15)
            records.append({"subject": f"subject_{group:03d}",
                            "event_time": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=group * 2, hours=row)).isoformat(),
                            "feature_1": round(float(f1), 7),
                            "feature_2": round(float(f2), 7),
                            "label": target})
    return pd.DataFrame(records)


def _execute(df, spec, *, stress, candidates, evaluate, output):
    result = shift_stress_split(df, spec, candidates=candidates) if stress else generate_split(df, spec)
    payload = {"package": "splitshiftlab", "version": __version__,
               "input_rows": len(df), "specification": asdict(spec),
               "split": result.to_dict()}
    if evaluate:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        features = list(spec.features) if spec.features else [c for c in df.select_dtypes(include="number").columns if c != spec.target]
        if not features:
            raise ValueError("Cannot train demonstration classifier without numeric features")
        estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=1000, random_state=spec.seed))
        payload["classifier_demo"] = evaluate_classifier(df, spec, result, estimator)
        payload["classifier_demo_note"] = "Example estimator only; no hyperparameter tuning or scientific benchmarking performed."
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved reproducible split manifest to: {path}")
    print(f"Protocol valid: {result.audit['valid']} | Train: {len(result.train_idx)} | Test: {len(result.test_idx)} | Excluded: {len(result.excluded_idx)}")
    if result.audit["warnings"]:
        print("Review warnings:", "; ".join(result.audit["warnings"]))
    if stress:
        print(f"Selected approximate MMD^2: {result.metadata['approx_mmd_squared']:.5f} (diagnostic stress test, NOT unbiased test performance)")
    if evaluate:
        print("Classifier demo:", payload["classifier_demo"])
    return payload


def _run_preflight(args):
    """Write a machine-readable constraint feasibility report without fitting any ML model."""
    df = pd.read_csv(args.csv)
    spec = ProtocolSpec(
        target=args.target,
        mode="group",
        group_col=args.group,
        test_size=args.test_size,
        seed=args.seed,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_validation_rows,
        min_class_support=args.minimum_class_support,
    )
    report = preflight_group_protocol(
        df, spec, max_groups=args.max_groups, max_rows=args.max_rows,
        max_class_allocations=args.max_class_allocations
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"PREFLIGHT: {report['status'].upper()}")
    print(f"Dataset: {len(df)} rows | Group field: {args.group}")
    if report.get("certificate") is not None:
        c = report["certificate"]
        print(f"Certificate: train={c['training_rows']} rows / {c['training_groups']} groups; "
              f"validation={c['validation_rows']} rows / {c['validation_groups']} groups; "
              f"overlap={c['subject_overlap']}")
    for blocker in report.get("blockers", []):
        print(f"Blocker: {blocker['code']} — {blocker['message']}")
    for requirement in report.get("unresolved_requirements", []):
        print(f"Unresolved: {requirement}")
    print(f"Saved report: {path}")
    print("Preflight checks feasibility only; no estimator is trained and no official test set is accessed unless supplied as --csv.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="splitshiftlab", description="Leakage-aware split protocols and development-set shift diagnostics")
    parser.add_argument("--version", action="version", version=f"splitshiftlab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create synthetic data and run group split + classifier")
    demo.add_argument("--output", default="out/demo_manifest.json")
    demo.add_argument("--csv-output", default="out/demo_data.csv")
    demo.add_argument("--stress", action="store_true")
    demo.add_argument("--candidates", type=int, default=80)
    run = commands.add_parser("run", help="Split your CSV and write JSON manifest")
    run.add_argument("--csv", required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--group", default=None)
    run.add_argument("--time", default=None)
    run.add_argument("--features", nargs="+", default=None, help="Numeric input features; never include target")
    run.add_argument("--mode", choices=["random", "group", "temporal", "group_temporal"], default="group")
    run.add_argument("--test-size", type=float, default=0.25)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--minimum-class-support", type=int, default=0)
    run.add_argument("--stress", action="store_true", help="Diagnostic group stress split; only use development data")
    run.add_argument("--candidates", type=int, default=80)
    run.add_argument("--evaluate", action="store_true", help="Run example logistic-regression classification baseline")
    run.add_argument("--output", default="out/manifest.json")
    preflight = commands.add_parser("preflight", help="Prove feasibility or explain impossibility of a group-disjoint protocol")
    preflight.add_argument("--csv", required=True, help="Development CSV only; never supply the protected final test dataset")
    preflight.add_argument("--target", required=True)
    preflight.add_argument("--group", required=True)
    preflight.add_argument("--test-size", type=float, default=0.25, help="Fraction of GROUPS, not rows")
    preflight.add_argument("--seed", type=int, default=42)
    preflight.add_argument("--min-train-rows", type=int, default=2)
    preflight.add_argument("--min-validation-rows", type=int, default=2)
    preflight.add_argument("--minimum-class-support", type=int, default=0)
    preflight.add_argument("--max-groups", type=int, default=60)
    preflight.add_argument("--max-rows", type=int, default=200000)
    preflight.add_argument("--max-class-allocations", type=int, default=250000,
                           help="Maximum exhaustive combinations for class-support proof")
    preflight.add_argument("--output", default="out/preflight_report.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            _run_preflight(args)
        elif args.command == "demo":
            df = demo_data()
            csvpath = Path(args.csv_output)
            csvpath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csvpath, index=False)
            print("Synthetic demonstration data:", csvpath)
            spec = ProtocolSpec(target="label", mode="group", group_col="subject",
                                features=("feature_1", "feature_2"), seed=42)
            _execute(df, spec, stress=args.stress, candidates=args.candidates,
                     evaluate=True, output=args.output)
        else:
            df = pd.read_csv(args.csv)
            spec = ProtocolSpec(target=args.target, mode=args.mode, group_col=args.group,
                                time_col=args.time, features=tuple(args.features) if args.features else None,
                                test_size=args.test_size, seed=args.seed,
                                min_class_support=args.minimum_class_support)
            _execute(df, spec, stress=args.stress, candidates=args.candidates,
                     evaluate=args.evaluate, output=args.output)
    except (ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"SplitShiftLab error: {exc}\n")


if __name__ == "__main__":
    main()
