"""Step 2.7: Controlled repeated UCI HAR DEVELOPMENT-only pilot.

Copy this file to: splitshiftlab_mvp/examples/repeated_uci_har_pilot.py
Run from the project root:
    python examples/repeated_uci_har_pilot.py

This code reads ONLY out/uci_har/uci_har_official_train.csv, never the
2,947-row official test file. Do not infer statistical significance from
five repeated, dependent development partitions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from splitshiftlab.core import (
    ProtocolSpec,
    audit_split,
    dataset_fingerprint,
    evaluate_classifier,
    generate_split,
    shift_stress_split,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "out" / "uci_har" / "uci_har_official_train.csv"
DEFAULT_OUT = ROOT / "out" / "uci_har"
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def verify_group_split(data: pd.DataFrame, spec: ProtocolSpec, split) -> tuple[int, int]:
    """Extra independent checks; audit_split is checked as well."""
    audit = audit_split(data, spec, split)
    if not audit["valid"]:
        raise AssertionError(f"Invalid split: {audit['violations']}")
    if audit.get("warnings"):
        print("  Audit warnings:", "; ".join(audit["warnings"]))
    tr, va = set(split.train_idx), set(split.test_idx)
    if tr & va or tr | va != set(range(len(data))):
        raise AssertionError("Invalid train/validation row assignment")
    tr_groups = set(data.iloc[split.train_idx]["subject"].astype(str))
    va_groups = set(data.iloc[split.test_idx]["subject"].astype(str))
    if tr_groups & va_groups:
        raise AssertionError("Participant overlap between train/validation")
    return len(tr_groups), len(va_groups)


def run_pilot(data: pd.DataFrame, n_seeds: int = 5, candidates: int = 20):
    """Compare one random candidate vs highest-shift candidate from SAME pool."""
    if n_seeds < 2 or candidates < 2:
        raise ValueError("Use at least two seeds and two candidate partitions")
    if not {"subject", "label"}.issubset(data.columns):
        raise ValueError("Missing required UCI HAR columns: subject, label")
    if data["subject"].nunique() < 4:
        raise ValueError("Need at least four distinct subjects")
    features = [c for c in data.columns if c.startswith("feature_")]
    if not features:
        raise ValueError("Missing feature_001 ... feature_561 columns")
    data = data.reset_index(drop=True)
    results = []

    # Keep the estimator and all preprocessing choices FIXED across both arms.
    # SimpleImputer and StandardScaler are fitted on the training side ONLY.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )

    for repeat in range(n_seeds):
        seed = 42 + 101 * repeat
        spec = ProtocolSpec(target="label", mode="group", group_col="subject",
                            test_size=0.25, seed=seed, features=tuple(features))

        # Existing SplitShiftLab stress optimizer: chooses highest RFF MMD²
        # among attempts seeded 0..candidates-1, using no target labels.
        stress = shift_stress_split(data, spec, candidates=candidates, components=256)
        if stress.metadata["valid_candidate_count"] != candidates:
            raise AssertionError("Candidate pool has invalid partitions; stop and inspect")

        # A random candidate from that EXACT pool, selected without inspecting
        # the labels, model predictions, or computed MMD scores.
        random_attempt = int(np.random.default_rng(seed + 99001).integers(0, candidates))
        baseline_spec = ProtocolSpec(target="label", mode="group", group_col="subject",
                                     test_size=0.25, seed=seed + random_attempt,
                                     features=tuple(features))
        baseline = generate_split(data, baseline_spec)

        n_base = verify_group_split(data, baseline_spec, baseline)
        n_stress = verify_group_split(data, spec, stress)

        # Same fixed classifier pipeline on both partitions. This is an
        # exploratory *split protocol* comparison, not causal inference.
        baseline_scores = evaluate_classifier(data, baseline_spec, baseline, model)
        stress_scores = evaluate_classifier(data, spec, stress, model)

        record = {
            "repeat": repeat + 1,
            "seed": seed,
            "random_candidate_attempt": random_attempt + 1,
            "stress_selected_attempt": int(stress.metadata["selected_attempt"]),
            "candidate_count": candidates,
            "valid_candidate_count": int(stress.metadata["valid_candidate_count"]),
            "stress_mmd2": float(stress.metadata["approx_mmd_squared"]),
            "candidate_mean_mmd2": float(stress.metadata["random_feasible_mean_mmd_squared"]),
            "random_train_rows": len(baseline.train_idx),
            "random_validation_rows": len(baseline.test_idx),
            "stress_train_rows": len(stress.train_idx),
            "stress_validation_rows": len(stress.test_idx),
            "random_train_subjects": n_base[0],
            "random_validation_subjects": n_base[1],
            "stress_train_subjects": n_stress[0],
            "stress_validation_subjects": n_stress[1],
            "group_overlap": 0,
        }
        for key in METRICS:
            record[f"random_{key}"] = baseline_scores[key]
            record[f"stress_{key}"] = stress_scores[key]
            record[f"stress_minus_random_{key}"] = stress_scores[key] - baseline_scores[key]
        results.append(record)
        print(f"Repeat {repeat+1}/{n_seeds} | seed={seed} | "
              f"random Acc={baseline_scores['accuracy']:.2%} | "
              f"stress Acc={stress_scores['accuracy']:.2%} | "
              f"stress MMD²={record['stress_mmd2']:.5f} | "
              "overlap=0", flush=True)
    return pd.DataFrame.from_records(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"Official TRAINING CSV not found: {args.csv}")
    if "official_test" in args.csv.name.lower():
        parser.error("Refusing to run on official test data")
    data = pd.read_csv(args.csv)
    if args.csv.resolve() == DEFAULT_CSV.resolve():
        if (len(data), data["subject"].nunique()) != (7352, 21):
            parser.error("Unexpected UCI HAR TRAIN shape; expected 7352 rows / 21 subjects")
        if len([c for c in data.columns if c.startswith("feature_")]) != 561:
            parser.error("Unexpected UCI HAR feature count; expected 561")

    print(f"Loading DEVELOPMENT data ONLY: {len(data)} rows, "
          f"{data['subject'].nunique()} subjects", flush=True)
    fingerprint = dataset_fingerprint(data)

    # Avoid a BLAS thread oversubscription slowdown on Windows laptops.
    with threadpool_limits(limits=2):
        table = run_pilot(data, n_seeds=args.seeds, candidates=args.candidates)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "uci_har_repeated_pilot.csv"
    json_path = args.out / "uci_har_repeated_pilot_summary.json"
    table.to_csv(csv_path, index=False)

    summary = {
        "purpose": "Exploratory, repeated development-only comparison of split-selection protocols",
        "official_test_used": False,
        "source_csv": str(args.csv),
        "dataset_sha256": fingerprint,
        "package_version": "0.1.0",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "classifier": "SimpleImputer -> StandardScaler -> RidgeClassifier(alpha=1.0)",
        "number_of_seeds": args.seeds,
        "candidates_per_seed": args.candidates,
        "results": {},
        "cautions": [
            "These are different but subject-disjoint partitions of the SAME development data.",
            "The stress score is optimized over candidate partitions; higher selected MMD is expected by construction.",
            "Mean paired differences are descriptive: seeds and partitions are not independent studies.",
            "Do not claim causality, statistical significance, or deployment generalization from this pilot.",
            "Preprocessing inside classifier pipeline is fitted to each training partition only.",
        ],
    }
    for metric in METRICS:
        delta = table[f"stress_minus_random_{metric}"].to_numpy()
        summary["results"][metric] = {
            "random_mean": float(table[f"random_{metric}"].mean()),
            "stress_mean": float(table[f"stress_{metric}"].mean()),
            "mean_stress_minus_random": float(delta.mean()),
            "sample_sd_delta": float(np.std(delta, ddof=1)),
        }
    summary["results"]["mmd"] = {
        "selected_mean": float(table["stress_mmd2"].mean()),
        "candidate_pool_mean": float(table["candidate_mean_mmd2"].mean()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nDEVELOPMENT-ONLY PILOT COMPLETED (NOT A SIGNIFICANCE TEST)")
    for metric in METRICS:
        score = summary["results"][metric]
        print(f"{metric}: random={score['random_mean']:.2%}, "
              f"stress={score['stress_mean']:.2%}, "
              f"mean difference={100*score['mean_stress_minus_random']:+.2f} pp")
    print("All tested partitions were group-disjoint.")
    print("Official UCI HAR test set NOT used.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
