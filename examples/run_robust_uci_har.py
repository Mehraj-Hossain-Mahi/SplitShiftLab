"""Step 2.10: compare legacy and experimental robust selectors using development data ONLY.

Do not interpret a more difficult split as generalization improvement or causation.
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from splitshiftlab.core import ProtocolSpec, evaluate_classifier, shift_stress_split
from splitshiftlab.robust_shift import robust_shift_split

DATA = Path("out/uci_har/uci_har_official_train.csv")
OUTPUT = Path("out/uci_har")


def main():
    if not DATA.is_file():
        raise SystemExit(f"Cannot find {DATA}. Run from the splitshiftlab_mvp project folder.")
    df = pd.read_csv(DATA)
    if len(df) != 7352:
        raise ValueError("Unexpected UCI HAR training length; verify the source CSV")
    spec = ProtocolSpec(target="label", mode="group", group_col="subject", test_size=0.25, seed=42)
    print(f"DEVELOPMENT DATA ONLY: {len(df)} rows, {df['subject'].nunique()} subjects")
    print("Computing multi-kernel shift ranking (20 candidates, 6 feature mappings)...")
    report = robust_shift_split(df, spec, candidates=20, components=64,
                                bandwidths=(0.5, 1.0, 2.0), repeats=2)
    legacy = shift_stress_split(df, spec, candidates=20, components=256)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeClassifier(alpha=1.0))
    robust_perf = evaluate_classifier(df, spec, report.selected, model)
    legacy_perf = evaluate_classifier(df, spec, legacy, model)
    for name, split in [("Legacy", legacy), ("Experimental", report.selected)]:
        t = set(df.iloc[split.train_idx]["subject"])
        v = set(df.iloc[split.test_idx]["subject"])
        assert split.audit["valid"] and t.isdisjoint(v)
        print(f"{name}: candidate={split.metadata['selected_attempt']}, "
              f"train={len(split.train_idx)}, validation={len(split.test_idx)}, "
              f"subjects={len(t)}/{len(v)}, overlap=0")
    print(f"Legacy accuracy:       {legacy_perf['accuracy']:.2%}")
    print(f"Experimental accuracy: {robust_perf['accuracy']:.2%}")
    print("These partitions can differ; higher/lower accuracy is NOT proof of a better method.")
    print("Official UCI HAR test file was NOT opened.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates_path = OUTPUT / "uci_har_robust_candidate_scores.csv"
    summary_path = OUTPUT / "uci_har_robust_experiment.json"
    report.candidates.to_csv(candidates_path, index=False)
    payload = {
        "study": "Step 2.10 exploratory development-only method prototype",
        "official_test_used": False,
        "data_sha256": report.metadata["dataset_sha256"],
        "constraint_valid": True,
        "candidate_count": len(report.candidates),
        "legacy": {"selected_candidate": legacy.metadata["selected_attempt"],
                   "approx_mmd_squared": legacy.metadata["approx_mmd_squared"],
                   "classifier": legacy_perf},
        "experimental": {"selected_candidate": report.metadata["selected_attempt"],
                         "consensus_score": report.metadata["score"],
                         "rank_disagreement_sd": report.metadata["rank_disagreement_sd"],
                         "per_mapping_winners": report.metadata["per_mapping_winners"],
                         "classifier": robust_perf},
        "cautions": [
            "Both selectors use unlabeled UCI HAR development features; final test is untouched.",
            "This is not a statistically independent or causal comparison.",
            "The experimental method remains a heuristic requiring multi-dataset validation.",
            "Neither selected diagnostic accuracy estimates average deployment performance.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {candidates_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
