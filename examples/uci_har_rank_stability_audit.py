"""Step 2.12: audit candidate-ranking consistency and diagnostic relevance.

Inputs must be produced by Step 2.8 and Step 2.11 on the *same*
UCI HAR development pool (same 20 group-disjoint partitions).

Reads only saved development experiment outputs. Never opens the official test data.
Correlation results are descriptive; candidates are not independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return None
    r = float(spearmanr(x, y).statistic)
    return r if np.isfinite(r) else None


def read_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; complete Steps 2.8 and 2.11 first.")
    table = pd.read_csv(path)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if table.empty or table["candidate"].isna().any() or table["candidate"].duplicated().any():
        raise ValueError(f"{path.name}: empty, missing or duplicate candidate IDs")
    table["candidate"] = pd.to_numeric(table["candidate"], errors="raise").astype(int)
    return table.sort_values("candidate").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("out/uci_har"))
    args = parser.parse_args()
    directory = args.input_dir
    legacy_path = directory / "uci_har_candidate_audit.csv"
    robust_path = directory / "uci_har_robust_candidate_scores.csv"
    legacy = read_table(legacy_path, {"candidate", "mmd_squared", "accuracy", "train_rows", "validation_rows", "subject_overlap"})
    robust = read_table(robust_path, {"candidate", "consensus_score", "is_selected", "train_rows", "validation_rows", "constraint_valid"})
    mappings = [c for c in robust if c.startswith("mmd_bw_")]
    if len(mappings) < 2:
        raise ValueError("At least two multi-kernel RFF mapping columns are required")
    if legacy["candidate"].tolist() != robust["candidate"].tolist():
        raise ValueError("Candidate IDs differ between files; do not compare different pools")
    for column in ("train_rows", "validation_rows"):
        if not np.array_equal(legacy[column].to_numpy(), robust[column].to_numpy()):
            raise ValueError(f"Candidate {column} differ between experiments; pools may not match")
    if not robust["constraint_valid"].astype(str).str.lower().eq("true").all():
        raise ValueError("Not all multi-kernel candidate splits pass constraint validation")
    if not (legacy["subject_overlap"] == 0).all():
        raise ValueError("Legacy candidate pool has subject overlap")
    if len(robust) != 20:
        print(f"WARNING: expected 20 candidates, received {len(robust)}")
    is_selected = robust["is_selected"].astype(str).str.lower().eq("true")
    if int(is_selected.sum()) != 1:
        raise ValueError("Expected exactly one consensus-selected candidate")

    # Verify dataset fingerprints when both manifest summaries are available.
    prior_path = directory / "uci_har_candidate_audit_summary.json"
    experimental_path = directory / "uci_har_robust_experiment.json"
    fingerprints_match = None
    if prior_path.exists() and experimental_path.exists():
        old = json.loads(prior_path.read_text(encoding="utf-8"))
        new = json.loads(experimental_path.read_text(encoding="utf-8"))
        a, b = old.get("dataset_sha256"), new.get("data_sha256")
        fingerprints_match = bool(a and b and a == b)
        if not fingerprints_match:
            raise ValueError("Dataset fingerprints differ between Step 2.8 and Step 2.11")
        if old.get("official_test_used") is not False or new.get("official_test_used") is not False:
            raise ValueError("At least one manifest does not certify development-only evaluation")

    all_scores = robust[mappings].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(all_scores).all():
        raise ValueError("Mapping scores contain nonfinite values")
    mmd = legacy["mmd_squared"].to_numpy(dtype=float)
    error = 1.0 - legacy["accuracy"].to_numpy(dtype=float)
    consensus = robust["consensus_score"].to_numpy(dtype=float)
    ids = robust["candidate"].to_numpy(dtype=int)
    winners = [int(ids[int(np.argmax(all_scores[:, j]))]) for j in range(len(mappings))]
    legacy_winner = int(ids[int(np.argmax(mmd))])
    robust_winner = int(ids[int(np.argmax(consensus))])
    flagged = int(robust.loc[is_selected, "candidate"].iloc[0])
    if robust_winner != flagged:
        raise ValueError("Consensus-selected flag contradicts maximum consensus score")

    pairwise = []
    for a, b in combinations(range(len(mappings)), 2):
        pairwise.append({"mapping_a": mappings[a], "mapping_b": mappings[b],
                         "spearman_rho": finite_spearman(all_scores[:, a], all_scores[:, b])})
    pairwise_df = pd.DataFrame(pairwise)
    pairwise_vals = pairwise_df["spearman_rho"].dropna()
    each_mapping = []
    for j, name in enumerate(mappings):
        each_mapping.append({"mapping": name, "winner": winners[j],
                             "spearman_with_error": finite_spearman(all_scores[:, j], error),
                             "spearman_with_legacy_mmd": finite_spearman(all_scores[:, j], mmd)})

    legacy_error_rho = finite_spearman(mmd, error)
    consensus_error_rho = finite_spearman(consensus, error)
    ranking_df = pd.DataFrame({
        "candidate": ids,
        "legacy_mmd_squared": mmd,
        "consensus_score": consensus,
        "classification_error": error,
        "legacy_rank_highest_first": pd.Series(-mmd).rank(method="min").astype(int),
        "consensus_rank_highest_first": pd.Series(-consensus).rank(method="min").astype(int),
    })
    ranking_df.to_csv(directory / "uci_har_ranking_comparison.csv", index=False)
    pairwise_df.to_csv(directory / "uci_har_mapping_rank_agreement.csv", index=False)
    summary = {
        "study": "Step 2.12 development-only fixed-candidate-pool ranking audit",
        "official_test_used": False,
        "candidate_count": len(ids),
        "mapping_count": len(mappings),
        "dataset_fingerprints_verified": fingerprints_match,
        "legacy_winner": legacy_winner,
        "consensus_winner": robust_winner,
        "per_mapping_winners": winners,
        "unanimous_mapping_winner": len(set(winners)) == 1,
        "selectors_choose_same_candidate": legacy_winner == robust_winner,
        "pairwise_mapping_spearman_mean": float(pairwise_vals.mean()) if len(pairwise_vals) else None,
        "pairwise_mapping_spearman_min": float(pairwise_vals.min()) if len(pairwise_vals) else None,
        "pairwise_mapping_spearman_max": float(pairwise_vals.max()) if len(pairwise_vals) else None,
        "legacy_mmd_vs_error_spearman": legacy_error_rho,
        "consensus_vs_error_spearman": consensus_error_rho,
        "consensus_vs_legacy_spearman": finite_spearman(consensus, mmd),
        "per_mapping_diagnostics": each_mapping,
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (legacy_path, robust_path)
        },
        "limitations": [
            "All correlations are descriptive; candidate partitions overlap and are not independent.",
            "The same training dataset and candidate pool were used to create both rankings.",
            "Comparing correlations with error is exploratory, not proof of better generalization.",
            "Candidate IDs and partition sizes agree; exact assignments for every candidate are not independently rehashed.",
            "Six related feature mappings are not six independent studies.",
            "The official UCI HAR test set was never read by this script.",
        ],
    }
    target = directory / "uci_har_rank_stability_summary.json"
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def fmt(v):
        return "undefined" if v is None else f"{v:+.3f}"
    print("STEP 2.12 — DEVELOPMENT-ONLY RANKING AUDIT COMPLETE")
    print(f"Candidates: {len(ids)} | Feature mappings: {len(mappings)}")
    print(f"Legacy winner: #{legacy_winner} | Consensus winner: #{robust_winner}")
    print(f"Unanimous mapping winner: {summary['unanimous_mapping_winner']}")
    print(f"Mean pairwise mapping Spearman: {fmt(summary['pairwise_mapping_spearman_mean'])}")
    print(f"Legacy MMD vs classification error: {fmt(legacy_error_rho)}")
    print(f"Consensus vs classification error: {fmt(consensus_error_rho)}")
    print("Descriptive analysis only; not statistical significance or causation.")
    print("Official UCI HAR test data NOT USED.")
    print(f"Saved: {target}")
    print(f"Saved: {directory / 'uci_har_ranking_comparison.csv'}")
    print(f"Saved: {directory / 'uci_har_mapping_rank_agreement.csv'}")


if __name__ == "__main__":
    main()
