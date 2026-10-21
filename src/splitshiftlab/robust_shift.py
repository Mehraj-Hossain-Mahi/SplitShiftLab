"""Experimental development-only ensemble MMD partition selection.

Not a new global optimizer, proof of harder cases, or final-test estimator.
This additive module is compatible with SplitShiftLab v0.1.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core import (ProtocolSpec, SplitResult, _selected_features, _split_group,
                   _validate_frame, audit_split, dataset_fingerprint)


@dataclass
class RobustShiftReport:
    selected: SplitResult
    candidates: pd.DataFrame
    metadata: dict[str, Any]


def _unlabeled_design_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Standardize ONLY the authorized development feature matrix, never final-test data."""
    x = df[features].apply(pd.to_numeric, errors="raise")
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.isna().all().any():
        raise ValueError("A feature column has no finite values")
    x = x.fillna(x.median()).to_numpy(dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale


def _rff(x: np.ndarray, *, bandwidth: float, components: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / (np.sqrt(x.shape[1]) * bandwidth),
                   size=(x.shape[1], components))
    b = rng.uniform(0.0, 2.0 * np.pi, components)
    return np.sqrt(2.0 / components) * np.cos(x @ w + b)


def _rank_percentiles(scores: np.ndarray) -> np.ndarray:
    """Rank each mapping from low to high score; deterministic tie behavior."""
    n, k = scores.shape
    ranks = np.zeros_like(scores)
    for j in range(k):
        ix = np.argsort(scores[:, j], kind="stable")
        ranks[ix, j] = np.arange(n, dtype=float) / max(1, n - 1)
    return ranks


def robust_shift_split(
    df: pd.DataFrame,
    spec: ProtocolSpec,
    *,
    candidates: int = 20,
    components: int = 64,
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0),
    repeats: int = 2,
    disagreement_penalty: float = 0.2,
) -> RobustShiftReport:
    """Rank valid development partitions using multi-bandwidth, multi-seed RFF-MMD.

    Candidate selection uses feature values, group IDs and explicit constraints;
    it NEVER uses target labels, model outcomes or the reserved official test set.
    A robustness-adjusted rank consensus is diagnostic, not deployment risk.
    """
    if spec.mode != "group":
        raise ValueError("Robust shift currently supports group-disjoint mode only")
    if spec.min_class_support:
        raise ValueError("min_class_support must be 0; label-conditioned candidate selection is excluded")
    if candidates < 2 or components < 8 or repeats < 1:
        raise ValueError("Require candidates>=2, components>=8, repeats>=1")
    if not bandwidths or any((not np.isfinite(s) or s <= 0) for s in bandwidths):
        raise ValueError("Bandwidths must be finite positive numbers")
    if not np.isfinite(disagreement_penalty) or disagreement_penalty < 0:
        raise ValueError("disagreement_penalty must be finite and nonnegative")
    _validate_frame(df, spec)
    features = _selected_features(df, spec)
    if not features:
        raise ValueError("No numeric features available")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError("All selected shift features must be numeric")

    splits: list[tuple[int, SplitResult]] = []
    for attempt in range(candidates):
        result = _split_group(df, spec, attempt)
        result.audit = audit_split(df, spec, result)
        if result.audit["valid"]:
            splits.append((attempt + 1, result))
    if len(splits) < 2:
        raise ValueError("Fewer than two feasible subject-disjoint partitions")

    X = _unlabeled_design_matrix(df, features)
    mapping_scores = []
    mapping_names = []
    # Use one fixed feature map for all candidates within each mapping.
    for b in bandwidths:
        for rep in range(repeats):
            seed = spec.seed + 10007 * rep  # deterministic independent RFF seeds
            z = _rff(X, bandwidth=float(b), components=components, seed=seed)
            values = []
            for _, split in splits:
                delta = z[split.train_idx].mean(axis=0) - z[split.test_idx].mean(axis=0)
                values.append(float(delta @ delta))
            mapping_scores.append(values)
            mapping_names.append(f"mmd_bw_{b:g}_repeat_{rep+1}")

    scores = np.array(mapping_scores, dtype=float).T  # candidates x mappings
    ranks = _rank_percentiles(scores)
    rank_mean = ranks.mean(axis=1)
    rank_sd = ranks.std(axis=1)
    consensus = rank_mean - disagreement_penalty * rank_sd
    best_idx = int(np.argmax(consensus))  # earliest candidate wins ties
    best_attempt, selected = splits[best_idx]
    selected.metadata.update({
        "method": "experimental_multi_kernel_rff_mmd_consensus",
        "selection_role": "development_only_diagnostic",
        "candidate_count_requested": candidates,
        "valid_candidate_count": len(splits),
        "selected_attempt": best_attempt,
        "score": float(consensus[best_idx]),
        "mean_rank_percentile": float(rank_mean[best_idx]),
        "rank_disagreement_sd": float(rank_sd[best_idx]),
        "rff_components_per_mapping": components,
        "bandwidths": list(bandwidths),
        "repeats": repeats,
        "disagreement_penalty": disagreement_penalty,
        "feature_count": len(features),
        "dataset_sha256": dataset_fingerprint(df),
        "seed": spec.seed,
        "warning": "Unlabeled feature-only selection using development data. Not an unbiased final test, a predictive-risk guarantee or a global optimum.",
    })
    rows = []
    for i, (attempt, split) in enumerate(splits):
        row = {
            "candidate": int(attempt),
            "train_rows": len(split.train_idx),
            "validation_rows": len(split.test_idx),
            "constraint_valid": bool(split.audit["valid"]),
            "mean_rank_percentile": float(rank_mean[i]),
            "rank_disagreement_sd": float(rank_sd[i]),
            "consensus_score": float(consensus[i]),
            "is_selected": bool(i == best_idx),
        }
        row.update({k: float(v) for k, v in zip(mapping_names, scores[i])})
        rows.append(row)
    frame = pd.DataFrame(rows)
    per_mapping_winners = [int(splits[int(np.argmax(scores[:, j]))][0])
                           for j in range(scores.shape[1])]
    meta = dict(selected.metadata)
    meta["per_mapping_winners"] = per_mapping_winners
    meta["mapping_count"] = scores.shape[1]
    return RobustShiftReport(selected=selected, candidates=frame, metadata=meta)
