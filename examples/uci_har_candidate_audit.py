"""Step 2.8: Audit all candidate partitions for one UCI HAR DEVELOPMENT seed.

Copy to splitshiftlab_mvp/examples/uci_har_candidate_audit.py and run:
    python examples/uci_har_candidate_audit.py

Uses ONLY the official training-derived CSV. Reproduces SplitShiftLab v0.1's
RFF MMD scoring independently, checks that its selected split is the true
maximum in the candidate pool, and tests how MMD relates to classification
error across candidate splits. This is diagnostic, not a significance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from splitshiftlab.core import (
    ProtocolSpec, audit_split, dataset_fingerprint,
    evaluate_classifier, generate_split, shift_stress_split,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / 'out' / 'uci_har' / 'uci_har_official_train.csv'
DEFAULT_OUT = ROOT / 'out' / 'uci_har'


def independent_rff_embedding(df: pd.DataFrame, features: list[str], seed: int,
                              components: int) -> np.ndarray:
    """Independently reproduce v0.1's full-DEVELOPMENT feature-only embedding.

    This is a diagnostic split-ranking score, NOT train-fitted preprocessing
    for the downstream classifier (which uses a separate training-only pipe).
    """
    X = df[features].apply(pd.to_numeric, errors='raise').replace(
        [np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        raise ValueError('All-missing input feature column')
    arr = X.fillna(X.median()).to_numpy(dtype=float)
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    sd[sd < 1e-12] = 1
    arr = (arr - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0 / np.sqrt(max(1, arr.shape[1])),
                   (arr.shape[1], components))
    b = rng.uniform(0, 2 * np.pi, components)
    return np.sqrt(2 / components) * np.cos(arr @ w + b)


def analyze(data: pd.DataFrame, seed: int = 42, candidates: int = 20,
            components: int = 256) -> tuple[pd.DataFrame, dict]:
    if candidates < 3 or components < 8:
        raise ValueError('Need >=3 candidates and >=8 RFF components')
    data = data.reset_index(drop=True)
    if not {'subject', 'label'}.issubset(data.columns):
        raise ValueError('Expected subject and label columns')
    features = [c for c in data.columns if c.startswith('feature_')]
    if not features:
        raise ValueError('Expected feature_001 ... feature_561 columns')
    spec = ProtocolSpec(target='label', mode='group', group_col='subject',
                        features=tuple(features), seed=seed, test_size=0.25)
    z = independent_rff_embedding(data, features, seed, components)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          RidgeClassifier(alpha=1.0))
    records = []
    splits = []
    for i in range(candidates):
        cand_spec = ProtocolSpec(target='label', mode='group', group_col='subject',
                                 features=tuple(features), seed=seed + i,
                                 test_size=0.25)
        split = generate_split(data, cand_spec)
        # This v0.1 candidate enumerator assumes the first attempt is feasible.
        if split.metadata.get('attempt') != 1:
            raise RuntimeError(f'Attempt {i+1} required internal retries. '
                               'Candidate-pool equivalence cannot be guaranteed.')
        audit = audit_split(data, cand_spec, split)
        if not audit['valid']:
            raise AssertionError(f'Invalid candidate: {audit["violations"]}')
        if set(data.iloc[split.train_idx]['subject']) & set(data.iloc[split.test_idx]['subject']):
            raise AssertionError('Subject leakage detected')
        mmd = float(np.sum((z[split.train_idx].mean(axis=0) -
                            z[split.test_idx].mean(axis=0)) ** 2))
        scores = evaluate_classifier(data, cand_spec, split, model)
        record = {
            'candidate': i + 1,
            'seed': seed + i,
            'mmd_squared': mmd,
            'train_rows': len(split.train_idx),
            'validation_rows': len(split.test_idx),
            'train_subjects': data.iloc[split.train_idx]['subject'].nunique(),
            'validation_subjects': data.iloc[split.test_idx]['subject'].nunique(),
            'subject_overlap': 0,
            **scores,
        }
        records.append(record)
        splits.append(split)
        print(f"Candidate {i+1:2d}/{candidates} | MMD²={mmd:.5f} | "
              f"Acc={scores['accuracy']:.2%} | overlap=0", flush=True)
    table = pd.DataFrame.from_records(records)
    selected = int(np.argmax(table['mmd_squared'].to_numpy()))
    worst_acc = int(np.argmin(table['accuracy'].to_numpy()))
    optimizer = shift_stress_split(data, spec, candidates=candidates, components=components)
    if optimizer.metadata['valid_candidate_count'] != candidates:
        raise AssertionError('Optimizer did not accept every candidate')
    if optimizer.metadata['selected_attempt'] != selected + 1:
        raise AssertionError('Independent maximum-MMD candidate disagrees with optimizer')
    if list(optimizer.test_idx) != list(splits[selected].test_idx):
        raise AssertionError('Optimizer candidate indices differ')
    if not np.isclose(optimizer.metadata['approx_mmd_squared'],
                      table.iloc[selected]['mmd_squared'], atol=1e-10, rtol=1e-9):
        raise AssertionError('Independent MMD score differs from optimizer')
    rho = float(spearmanr(table['mmd_squared'], 1-table['accuracy']).statistic)
    if not np.isfinite(rho):
        rho = None
    summary = {
        'study': 'Candidate-level development-only diagnostic; NOT a significance test',
        'official_test_used': False,
        'dataset_sha256': dataset_fingerprint(data),
        'package_version': '0.1.0',
        'python_version': platform.python_version(),
        'scikit_learn_version': sklearn.__version__,
        'seed': seed,
        'candidate_count': candidates,
        'rff_components': components,
        'classifier': 'SimpleImputer -> StandardScaler -> RidgeClassifier(alpha=1.0)',
        'optimizer_matches_independent_maximum': True,
        'selected_candidate': selected + 1,
        'selected_mmd_squared': float(table.iloc[selected]['mmd_squared']),
        'mean_candidate_mmd_squared': float(table['mmd_squared'].mean()),
        'selected_accuracy': float(table.iloc[selected]['accuracy']),
        'hardest_accuracy_candidate': worst_acc + 1,
        'hardest_candidate_accuracy': float(table.iloc[worst_acc]['accuracy']),
        'selected_is_lowest_accuracy': bool(selected == worst_acc),
        'spearman_mmd_vs_classification_error': rho,
        'cautions': [
            'All candidates share data, participants and a search pool; observations are dependent.',
            'This is a diagnostic association, not statistical significance or causality.',
            'Larger feature MMD does not necessarily produce greater predictive error.',
            'Shift ranking uses unlabeled development features; final UCI HAR test remains untouched.',
            'The v0.1 optimizer picks the highest of the enumerated random candidates; '
            'this is search, not a novel global optimization algorithm.',
        ],
    }
    return table, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--candidates', type=int, default=20)
    p.add_argument('--components', type=int, default=256)
    args = p.parse_args()
    if not args.csv.is_file():
        p.error(f'CSV not found: {args.csv}')
    if 'official_test' in args.csv.name.lower():
        p.error('Refusing to run on official test data')
    data = pd.read_csv(args.csv)
    if args.csv.resolve() == DEFAULT_CSV.resolve():
        if len(data) != 7352 or data.subject.nunique() != 21:
            p.error('Unexpected official UCI HAR training set size/subjects')
        if len([c for c in data if c.startswith('feature_')]) != 561:
            p.error('Expected 561 numeric feature columns')
    print(f'DEVELOPMENT ONLY: {len(data)} rows; {data.subject.nunique()} subjects', flush=True)
    with threadpool_limits(limits=2):
        table, summary = analyze(data, seed=args.seed, candidates=args.candidates,
                                 components=args.components)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / 'uci_har_candidate_audit.csv'
    json_path = args.out / 'uci_har_candidate_audit_summary.json'
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('\nCANDIDATE-LEVEL AUDIT COMPLETED')
    print('Independent maximum MMD matches SplitShiftLab: YES')
    print(f"Highest-MMD candidate: {summary['selected_candidate']} "
          f"(MMD²={summary['selected_mmd_squared']:.5f}, "
          f"accuracy={summary['selected_accuracy']:.2%})")
    print(f"Lowest-accuracy candidate: {summary['hardest_accuracy_candidate']} "
          f"(accuracy={summary['hardest_candidate_accuracy']:.2%})")
    print(f"Highest-MMD candidate also lowest-accuracy? {summary['selected_is_lowest_accuracy']}")
    rho = summary['spearman_mmd_vs_classification_error']
    print(f'Spearman MMD vs error: {rho:.3f}' if rho is not None else
          'Spearman MMD vs error: undefined (constant scores)')
    print('Official test set NOT used. No significance claims.')
    print('Saved:', csv_path)
    print('Saved:', json_path)


if __name__ == '__main__':
    main()
