"""SplitShiftLab Step 2.9: Audit class-prior shift in UCI HAR development partitions.

Place in splitshiftlab_mvp/examples/ and run from the project root:
    python examples/uci_har_label_shift_audit.py

Only uses the official UCI HAR TRAINING-derived CSV and the Step 2.8 candidate
summary. Reconstructs the same 20 subject-disjoint candidate partitions and
measures class-prior shift AFTER partition selection; target labels are NOT
used to select candidate partitions. It makes no significance claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from splitshiftlab.core import ProtocolSpec, audit_split, dataset_fingerprint, generate_split

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT / 'out' / 'uci_har'


def as_python_float(x: float) -> float | None:
    value = float(x)
    return value if np.isfinite(value) else None


def spearman_or_none(x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique() <= 1 or y.nunique() <= 1:
        return None
    return as_python_float(spearmanr(x.to_numpy(), y.to_numpy()).statistic)


def analyze(data: pd.DataFrame, previous: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    data = data.reset_index(drop=True)
    labels = sorted(data['label'].dropna().unique().tolist())
    if not labels or data['label'].isna().any():
        raise ValueError('Expected complete target labels in DEVELOPMENT data')
    if len(previous) == 0:
        raise ValueError('Step 2.8 candidate audit has no rows')
    if previous['candidate'].duplicated().any():
        raise ValueError('Step 2.8 candidate indices are duplicated')
    results = []
    for i, old in previous.reset_index(drop=True).iterrows():
        expected_i, candidate_seed = i + 1, seed + i
        if int(old['candidate']) != expected_i or int(old['seed']) != candidate_seed:
            raise ValueError('Candidate or seed ordering mismatch with Step 2.8')
        spec = ProtocolSpec(target='label', mode='group', group_col='subject',
                            seed=candidate_seed, test_size=0.25)
        split = generate_split(data, spec)
        if split.metadata.get('attempt') != 1:
            raise ValueError('Candidate required internal retries; cannot align with Step 2.8')
        audit = audit_split(data, spec, split)
        if not audit['valid']:
            raise AssertionError('Candidate violates split constraints')
        if len(split.train_idx) != int(old['train_rows']) or len(split.test_idx) != int(old['validation_rows']):
            raise ValueError(f'Candidate {expected_i} row counts do not match Step 2.8')
        train_ids = set(data.iloc[split.train_idx]['subject'])
        validation_ids = set(data.iloc[split.test_idx]['subject'])
        if train_ids & validation_ids:
            raise AssertionError('Subject overlap detected')
        a = data.iloc[split.train_idx]['label'].value_counts().reindex(labels, fill_value=0)
        b = data.iloc[split.test_idx]['label'].value_counts().reindex(labels, fill_value=0)
        p = a.to_numpy(dtype=float) / a.sum()
        q = b.to_numpy(dtype=float) / b.sum()
        tv = float(0.5 * np.abs(p - q).sum())
        js2 = float(jensenshannon(p, q, base=2.0) ** 2)
        max_gap = float(np.max(np.abs(p - q)))
        rec = dict(old.to_dict())
        rec.update({
            'label_total_variation': tv,
            'label_js_divergence_bits': js2,
            'largest_class_share_gap': max_gap,
            'minimum_train_class_count': int(a.min()),
            'minimum_validation_class_count': int(b.min()),
            'missing_validation_classes': int((b == 0).sum()),
            'missing_train_classes': int((a == 0).sum()),
        })
        for label, tr_count, te_count, tr_p, te_p in zip(labels, a, b, p, q):
            key = str(label).replace(' ', '_')
            rec[f'label_{key}_train_count'] = int(tr_count)
            rec[f'label_{key}_validation_count'] = int(te_count)
            rec[f'label_{key}_train_share'] = float(tr_p)
            rec[f'label_{key}_validation_share'] = float(te_p)
        results.append(rec)
    result = pd.DataFrame(results)
    error = 1 - result['accuracy']
    highest_shift_row = result.loc[result['mmd_squared'].idxmax()]
    most_label_shift_row = result.loc[result['label_total_variation'].idxmax()]
    hardest_row = result.loc[result['accuracy'].idxmin()]
    summary = {
        'study': 'Development-only, post-selection class-prior diagnostics (Step 2.9)',
        'official_test_used': False,
        'dataset_sha256': dataset_fingerprint(data),
        'candidate_count': int(len(result)),
        'labels': [str(x) for x in labels],
        'all_subject_disjoint': True,
        'all_classes_in_both_partitions': bool((result['missing_train_classes'].eq(0) & result['missing_validation_classes'].eq(0)).all()),
        'max_mmd_candidate': int(highest_shift_row['candidate']),
        'max_mmd_label_tv': float(highest_shift_row['label_total_variation']),
        'max_label_tv_candidate': int(most_label_shift_row['candidate']),
        'max_label_tv': float(most_label_shift_row['label_total_variation']),
        'hardest_accuracy_candidate': int(hardest_row['candidate']),
        'spearman_mmd_vs_error': spearman_or_none(result['mmd_squared'], error),
        'spearman_label_tv_vs_error': spearman_or_none(result['label_total_variation'], error),
        'spearman_mmd_vs_label_tv': spearman_or_none(result['mmd_squared'], result['label_total_variation']),
        'cautions': [
            'Label distributions were computed AFTER candidate selection, for retrospective diagnosis only.',
            'Do not use target labels to optimize a held-out evaluation split or tune the final test set.',
            'Candidate partitions share observations; these correlations are descriptive, not significance tests.',
            'Class-prior imbalance may contribute to performance variation but this audit does not establish causation.',
            'SplitShiftLab v0.1 selects the maximum MMD among enumerated candidates, not a global optimum.',
        ],
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_OUT / 'uci_har_official_train.csv')
    parser.add_argument('--candidate-csv', type=Path, default=DEFAULT_OUT / 'uci_har_candidate_audit.csv')
    parser.add_argument('--summary-json', type=Path, default=DEFAULT_OUT / 'uci_har_candidate_audit_summary.json')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for name, path in [('development CSV', args.csv), ('Step 2.8 CSV', args.candidate_csv), ('Step 2.8 summary', args.summary_json)]:
        if not path.is_file():
            parser.error(f'{name} missing: {path}')
    if 'official_test' in str(args.csv).lower():
        parser.error('Refusing to read the official test dataset')
    data = pd.read_csv(args.csv)
    prior = pd.read_csv(args.candidate_csv)
    details = json.loads(args.summary_json.read_text(encoding='utf-8'))
    if details.get('official_test_used') is not False:
        parser.error('Step 2.8 audit does not certify development-only evaluation')
    if details.get('dataset_sha256') != dataset_fingerprint(data):
        parser.error('Development dataset fingerprint differs from Step 2.8')
    if int(details['candidate_count']) != len(prior):
        parser.error('Candidate CSV count differs from Step 2.8 summary')
    result, summary = analyze(data, prior, seed=int(details['seed']))
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / 'uci_har_label_shift_audit.csv'
    json_path = args.out / 'uci_har_label_shift_audit_summary.json'
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('\nSTEP 2.9 LABEL-SHIFT AUDIT COMPLETED')
    print(f"Group-disjoint candidates: {summary['candidate_count']} / {summary['candidate_count']}")
    print(f"Classes represented in every train + validation partition: {summary['all_classes_in_both_partitions']}")
    print(f"Highest feature-MMD candidate: {summary['max_mmd_candidate']}")
    print(f"Largest label-prior change candidate: {summary['max_label_tv_candidate']}")
    print(f"Lowest accuracy candidate: {summary['hardest_accuracy_candidate']}")
    for k in ('spearman_mmd_vs_error','spearman_label_tv_vs_error','spearman_mmd_vs_label_tv'):
        v = summary[k]
        print(f'{k}: {v:.3f}' if v is not None else f'{k}: undefined')
    print('OFFICIAL TEST NOT USED. LABELS USED FOR POST-SELECTION DIAGNOSTIC ONLY.')
    print('Saved:', csv_path)
    print('Saved:', json_path)


if __name__ == '__main__':
    main()
