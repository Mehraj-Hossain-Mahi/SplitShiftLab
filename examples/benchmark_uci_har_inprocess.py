"""Step 2.41: reproducible in-process UCI HAR SplitShiftLab preflight benchmark.

Measures preflight_group_protocol() after a one-time import and CSV load.
Each timed call STILL includes dataset fingerprinting, input validation,
row-feasibility dynamic programming, and class-support allocation search;
this is NOT a microbenchmark of the enumeration kernel alone.
The official UCI HAR test set is never used.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol


def describe_seconds(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError('No timing results to summarize')
    return {
        'n': len(values),
        'mean_seconds': round(statistics.mean(values), 6),
        'median_seconds': round(statistics.median(values), 6),
        'stdev_seconds': round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        'min_seconds': round(min(values), 6),
        'max_seconds': round(max(values), 6),
    }


def run_preflight(df: pd.DataFrame, threshold: int) -> dict:
    return preflight_group_protocol(
        df,
        ProtocolSpec(
            target='label',
            mode='group',
            group_col='subject',
            test_size=0.25,
            seed=42,
            min_train_rows=2,
            min_test_rows=2,
            min_class_support=threshold,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        default=Path('out/uci_har/uci_har_official_train.csv'))
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--feasible-support', type=int, default=322)
    parser.add_argument('--infeasible-support', type=int, default=323)
    parser.add_argument('--output', type=Path,
                        default=Path('out/uci_har/benchmark_inprocess_preflight.json'))
    args = parser.parse_args()
    if args.runs < 2 or args.warmups < 0:
        parser.error('--runs must be >= 2 and --warmups must be >= 0')
    if args.feasible_support < 1 or args.infeasible_support < 1:
        parser.error('Class-support thresholds must be positive')
    if args.feasible_support == args.infeasible_support:
        parser.error('The two class-support thresholds must be distinct')
    if not args.csv.is_file():
        parser.error(f'Missing development CSV: {args.csv}')

    # CSV loading happens exactly once, outside all timed calls.
    df = pd.read_csv(args.csv)
    if not {'label', 'subject'}.issubset(df.columns):
        parser.error('CSV requires label and subject columns')
    if len(df) != 7352 or df['subject'].nunique() != 21 or df['label'].nunique() != 6:
        parser.error('Expected UCI HAR official TRAIN data: 7352 rows, 21 subjects, 6 labels')

    thresholds = [args.feasible_support, args.infeasible_support]
    expected_status = {
        args.feasible_support: 'feasible',
        args.infeasible_support: 'infeasible',
    }
    expected_allocations = {
        args.feasible_support: 3845,
        args.infeasible_support: 54264,
    } if (args.feasible_support, args.infeasible_support) == (322, 323) else {}
    expected_subjects = sorted([
        'subject_01', 'subject_03', 'subject_23',
        'subject_25', 'subject_26', 'subject_30',
    ]) if (args.feasible_support, args.infeasible_support) == (322, 323) else None

    def verify(threshold: int, report: dict) -> None:
        status = report['status']
        if status != expected_status[threshold]:
            raise RuntimeError(f'Unexpected status at threshold {threshold}: {status!r}; '
                               f'expected {expected_status[threshold]!r}. Benchmark aborted.')
        if threshold in expected_allocations:
            observed = report.get('class_support_allocations_examined')
            if observed != expected_allocations[threshold]:
                raise RuntimeError(f'Unexpected allocation count at threshold {threshold}: '
                                   f'{observed} != {expected_allocations[threshold]}')
        if status == 'feasible':
            certificate = report['certificate']
            if not certificate or certificate['subject_overlap'] != 0:
                raise RuntimeError('Missing or overlapping feasibility certificate')
            if expected_subjects is not None and sorted(certificate['validation_group_ids']) != expected_subjects:
                raise RuntimeError('Feasible allocation differs from independent exhaustive reference')
            if (min(certificate['per_class_train_counts'].values()) < threshold or
                    min(certificate['per_class_validation_counts'].values()) < threshold):
                raise RuntimeError('Certificate violates requested class support')
        elif status == 'infeasible':
            if report.get('certificate') is not None or not report.get('class_support_search_exhausted'):
                raise RuntimeError('Infeasible status lacks exhausted search or has certificate')

    print(f'Loaded development data once: {len(df)} rows, '
          f'{df["subject"].nunique()} subjects, {df["label"].nunique()} classes')
    print(f'Warming up: {args.warmups} paired repetitions (not timed)')
    for _ in range(args.warmups):
        for threshold in thresholds:
            verify(threshold, run_preflight(df, threshold))

    print(f'Running {args.runs} alternating paired repetitions per threshold...')
    records: list[dict] = []
    first_hash = None
    for run in range(1, args.runs + 1):
        order = thresholds if run % 2 == 1 else list(reversed(thresholds))
        for threshold in order:
            t0 = perf_counter()
            report = run_preflight(df, threshold)
            elapsed = perf_counter() - t0
            verify(threshold, report)
            data_hash = report['dataset_sha256']
            if first_hash is None:
                first_hash = data_hash
            elif first_hash != data_hash:
                raise RuntimeError('Dataset fingerprint changed during benchmark')
            records.append({
                'run': run,
                'minimum_class_support': threshold,
                'seconds': round(elapsed, 7),
                'status': report['status'],
                'allocations_examined': report.get('class_support_allocations_examined'),
                'search_exhausted': report.get('class_support_search_exhausted'),
            })
        print(f'  Pair {run}/{args.runs} complete')

    summaries = {
        str(threshold): describe_seconds([
            row['seconds'] for row in records
            if row['minimum_class_support'] == threshold
        ]) for threshold in thresholds
    }
    result = {
        'study': 'Step 2.41 in-process preflight benchmark',
        'scope': 'preflight_group_protocol including hashing and row DP; excludes startup and CSV load',
        'official_test_used': False,
        'csv': str(args.csv),
        'development_rows': len(df),
        'dataset_sha256': first_hash,
        'subjects': int(df['subject'].nunique()),
        'classes': int(df['label'].nunique()),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'warmup_pairs': args.warmups,
        'timed_pairs': args.runs,
        'timed_executions': len(records),
        'summary': summaries,
        'records': records,
        'limitations': [
            'Only the currently installed SplitShiftLab version is measured.',
            'No baseline-versus-new solver comparison is performed.',
            'Timings include fingerprinting and row-feasibility DP, not only class enumeration.',
            'Run-time variability may reflect CPU scheduling and environmental effects.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    csv_output = args.output.with_suffix('.csv')
    with csv_output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    for threshold in thresholds:
        s = summaries[str(threshold)]
        print(f'Support {threshold}: mean={s["mean_seconds"]:.4f}s; '
              f'median={s["median_seconds"]:.4f}s; '
              f'sdev={s["stdev_seconds"]:.4f}s; n={s["n"]}')
    print('Correctness checks: PASSED')
    print('Official test dataset: NOT ACCESSED')
    print(f'JSON saved: {args.output}')
    print(f'CSV saved: {csv_output}')


if __name__ == '__main__':
    main()
