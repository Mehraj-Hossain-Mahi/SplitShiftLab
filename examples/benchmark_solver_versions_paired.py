"""Step 2.42: paired old-vs-batched SplitShiftLab class-support benchmark.

Neither version trains a classifier, nor accesses the protected official UCI HAR
TEST data. Both versions run in one Python interpreter against one loaded DataFrame.
The reference Step 2.32 solver lives in examples/legacy_protocol_compiler_step232.py;
this script does not replace or modify the installed SplitShiftLab package.

The measured function includes dataset hashing, validation, row-DP and exact
class support checks. Do NOT treat these as isolated class-search loop timings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol as batched_preflight

EXPECTED_UCI_SHA256 = 'd568f9e01725fb5d9f45aa5bec3b08312293e18f05ca55aa9cfa54813cc623ff'
EXPECTED_WITNESS = sorted([
    'subject_01', 'subject_03', 'subject_23',
    'subject_25', 'subject_26', 'subject_30',
])


def load_previous_solver(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Previous solver source missing: {path}')
    # The module name is inside the installed SplitShiftLab package so the
    # archived source's relative ".core" import resolves correctly.
    name = 'splitshiftlab._benchmark_legacy_protocol_compiler_step232'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load previous solver')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module.preflight_group_protocol


def describe(values: list[float]) -> dict:
    return {
        'runs': len(values),
        'mean_seconds': round(statistics.mean(values), 6),
        'median_seconds': round(statistics.median(values), 6),
        'stdev_seconds': round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        'min_seconds': round(min(values), 6),
        'max_seconds': round(max(values), 6),
    }


def validate_report(report: dict, threshold: int, *, smoke: bool):
    expected_status = ('feasible' if threshold == (1 if smoke else 322)
                       else 'infeasible')
    if report['status'] != expected_status:
        raise RuntimeError(f'Threshold {threshold}: status {report["status"]!r} '
                           f'but expected {expected_status!r}')
    if report['certificate'] is None:
        if not report.get('class_support_search_exhausted'):
            raise RuntimeError('Infeasible report lacks complete class-support search')
        if not smoke and report.get('class_support_allocations_examined') != 54264:
            raise RuntimeError('Threshold 323: expected 54,264 allocations examined')
    else:
        c = report['certificate']
        if c['subject_overlap'] != 0:
            raise RuntimeError('A reported certificate contains overlapping subjects')
        if min(c['per_class_train_counts'].values()) < threshold or \
           min(c['per_class_validation_counts'].values()) < threshold:
            raise RuntimeError('A reported certificate violates minimum class support')
        if not smoke:
            if sorted(c['validation_group_ids']) != EXPECTED_WITNESS:
                raise RuntimeError('Threshold 322: certificate differs from exhaustive reference')
            if c['training_rows'] != 5108 or c['validation_rows'] != 2244:
                raise RuntimeError('Threshold 322: certificate has wrong row counts')
            if report.get('class_support_allocations_examined') != 3845:
                raise RuntimeError('Threshold 322: expected 3,845 allocations examined')
    return (
        report['status'],
        report.get('class_support_allocations_examined'),
        report.get('class_support_search_exhausted'),
        None if report['certificate'] is None else {
            'groups': sorted(report['certificate']['validation_group_ids']),
            'train_counts': report['certificate']['per_class_train_counts'],
            'validation_counts': report['certificate']['per_class_validation_counts'],
            'train_rows': report['certificate']['training_rows'],
            'validation_rows': report['certificate']['validation_rows'],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        default=Path('out/uci_har/uci_har_official_train.csv'))
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path,
                        default=Path('out/uci_har/benchmark_solver_versions_paired.json'))
    parser.add_argument('--synthetic-smoke', action='store_true',
                        help='Development-only smoke check on a synthetic 21-group CSV')
    args = parser.parse_args()
    if args.runs < 2 or args.warmups < 0:
        parser.error('Use --runs >= 2 and --warmups >= 0')
    if not args.csv.is_file():
        parser.error(f'CSV missing: {args.csv}')
    previous_path = Path(__file__).resolve().parent / 'legacy_protocol_compiler_step232.py'
    old_solver = load_previous_solver(previous_path)
    solvers = {'previous': old_solver, 'batched': batched_preflight}

    df = pd.read_csv(args.csv)
    if not {'subject', 'label'}.issubset(df.columns):
        parser.error('Dataset must contain subject and label columns')
    if len(df) != 7352 or df['subject'].nunique() != 21 or df['label'].nunique() != 6:
        parser.error('Expected 7,352 rows, 21 subjects and six labels')
    # The synthetic mode is used only for testing that the benchmark script works.
    thresholds = [1, 400] if args.synthetic_smoke else [322, 323]
    specs = {x: ProtocolSpec(target='label', mode='group', group_col='subject',
                             test_size=0.25, seed=42, min_train_rows=2,
                             min_test_rows=2, min_class_support=x)
             for x in thresholds}
    original_hash = None

    def execute(version, threshold):
        nonlocal original_hash
        t0 = perf_counter()
        report = solvers[version](df, specs[threshold])
        elapsed = perf_counter() - t0
        signature = validate_report(report, threshold, smoke=args.synthetic_smoke)
        if original_hash is None:
            original_hash = report['dataset_sha256']
            if not args.synthetic_smoke and original_hash != EXPECTED_UCI_SHA256:
                raise RuntimeError('Development dataset fingerprint does not match the established UCI HAR reference')
        elif report['dataset_sha256'] != original_hash:
            raise RuntimeError('Data fingerprint differs across runs')
        return elapsed, signature

    print(f'Loaded CSV once: {len(df)} rows, {df["subject"].nunique()} subjects')
    print(f'Previous source: {previous_path.name}')
    print('Compare both solvers in one Python process (no installed module is replaced).')
    print(f'Warm-up rounds: {args.warmups}; timed rounds: {args.runs}')
    for round_idx in range(args.warmups):
        for threshold in thresholds:
            for version in ('previous', 'batched'):
                execute(version, threshold)

    records = []
    for round_idx in range(1, args.runs + 1):
        # Alternating threshold and solver order mitigates, but cannot remove,
        # carryover and environmental scheduling effects.
        order_thresholds = thresholds if round_idx % 2 else list(reversed(thresholds))
        order_versions = ('previous', 'batched') if round_idx % 2 else ('batched', 'previous')
        for threshold in order_thresholds:
            observed = {}
            for version in order_versions:
                elapsed, signature = execute(version, threshold)
                observed[version] = signature
                records.append({
                    'round': round_idx,
                    'minimum_class_support': threshold,
                    'solver': version,
                    'seconds': round(elapsed, 7),
                    'status': signature[0],
                    'allocations_examined': signature[1],
                    'search_exhausted': signature[2],
                })
            if observed['previous'] != observed['batched']:
                raise RuntimeError(f'Old/new solver results disagree at threshold {threshold}')
        print(f'Round {round_idx}/{args.runs} completed; old/new correctness matched')

    statistics_by_threshold = {}
    for threshold in thresholds:
        subset = {version: [r['seconds'] for r in records
                             if r['minimum_class_support'] == threshold and r['solver'] == version]
                  for version in solvers}
        previous = describe(subset['previous'])
        batched = describe(subset['batched'])
        differences = [old - new for old, new in zip(subset['previous'], subset['batched'])]
        statistics_by_threshold[str(threshold)] = {
            'previous': previous,
            'batched': batched,
            'paired_differences_seconds_previous_minus_batched': describe(differences),
            'batched_faster_in_pairs': sum(old > new for old, new in zip(subset['previous'], subset['batched'])),
            'median_speedup_previous_over_batched': round(previous['median_seconds'] / batched['median_seconds'], 4),
        }
        print(f'Support {threshold}: previous median={previous["median_seconds"]:.4f}s; '
              f'batched median={batched["median_seconds"]:.4f}s; '
              f'batched faster in {statistics_by_threshold[str(threshold)]["batched_faster_in_pairs"]}/{args.runs} pairs')

    result = {
        'study': 'Step 2.42 paired previous-vs-batched preflight benchmark',
        'scope': 'preflight_group_protocol excluding startup and CSV loading; includes hashing, DP and class search',
        'development_only': True,
        'official_test_used': False,
        'csv': str(args.csv),
        'dataset_sha256': original_hash,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'previous_solver_source': str(previous_path),
        'previous_solver_sha256': hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        'batched_solver_source': batched_preflight.__code__.co_filename,
        'batched_solver_sha256': hashlib.sha256(Path(batched_preflight.__code__.co_filename).read_bytes()).hexdigest(),
        'runs_per_threshold_and_version': args.runs,
        'warmup_rounds': args.warmups,
        'summary': statistics_by_threshold,
        'records': records,
        'limitations': [
            'Each call includes dataset fingerprinting and row-feasibility DP, not just class-support enumeration.',
            'This is a single-machine exploratory benchmark, not a cross-platform performance study.',
            'Alternating paired execution mitigates but does not eliminate thermal, cache or CPU scheduling effects.',
            'No statistical significance claim is made; test set is not accessed.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    output_csv = args.output.with_suffix('.csv')
    with output_csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Old/new feasibility and witness checks: PASSED')
    print('Official test dataset: NOT ACCESSED')
    print(f'JSON saved: {args.output}')
    print(f'CSV saved: {output_csv}')


if __name__ == '__main__':
    main()
