"""Step 2.13: development-only protocol preflight and infeasibility demo."""

from pathlib import Path
import json

import pandas as pd

from splitshiftlab.core import ProtocolSpec
from splitshiftlab.protocol_compiler import preflight_group_protocol


DATA = Path('out/uci_har/uci_har_official_train.csv')
OUTPUT = Path('out/uci_har/uci_har_preflight_report.json')


def main():
    if not DATA.is_file():
        raise SystemExit(f"Missing prepared UCI HAR DEVELOPMENT file: {DATA}. Run Step 2.2 first.")
    df = pd.read_csv(DATA)
    print(f"UCI HAR development records: {len(df)} | subjects: {df['subject'].nunique()}")
    valid = preflight_group_protocol(df, ProtocolSpec(
        target='label', mode='group', group_col='subject',
        test_size=0.25, min_train_rows=2, min_test_rows=2,
    ))
    impossible = preflight_group_protocol(df, ProtocolSpec(
        target='label', mode='group', group_col='subject',
        test_size=0.25, min_train_rows=2, min_test_rows=len(df) - 2,
    ))
    if valid['status'] != 'feasible' or impossible['status'] != 'infeasible':
        raise RuntimeError('Preflight did not return expected feasibility statuses.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'study': 'Step 2.13 constrained-group preflight; development only',
        'official_test_used': False,
        'feasible_case': valid,
        'infeasible_case': impossible,
    }, indent=2) + '\n', encoding='utf-8')
    cert = valid['certificate']
    print(f"FEASIBLE: train={cert['training_rows']} rows / {cert['training_groups']} subjects; "
          f"validation={cert['validation_rows']} rows / {cert['validation_groups']} subjects")
    print(f"INFEASIBLE: {impossible['blockers'][0]['code']}")
    print('Development-only preflight complete. Official test file NOT opened.')
    print(f'Saved: {OUTPUT}')


if __name__ == '__main__':
    main()
