"""Convert official UCI HAR feature tables to reproducible CSV files.

Run from the splitshiftlab_mvp project root:
    python examples/prepare_uci_har.py

The original official train/test partition is preserved in separate files.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'uci_har' / 'UCI HAR Dataset'
OUTPUT = ROOT / 'out' / 'uci_har'


def load_partition(name: str) -> pd.DataFrame:
    folder = DATA / name
    files = [folder / f'X_{name}.txt', folder / f'y_{name}.txt', folder / f'subject_{name}.txt']
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing official UCI HAR files: {missing}')

    x = pd.read_csv(files[0], sep=r'\s+', header=None)
    y = pd.read_csv(files[1], sep=r'\s+', header=None)
    subjects = pd.read_csv(files[2], sep=r'\s+', header=None)
    if not (len(x) == len(y) == len(subjects)):
        raise ValueError(f'Row counts mismatch in {name}')
    if x.shape[1] != 561:
        raise ValueError(f'Expected 561 features, found {x.shape[1]} in {name}')
    if y[0].isna().any() or subjects[0].isna().any():
        raise ValueError(f'Missing class/subject in {name}')

    # Stable unique field names avoid duplicate official feature-name collisions.
    x.columns = [f'feature_{i:03d}' for i in range(1, x.shape[1] + 1)]
    x.insert(0, 'subject', subjects[0].astype(int).map(lambda v: f'subject_{v:02d}'))
    x.insert(1, 'label', y[0].astype(int))
    return x


def main():
    train = load_partition('train')
    test = load_partition('test')
    overlap = set(train['subject']) & set(test['subject'])
    if overlap:
        raise ValueError(f'Official partitions have overlapping subjects: {sorted(overlap)}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT / 'uci_har_official_train.csv'
    test_path = OUTPUT / 'uci_har_official_test.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f'Official TRAIN: {len(train):,} rows, {train.subject.nunique()} subjects -> {train_path}')
    print(f'Official TEST:  {len(test):,} rows, {test.subject.nunique()} subjects -> {test_path}')
    print(f'Feature columns: {len(train.columns) - 2}; Subject overlap: {len(overlap)}')
    print('Official test file has NOT been used for development or stress partition search.')


if __name__ == '__main__':
    main()
