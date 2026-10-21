# Worked examples

1. `python -m splitshiftlab demo` creates demo records and a training/testing manifest.
2. `python -m splitshiftlab demo --stress` uses unlabeled numeric feature shifts to choose a challenging group holdout.
3. Try `--mode group_temporal --group subject --time event_time` against the generated CSV to enforce both entity and time requirements.
4. Check `out/demo_manifest.json` for explicit `train_idx`, `test_idx`, `audit`, and dataset fingerprint.

The demonstration dataset is generated and therefore not a real-world benchmark.
You need external benchmark case studies before a scientific paper is ready.
