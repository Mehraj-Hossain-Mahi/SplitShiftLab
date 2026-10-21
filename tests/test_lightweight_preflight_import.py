"""Preflight and package import should not eagerly load model-training libraries."""

import os
from pathlib import Path
import subprocess
import sys


def _run_fresh_python(snippet: str) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-c", snippet], env=env, cwd=root,
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_api_import_does_not_load_sklearn():
    _run_fresh_python(
        "from splitshiftlab import ProtocolSpec, SplitResult, audit_split, "
        "dataset_fingerprint, generate_split, shift_stress_split, evaluate_classifier; "
        "import sys; assert not any(m == 'sklearn' or m.startswith('sklearn.') "
        "for m in sys.modules), 'scikit-learn loaded during package import'"
    )


def test_preflight_cli_import_does_not_load_sklearn():
    _run_fresh_python(
        "from splitshiftlab.cli import main; "
        "from splitshiftlab.protocol_compiler import preflight_group_protocol; "
        "import sys; assert not any(m == 'sklearn' or m.startswith('sklearn.') "
        "for m in sys.modules), 'scikit-learn loaded during preflight CLI import'"
    )
