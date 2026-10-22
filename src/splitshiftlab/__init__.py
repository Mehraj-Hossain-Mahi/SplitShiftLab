"""SplitShiftLab: auditable data partitioning and distribution-shift diagnostics.

Research prototype (v0.1.0). Not a comprehensive leakage detector.
"""

from .core import (ProtocolSpec, SplitResult, audit_split, dataset_fingerprint,
                   generate_split, shift_stress_split, evaluate_classifier)

__version__ = "0.1.1"
__all__ = [
    "ProtocolSpec", "SplitResult", "audit_split", "dataset_fingerprint",
    "generate_split", "shift_stress_split", "evaluate_classifier",
]
