#!/usr/bin/env python3
"""Strict-source Stage-A entry point with a non-overridable mode."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode
from scripts.liveedit_med.run_upstream_end_to_end_trace_parity import main


if __name__ == "__main__":
    expected = SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21.value
    flag = "--port-continuation-mode"
    if flag in sys.argv:
        value = sys.argv[sys.argv.index(flag) + 1]
        if value != expected:
            raise RuntimeError(f"LIVEEDIT_MED_STRICT_STAGE_A_MODE_DRIFT:{value}")
    else:
        sys.argv.extend([flag, expected])
    main()
