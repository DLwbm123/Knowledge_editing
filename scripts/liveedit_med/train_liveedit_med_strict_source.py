#!/usr/bin/env python3
"""Strict-source entry point; refuses continuation-mode drift."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode
from scripts.liveedit_med.train_liveedit_med_v4_source import main


if __name__ == "__main__":
    expected = SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21.value
    flag = "--source-training-continuation-mode"
    if flag in sys.argv:
        value = sys.argv[sys.argv.index(flag) + 1]
        if value != expected:
            raise RuntimeError(f"LIVEEDIT_MED_STRICT_ENTRYPOINT_MODE_DRIFT:{value}")
    else:
        sys.argv.extend([flag, expected])
    main()
