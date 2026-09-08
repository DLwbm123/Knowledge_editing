#!/usr/bin/env python3
"""Stage3 allowlist using the existing bounded, guarded local public delivery."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.medtrace import closeout_stage2_local as delivery

delivery.RELDIR = 'reports/medtrace_stage3_20260908'
delivery.REPORTS = ('RUN_MANIFEST.json', 'SINGLE_EVAL_RESULTS.csv', 'EXPERT_BANK16_RESULTS.csv',
    'EXECUTION_STATUS.json', 'GPT_PRO_REVIEW.md', 'SINGLE_PAIRED_EFFECTS.csv',
    'METHOD_COSTS.csv', 'EXPERT_BANK16_HISTORY.csv', 'JUDGE_CLOSURE.json',
    'JUDGE_CONTEXT_VERSION.json', 'SOURCE_COVERAGE.json')

if __name__ == '__main__':
    sys.argv.extend(['--branch', 'medtrace-stage3-20260908', '--terminal-report', 'EXECUTION_STATUS.json'])
    delivery.main()
