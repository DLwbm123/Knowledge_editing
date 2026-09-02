#!/usr/bin/env python3
"""Run a configured Python script without exposing private paths in argv."""

import json
import os
import runpy
import sys


config = json.loads(os.environ.pop("AUDIT_GPU_LAUNCH"))
os.environ.update(config.get("env", {}))
sys.path.insert(0, os.getcwd())
sys.argv = [config["script"], *config.get("args", [])]
runpy.run_path(config["script"], run_name="__main__")
