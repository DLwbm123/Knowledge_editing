"""Neutral process-list names; original commands remain in private run bookkeeping.

This is display hygiene, not an access-control or anonymity boundary: authorized
inspection of the environment, working directory or open files reveals real paths.
"""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys


def neutral_command(command, env, role):
    if role not in {"main", "run", "job"}:
        raise ValueError("expected neutral process role")
    entry = Path(env["JOB_ENTRYPOINT"])
    # Alias the complete environment directory so pyvenv.cfg and packages stay intact.
    runtime_root = Path(command[0]).parent.parent.resolve()
    current_root = Path(sys.executable).parent.parent.resolve()
    alias = entry.parent / ("run" if runtime_root == current_root else "job")
    if alias.is_symlink():
        if alias.resolve() != runtime_root:
            raise ValueError("neutral runtime alias points to another environment")
    else:
        alias.symlink_to(runtime_root, target_is_directory=True)
    visible = [str(alias / "bin" / Path(command[0]).name), str(entry), role]
    return visible, dict(env, JOB_ARGV=json.dumps(command[1:]))


def main():
    role = sys.argv[1]
    if role not in {"main", "run", "job"}:
        raise ValueError("expected neutral process role")
    argv = json.loads(os.environ.pop("JOB_ARGV"))
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError("invalid private Python command")
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(15, ctypes.c_char_p(role.encode()), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set process name")
    sys.argv = argv
    sys.path[0] = str(Path(argv[0]).resolve().parent)
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
