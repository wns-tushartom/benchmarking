"""Bounded Linux child execution with exclusive private log creation.

Output stays in a local diagnostic log, never in queue state or exceptions.
Logs can contain provider diagnostics; do not publish them as source artifacts.
"""
import math
import os
from pathlib import Path
import signal
import subprocess


def _kill_group(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_child(command, *, cwd, env, log_path, timeout, pass_fds=()):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite child timeout required')
    log_path = Path(log_path).absolute()
    cwd = Path(cwd).absolute()
    if log_path.resolve() != log_path or cwd.resolve() != cwd:
        raise ValueError('child paths must be canonical without symlinks')
    # Parent must already exist. No recursive creation across untrusted paths.
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    timed_out = False
    with os.fdopen(fd, 'wb') as log:
        try:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                pass_fds=pass_fds)
        except Exception:
            raise RuntimeError('child could not start; inspect private diagnostic log') from None
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # This executor owns the isolated process group. A finished leader
            # must not leave workers retaining its inherited repository lock.
            # Descendants that deliberately escape the group require stronger
            # containment; this is not a cgroup boundary.
            _kill_group(child)
            log.flush()
            os.fsync(log.fileno())
    return {'returncode': child.returncode, 'timed_out': timed_out, 'log_path': str(log_path)}
