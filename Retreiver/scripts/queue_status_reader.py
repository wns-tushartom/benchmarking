"""Bounded Linux read of a server-configured state directory, never browser input.

Freshness is file age only; neither fresh nor running means process liveness.
"""
import json
import math
import os
from pathlib import Path
import stat
import time
from queue_public_status import public_status


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate field')
        result[key] = value
    return result


def read_status(state_dir, *, now=None, max_age=60):
    unavailable = dict(public_status(None), freshness='unavailable', live_verified=False)
    descriptors = []
    try:
        clock = time.time() if now is None else now
        if not math.isfinite(clock) or not math.isfinite(max_age) or max_age <= 0:
            return unavailable
        path = Path(state_dir)
        if not path.is_absolute() or '..' in path.parts:
            return unavailable
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory = os.open('/', flags)
        descriptors.append(directory)
        for component in path.parts[1:]:
            directory = os.open(component, flags, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open('queue_status.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        descriptors.append(fd)
        before = os.fstat(fd)
        limit = 1048576
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            return unavailable
        chunks = []
        total = 0
        while total <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        if total > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return unavailable
        raw = json.loads(b''.join(chunks), object_pairs_hook=_unique_object,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        result = public_status(raw)
        if not result['available']:
            return unavailable
        age = clock - after.st_mtime
        result.update(freshness='clock_mismatch' if age < 0 else 'stale' if age > max_age else 'fresh', live_verified=False)
        return result
    except (OSError, ValueError, TypeError, RecursionError):
        return unavailable
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
