"""Run a production stage and reclaim all of its subprocesses."""

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class StageInterrupted(InterruptedError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Production stage interrupted by signal {signum}")


def process_table():
    """Return PID -> (parent PID, start time); start time protects against reuse."""
    result = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = entry.read_text().rsplit(")", 1)[1].split()
            result[int(entry.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(table, parent):
    found = {parent}
    while True:
        added = {pid for pid, (ppid, _) in table.items() if ppid in found} - found
        if not added:
            return found - {parent}
        found.update(added)


def subreaper(enabled):
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot read stage subreaper setting")
    if libc.prctl(36, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set stage subreaper")
    return bool(previous.value)


def cleanup(process, baseline):
    # A process group alone misses Codex tool sessions that called setsid().
    # Linux adoption keeps those children discoverable after their parent exits.
    tracked = {}
    deadline = time.monotonic() + 2
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        until = deadline if signum == signal.SIGTERM else time.monotonic() + 3
        while True:
            table = process_table()
            for pid in descendants(table, os.getpid()):
                if baseline.get(pid) != table[pid][1]:
                    tracked[pid] = table[pid][1]
            live = {pid for pid, start in tracked.items()
                    if pid in table and table[pid][1] == start}
            for pid in live:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
            # Let Popen reap its own child before collecting adopted descendants.
            process.poll()
            for pid in live - {process.pid}:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if not live and process.poll() is not None:
                return
            if time.monotonic() >= until:
                break
            time.sleep(.05)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=3)
    remaining = process_table()
    if any(pid in remaining and remaining[pid][1] == start for pid, start in tracked.items()):
        raise RuntimeError("Stage descendants did not exit after SIGKILL; refusing to continue")


def run_stage(command, *, timeout, check=False, **kwargs):
    previous = subreaper(True)
    table = process_table()
    baseline = {pid: table[pid][1] for pid in descendants(table, os.getpid())}
    handlers = {}
    process = None

    def interrupted(signum, _frame):
        raise StageInterrupted(signum)

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        returncode = process.wait(timeout=timeout)
        result = subprocess.CompletedProcess(command, returncode)
        if check:
            result.check_returncode()
        return result
    finally:
        # A second shutdown signal must not interrupt descendant cleanup.
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                cleanup(process, baseline)
        finally:
            subreaper(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
