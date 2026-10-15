"""Serialize heavy media allocation across every container on the cloud host.

The root-owned cloud configuration and cgroup are installed by deployment.
Local workflows without that configuration keep their existing concurrency.
Acquire before loading an ASR model; FFmpeg inherits the lease until it exits.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


CONFIG_PATH = Path("/etc/studio/media-resources.json")


class ResourceError(RuntimeError):
    pass


def read_config(path: Path = CONFIG_PATH, *, required: bool = False) -> dict | None:
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        if not required:
            return None
        raise ResourceError("Cloud media resource configuration is missing") from None
    except (OSError, ValueError) as error:
        raise ResourceError("Cloud media resource configuration cannot be read") from error
    if not isinstance(config, dict):
        raise ResourceError("Cloud media resource configuration must be an object")
    for name in ("lock_path", "budget_path"):
        if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
            raise ResourceError(f"Invalid resource {name}")
    for name in ("working_set_bytes", "reserve_bytes", "timeout_seconds"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ResourceError(f"Invalid resource {name}")
    return config


def available_bytes(config: dict, *, meminfo_path: Path = Path("/proc/meminfo")) -> int:
    """Use both host availability and the aggregate production cgroup budget."""
    try:
        budget = Path(config["budget_path"])
        limit = int((budget / "memory.max").read_text().strip())
        current = int((budget / "memory.current").read_text().strip())
        memory_stat = dict(line.split() for line in (budget / "memory.stat").read_text().splitlines())
        fields = ("inactive_file", "active_file", "file_mapped", "file_dirty", "file_writeback")
        cache = {name: int(memory_stat.get(name, "0")) for name in fields}
        # Both file LRUs can be reclaimed; recently read media can remain on
        # active_file indefinitely when admission itself prevents pressure.
        # Exclude mapped and unflushed pages conservatively. Shmem lives on
        # the anon LRUs and is deliberately not counted as file headroom.
        reclaimable_file = max(0, cache["inactive_file"] + cache["active_file"]
                               - cache["file_mapped"] - cache["file_dirty"]
                               - cache["file_writeback"])
        meminfo = dict(line.split(":", 1) for line in meminfo_path.read_text().splitlines())
        host_available = int(meminfo["MemAvailable"].split()[0]) * 1024
        if min(limit, current, host_available, *cache.values()) < 0 or limit == 0:
            raise ValueError("Invalid memory accounting")
    except (OSError, ValueError, KeyError) as error:
        raise ResourceError("Aggregate memory budget is unavailable; refusing an unbounded media job") from error
    return min(host_available, limit - max(0, current - reclaimable_file))


@contextlib.contextmanager
def heavy_job(kind: str, *, config: dict | None = None, required: bool = False):
    """Wait without a model in memory, then hold one shared lease to completion."""
    if config is None:
        config = read_config(CONFIG_PATH, required=required)
    if config is None:
        yield None
        return
    fd = None
    started = time.monotonic()
    next_notice = started
    required_bytes = config["working_set_bytes"] + config["reserve_bytes"]
    try:
        fd = os.open(config["lock_path"], os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ResourceError("Media lease must be a single regular file")
        while True:
            acquired = False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            if acquired:
                if available_bytes(config) >= required_bytes:
                    break
                # Do not reserve capacity while waiting for other processes
                # or the kernel to release memory.
                fcntl.flock(fd, fcntl.LOCK_UN)
            now = time.monotonic()
            if now - started >= config["timeout_seconds"]:
                raise ResourceError("Timed out waiting for media capacity; no heavy job was started")
            if now >= next_notice:
                print(f"media-resources: {kind} waiting for shared capacity", file=sys.stderr, flush=True)
                next_notice = now + 30
            time.sleep(1)
        yield fd
    finally:
        if fd is not None:
            # Never unlink a lock file: another container may be waiting on
            # that inode. Closing also releases it after exceptions/signals.
            os.close(fd)


def resident_bytes(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    """Read the running encoder's resident pages, including all its threads."""
    try:
        fields = dict(line.split(":", 1) for line in
                      (proc_root / str(pid) / "status").read_text().splitlines() if ":" in line)
        value, unit = fields["VmRSS"].split()
        if unit != "kB" or int(value) < 0:
            raise ValueError("Invalid process RSS")
        return int(value) * 1024
    except (OSError, ValueError, KeyError) as error:
        raise ResourceError("Cannot verify FFmpeg working memory") from error


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def encoder_parent_guard():
    """Bind a Linux encoder to this single-threaded CLI supervisor's lifetime."""
    if sys.platform != "linux":
        return None
    parent_pid = os.getpid()
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int

    def attach():
        # PR_SET_PDEATHSIG must be installed in the child before exec. A
        # subprocess.run timeout kills only the wrapper with SIGKILL, which
        # cannot be forwarded from Python. The encoder must die with it.
        if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os.write(2, b"media-resources: cannot bind encoder to supervisor lifetime\n")
            os._exit(75)
        # Cover the parent exiting between fork and the prctl call.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)

    return attach


def run_bounded_ffmpeg(command: list[str], descriptor: int, limit: int) -> int:
    """Keep the lease and inherited pipes while enforcing the admitted budget.

    Out-of-order split/trim/concat graphs can buffer gigabytes of decoded
    frames. With swap disabled, memory.high cannot reclaim those anonymous
    pages and the single lease holder can stall the entire media queue.
    """
    child = subprocess.Popen(command, pass_fds=(descriptor,),
                             preexec_fn=encoder_parent_guard())
    handlers = {}

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, forward)
        while child.poll() is None:
            try:
                current = resident_bytes(child.pid)
            except ResourceError:
                # Linux can release mm (and VmRSS) before the child becomes
                # waitable. Give that exit transition one sampling interval;
                # a still-running, unmeasurable encoder remains fail-closed.
                try:
                    child.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    break
                raise
            if current > limit:
                raise ResourceError(
                    f"FFmpeg exceeded its admitted working-memory budget "
                    f"({current} > {limit} bytes); stopped to release shared capacity. "
                    "Render reordered spans separately and concatenate the temporary "
                    "outputs instead of buffering them in one split/trim graph. "
                    "Keep the source timeline and quality checks unchanged.")
            try:
                child.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        stop_child(child)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main() -> int:
    """Run cloud FFmpeg with shared admission and a bounded working set."""
    try:
        config = read_config(required=True)
        with heavy_job("ffmpeg", config=config, required=True) as descriptor:
            return run_bounded_ffmpeg(["/usr/lib/studio/ffmpeg.bin", *sys.argv[1:]],
                                     descriptor, config["working_set_bytes"])
    except (ResourceError, OSError) as error:
        print(f"media-resources: {error}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
