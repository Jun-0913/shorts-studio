#!/usr/bin/env python3
"""Two independent source jobs, with existing shared heavy-media admission."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from store import Store

APP = Path(__file__).resolve().parent
REPO = APP.parents[1]


def yt_dlp():
    return os.environ.get("YT_DLP_BIN", "yt-dlp")


def supervise(root, job_id, attempt):
    spec = importlib.util.spec_from_file_location("shorts_stage_process", APP / "stage_process.py")
    stages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stages)
    folder = root / "jobs" / job_id / f"attempt-{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(APP / "producer.py"), "--root", str(root), "--job", job_id, "--attempt", str(attempt)]
    with (folder / "producer.log").open("w") as log:
        try:
            result = stages.run_stage(command, timeout=4 * 3600, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            return result.returncode
        except subprocess.TimeoutExpired:
            return 124
        except stages.StageInterrupted:
            return 130


def serve(store):
    process_lock = (store.root / ".worker.lock").open("a")
    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store.recover()
    active = {}
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            store.set_meta("worker", {"heartbeat": time.time(), "slots": store.slots, "pid": os.getpid()})
            for job_id, process in list(active.items()):
                if process.poll() is not None:
                    job = store.job(job_id)
                    error = ""
                    if process.returncode:
                        error = "제작이 중단되었습니다. 다시 시도해주세요."
                        issue = store.root / "jobs" / job_id / f"attempt-{job['attempts']}" / "error.json"
                        if issue.is_file():
                            error = json.loads(issue.read_text()).get("message", error)
                    store.finish(job_id, error)
                    del active[job_id]
            while len(active) < store.slots:
                # Deployment takes this lock exclusively before recreating services.
                lease = (store.root / ".lifecycle.lock").open("a")
                try:
                    fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    break
                job = store.claim()
                if not job:
                    lease.close()
                    break
                try:
                    process = subprocess.Popen([sys.executable, str(APP / "worker.py"), "--root", str(store.root),
                                                "--supervise", job["id"], "--attempt", str(job["attempts"])],
                                               cwd=REPO, start_new_session=True, pass_fds=(lease.fileno(),))
                    active[job["id"]] = process
                except Exception:
                    store.finish(job["id"], "제작 프로세스를 시작하지 못했습니다.")
                    raise
                finally:
                    # Do not LOCK_UN: the child inherits this open-file description.
                    lease.close()
            time.sleep(2)
    finally:
        for process in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in active.values():
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        store.set_meta("worker", {"heartbeat": 0, "slots": store.slots})
        process_lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "artifacts/shorts-lab")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--supervise")
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if args.supervise:
        raise SystemExit(supervise(args.root.resolve(), args.supervise, args.attempt))
    serve(Store(args.root, slots=args.slots))
