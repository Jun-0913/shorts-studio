"""Durable, bounded Shorts test queue, independent of publication workflows."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ID = re.compile(r"[A-Za-z0-9_-]{11}")
JOB_ID = re.compile(r"[a-f0-9]{32}")
STAGES = {"queued": "대기 중", "source": "원본 준비", "transcript": "내용 읽기",
          "selection": "이야기 고르기", "speech": "발화·자막 확인", "render": "숏츠 만들기",
          "verify": "결과 확인", "completed": "완료", "partial": "일부 완료", "failed": "제작 실패"}


def youtube_id(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("유튜브 영상 링크를 입력해주세요.")
    p = urlparse(value.strip())
    if p.scheme != "https" or p.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"} or p.username or p.password or p.port not in {None, 443}:
        raise ValueError("https://youtube.com 또는 https://youtu.be 영상 링크만 사용할 수 있습니다.")
    if p.hostname == "youtu.be":
        video_id = p.path.strip("/")
    elif p.path == "/watch":
        video_id = (parse_qs(p.query).get("v") or [""])[0]
    else:
        match = re.fullmatch(r"/(?:shorts|live|embed)/([A-Za-z0-9_-]{11})/?", p.path)
        video_id = match[1] if match else ""
    if not ID.fullmatch(video_id):
        raise ValueError("채널·재생목록 대신 개별 유튜브 영상 링크를 넣어주세요.")
    return video_id


class Store:
    def __init__(self, root: Path, *, slots: int = 2, minimum_free_bytes: int = 20 * 1024**3):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots = min(4, max(1, int(slots)))
        self.minimum_free_bytes = minimum_free_bytes
        self.db = self.root / "queue.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, video_id TEXT NOT NULL, source_url TEXT NOT NULL,
                    title TEXT NOT NULL, account TEXT NOT NULL, max_clips INTEGER NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                    created REAL NOT NULL, updated REAL NOT NULL, started REAL, finished REAL,
                    attempts INTEGER NOT NULL DEFAULT 0, clips TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT '', request_key TEXT NOT NULL,
                    UNIQUE(account, request_key, video_id)
                );
                CREATE INDEX IF NOT EXISTS queue_order ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    job_id TEXT NOT NULL, clip_id TEXT NOT NULL, account TEXT NOT NULL,
                    decision TEXT NOT NULL, notes TEXT NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(job_id, clip_id, account)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, stage TEXT NOT NULL,
                    detail TEXT NOT NULL, created REAL NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def meta(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def enqueue(self, urls, max_clips, account, request_key):
        if not isinstance(urls, list) or not 1 <= len(urls) <= 10:
            raise ValueError("한 번에 링크 1~10개를 넣어주세요.")
        if type(max_clips) is not int or not 1 <= max_clips <= 4:
            raise ValueError("최대 숏츠 수는 1~4개입니다.")
        if not isinstance(request_key, str) or not re.fullmatch(r"[A-Za-z0-9-]{16,80}", request_key):
            raise ValueError("요청 식별자가 없습니다. 화면을 새로고침해주세요.")
        ids = list(dict.fromkeys(youtube_id(url) for url in urls))
        now = time.time()
        result = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for video_id in ids:
                old = db.execute("SELECT * FROM jobs WHERE (account=? AND request_key=? AND video_id=?) OR (video_id=? AND status IN ('queued','running')) ORDER BY created DESC LIMIT 1", (account, request_key, video_id, video_id)).fetchone()
                if old:
                    result.append({"id": old["id"], "reused": True})
                    continue
                if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 30:
                    raise ValueError("현재 제작 대기가 많습니다. 진행 중인 영상이 끝난 뒤 다시 넣어주세요.")
                job_id = uuid.uuid4().hex
                title = "원본 정보를 확인하고 있습니다"
                db.execute("INSERT INTO jobs (id,video_id,source_url,title,account,max_clips,status,stage,created,updated,request_key) VALUES (?,?,?,?,?,?,'queued','queued',?,?,?)",
                           (job_id, video_id, f"https://www.youtube.com/watch?v={video_id}", title, account, max_clips, now, now, request_key))
                db.execute("INSERT INTO events(job_id,stage,detail,created) VALUES (?,'queued','링크 접수',?)", (job_id, now))
                result.append({"id": job_id, "reused": False})
        return {"jobs": result}

    def job(self, job_id):
        if not JOB_ID.fullmatch(str(job_id)):
            raise FileNotFoundError("작업을 찾을 수 없습니다.")
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("작업을 찾을 수 없습니다.")
        value = dict(row)
        value["clips"] = json.loads(value["clips"])
        return value

    def snapshot(self, account):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 100").fetchall()
            feedback = {(r["job_id"], r["clip_id"]): dict(r) for r in db.execute("SELECT * FROM feedback WHERE account=?", (account,))}
            counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        jobs = []
        for row in rows:
            item = {k: row[k] for k in ("id", "video_id", "source_url", "title", "max_clips", "status", "stage", "detail", "created", "updated", "started", "finished", "attempts", "error")}
            if item["source_url"].startswith("local:"):
                item["source_url"] = ""
            item["stage_label"] = STAGES.get(item["stage"], "제작 중")
            item["clips"] = []
            for clip in json.loads(row["clips"]):
                item["clips"].append({k: clip[k] for k in ("id", "title", "duration", "sha256", "source_start", "source_end")})
                exposed = item["clips"][-1]
                exposed["review_notes"] = clip.get("review_notes", [])
                exposed["revised"] = bool(clip.get("previous_revision"))
                exposed["media_url"] = f"/api/shorts/jobs/{row['id']}/clips/{clip['id']}/video?sha={clip['sha256']}"
                exposed["poster_url"] = f"/api/shorts/jobs/{row['id']}/clips/{clip['id']}/poster?sha={clip['sha256']}"
                exposed["feedback"] = feedback.get((row["id"], clip["id"]), {"decision": "", "notes": ""})
            jobs.append(item)
        worker = self.meta("worker", {})
        online = time.time() - worker.get("heartbeat", 0) < 45
        return {"jobs": jobs, "counts": counts, "worker": {"online": online, "slots": worker.get("slots", self.slots), "active": counts.get("running", 0)},
                "storage_available": shutil.disk_usage(self.root).free >= self.minimum_free_bytes}

    def claim(self):
        if shutil.disk_usage(self.root).free < self.minimum_free_bytes:
            return None
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0] >= self.slots:
                return None
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='running',stage='source',started=?,updated=?,finished=NULL,error='',attempts=attempts+1 WHERE id=?", (now, now, row[0]))
        return self.job(row[0])

    def progress(self, job_id, stage, detail="", *, title=None):
        if stage not in STAGES:
            raise ValueError("Unknown production stage")
        now = time.time()
        with self.connect() as db:
            db.execute("UPDATE jobs SET stage=?,detail=?,updated=? WHERE id=? AND status='running'", (stage, detail[:240], now, job_id))
            if title:
                db.execute("UPDATE jobs SET title=? WHERE id=?", (str(title)[:400], job_id))
            db.execute("INSERT INTO events(job_id,stage,detail,created) VALUES (?,?,?,?)", (job_id, stage, detail[:240], now))

    def add_clip(self, job_id, clip):
        for key in ("media", "poster", "manifest"):
            path = (self.root / clip[key]).resolve()
            if self.root / "jobs" / job_id not in path.parents or not path.is_file():
                raise ValueError("Result is outside this job")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT clips FROM jobs WHERE id=? AND status='running'", (job_id,)).fetchone()
            if not row:
                raise ValueError("Job is no longer running")
            clips = json.loads(row[0])
            if any(c["id"] == clip["id"] for c in clips):
                raise ValueError("Duplicate clip")
            clips.append(clip)
            db.execute("UPDATE jobs SET clips=?,updated=? WHERE id=?", (json.dumps(clips, ensure_ascii=False), time.time(), job_id))

    def finish(self, job_id, error=""):
        job = self.job(job_id)
        count = len(job["clips"])
        status = "partial" if count and error else "completed" if count else "failed"
        if not error and not count:
            error = "검증을 통과한 숏츠가 없습니다. 다른 구간으로 다시 시도해주세요."
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,stage=?,detail=?,error=?,updated=?,finished=? WHERE id=? AND status='running'", (status, status, f"{count}개 생성", error[:500], time.time(), time.time(), job_id))

    def replace_clip(self, job_id, old_id, expected_sha, clip):
        """Install a verified local revision without inheriting the old review."""
        for key in ("media", "poster", "manifest"):
            path = (self.root / clip[key]).resolve()
            if self.root / "jobs" / job_id not in path.parents or not path.is_file():
                raise ValueError("Revision is outside this job")
        manifest = json.loads((self.root / clip["manifest"]).read_text())
        digest = hashlib.sha256()
        with (self.root / clip["media"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if (manifest.get("status") != "ready-for-human-review" or manifest.get("output", {}).get("sha256") != clip["sha256"]
                or actual_sha != clip["sha256"] or not manifest.get("verification") or not all(manifest["verification"].values())):
            raise ValueError("Revision has not passed render verification")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT clips FROM jobs WHERE id=? AND status IN ('completed','partial')", (job_id,)).fetchone()
            if not row:
                raise ValueError("Only completed results can be revised")
            clips = json.loads(row[0])
            index = next((i for i, value in enumerate(clips) if value["id"] == old_id), None)
            if index is None or clips[index]["sha256"] != expected_sha or any(c["id"] == clip["id"] for c in clips):
                raise ValueError("Stale or duplicate revision")
            revised = {**clip, "previous_revision": clips[index]}
            clips[index] = revised
            db.execute("UPDATE jobs SET clips=?,updated=? WHERE id=?", (json.dumps(clips, ensure_ascii=False), time.time(), job_id))
            db.execute("INSERT INTO events(job_id,stage,detail,created) VALUES (?,?,?,?)", (job_id, "verify", "앞뒤 발화와 자막 줄바꿈 수정본", time.time()))

    def recover(self):
        # The worker's exclusive process lock is required before recovery.
        with self.connect() as db:
            rows = db.execute("SELECT id,clips,attempts FROM jobs WHERE status='running'").fetchall()
            for row in rows:
                status = "partial" if json.loads(row["clips"]) else "queued" if row["attempts"] < 3 else "failed"
                db.execute("UPDATE jobs SET status=?,stage=?,error=?,updated=? WHERE id=?", (status, status, "서버 재시작으로 중단되었습니다." if status != "queued" else "", time.time(), row["id"]))

    def retry(self, job_id, account):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "failed":
                raise ValueError("실패한 작업만 다시 시도할 수 있습니다.")
            if db.execute("SELECT 1 FROM jobs WHERE video_id=? AND status IN ('queued','running')", (row["video_id"],)).fetchone():
                raise ValueError("같은 영상이 이미 제작 중입니다.")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 30:
                raise ValueError("제작 대기가 많습니다. 잠시 뒤 다시 시도해주세요.")
            db.execute("UPDATE jobs SET status='queued',stage='queued',error='',detail='',updated=?,finished=NULL WHERE id=?", (time.time(), job_id))
        return {"id": job_id}

    def save_feedback(self, job_id, clip_id, account, decision, notes):
        job = self.job(job_id)
        if not any(c["id"] == clip_id for c in job["clips"]):
            raise FileNotFoundError("숏츠를 찾을 수 없습니다.")
        if decision not in {"", "good", "edit", "hold"} or not isinstance(notes, str) or len(notes) > 3000:
            raise ValueError("평가 값이나 메모 길이를 확인해주세요.")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?)", (job_id, clip_id, account, decision, notes, time.time()))
        return {"saved": True}

    def media(self, job_id, clip_id, kind):
        job = self.job(job_id)
        clip = next((c for c in job["clips"] if c["id"] == clip_id), None)
        field = {"video": "media", "poster": "poster"}.get(kind)
        if not clip or not field:
            raise FileNotFoundError("결과를 찾을 수 없습니다.")
        path = (self.root / clip[field]).resolve()
        if self.root / "jobs" / job_id not in path.parents or not path.is_file():
            raise FileNotFoundError("결과 파일을 찾을 수 없습니다.")
        return path
