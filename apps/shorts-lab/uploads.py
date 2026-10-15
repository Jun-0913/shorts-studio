"""Stream private source files into a verified, immutable local source cache."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

MAX_BYTES = 2 * 1024**3
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.mpeg', '.mpg', '.ts', '.mts'}

def accept(store, stream, length, filename, maximum, request_key):
    if type(length) is not int or not 0 < length <= MAX_BYTES:
        raise ValueError('파일은 2GB 이하여야 합니다.')
    if type(maximum) is not int or not 1 <= maximum <= 4:
        raise ValueError('최대 숏츠 수는 1~4개입니다.')
    if not isinstance(request_key, str) or not re.fullmatch(r'[A-Za-z0-9-]{16,80}', request_key):
        raise ValueError('요청 식별자가 올바르지 않습니다.')
    filename = Path(filename.replace('\\', '/')).name[:200]
    suffix = Path(filename).suffix.lower()
    if suffix not in EXTENSIONS:
        raise ValueError('MP4, MOV, MKV, WebM 등 일반 영상 파일을 선택해주세요.')
    if shutil.disk_usage(store.root).free < store.minimum_free_bytes + length:
        raise ValueError('영상 제작을 위한 디스크 공간이 부족합니다.')
    cache = store.root / 'sources'
    cache.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.upload-', dir=cache))
    try:
        path = temporary / ('source' + suffix)
        digest = hashlib.sha256()
        with path.open('wb') as handle:
            remaining = length
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError('파일 전송이 중단되었습니다.')
                handle.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                                 '-show_format', '-show_streams', '-of', 'json', str(path)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ValueError('읽을 수 있는 영상 파일이 아닙니다.')
        probe = json.loads(result.stdout)
        formats = set(probe.get('format', {}).get('format_name', '').split(','))
        if not formats & {'mov','mp4','matroska','webm','avi','mpeg','mpegts'}:
            raise ValueError('지원하지 않는 영상 컨테이너입니다.')
        duration = float(probe.get('format', {}).get('duration') or 0)
        kinds = {s.get('codec_type') for s in probe.get('streams', [])}
        if not math.isfinite(duration) or not 10 <= duration <= 7200 or not {'video','audio'} <= kinds:
            raise ValueError('음성이 있는 10초~2시간 영상을 넣어주세요.')
        video_id = 'file-' + digest.hexdigest()[:32]
        folder = cache / video_id
        now = time.time()
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute("SELECT id FROM jobs WHERE (request_key=? AND video_id=?) OR (video_id=? AND status IN ('queued','running'))", (request_key, video_id, video_id)).fetchone()
            if old:
                return {'jobs': [{'id': old['id'], 'reused': True}]}
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 30:
                raise ValueError('제작 대기가 많습니다. 잠시 뒤 다시 시도해주세요.')
            if not folder.exists():
                (temporary / 'source.info.json').write_text(json.dumps({'id':video_id,'title':Path(filename).stem,'duration':duration,'local_file':True}, ensure_ascii=False))
                manifest = {'schema_version':1,'agent':'file-import','stage':'source',
                    'source':{'video_id':video_id,'title':Path(filename).stem},
                    'output':{'media_path':path.name,'sha256':digest.hexdigest(),'duration_seconds':duration},
                    'verification':{'has_audio':True,'has_video':True,'duration_valid':True,'sha256_verified':True}}
                (temporary / 'manifest.json').write_text(json.dumps(manifest))
                temporary.rename(folder)
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs (id,video_id,source_url,title,account,max_clips,status,stage,created,updated,request_key) VALUES (?,?,?,?,?,?,'queued','queued',?,?,?)",
                       (job_id,video_id,'local:'+video_id,Path(filename).stem,'local',maximum,now,now,request_key))
            db.execute("INSERT INTO events(job_id,stage,detail,created) VALUES (?,'queued','영상 파일 접수',?)",(job_id,now))
        return {'jobs':[{'id':job_id,'reused':False}]}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
