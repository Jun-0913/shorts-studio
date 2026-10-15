#!/usr/bin/env python3
"""Loopback-only source submission, durable progress, previews and downloads."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from urllib.parse import urlparse, unquote

from store import Store
import uploads

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
STATIC = APP / 'static'

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, store):
        super().__init__(address, Handler)
        self.store = store

class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def trusted(self, mutation=False):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self.send_error(403); return False
        if mutation:
            origin = self.headers.get('Origin')
            if (origin and origin not in {'http://' + h for h in allowed}) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send_error(403); return False
        return True

    def respond_headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key,value in (extra or {}).items(): self.send_header(key,value)
        self.end_headers()

    def json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.respond_headers(status,'application/json; charset=utf-8',len(data))
        self.wfile.write(data)

    def file(self, path, download=False):
        size = path.stat().st_size
        start, end, status = 0, size-1, 200
        extra = {'Accept-Ranges':'bytes'}
        if download: extra['Content-Disposition'] = 'attachment; filename="short.mp4"'
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                self.send_error(416);return
            first,last = match.groups()
            if first:
                start = int(first); end = min(int(last),end) if last else end
            else:
                start = max(0,size-int(last))
            if start > end or start >= size:
                self.respond_headers(416,'application/octet-stream',0,{'Content-Range':f'bytes */{size}'});return
            status = 206
            extra['Content-Range'] = f'bytes {start}-{end}/{size}'
        self.respond_headers(status,mimetypes.guess_type(str(path))[0] or 'application/octet-stream',end-start+1,extra)
        if self.command == 'HEAD': return
        with path.open('rb') as source:
            source.seek(start);remaining=end-start+1
            while remaining:
                chunk=source.read(min(1024**2,remaining))
                if not chunk:break
                self.wfile.write(chunk);remaining-=len(chunk)

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        if not self.trusted():return
        path=urlparse(self.path).path
        try:
            if path == '/api/shorts/state':return self.json(self.server.store.snapshot('local'))
            if path in {'/','/index.html'}:return self.file(STATIC/'index.html')
            if path in {'/static/app.js','/static/styles.css','/static/favicon.svg'}:return self.file(STATIC/Path(path).name)
            match = re.fullmatch(r'/api/shorts/jobs/([a-f0-9]{32})/clips/([A-Za-z0-9-]+)/(video|poster)',path)
            if match:return self.file(self.server.store.media(*match.groups()),'download=1' in urlparse(self.path).query)
            self.send_error(404)
        except (ValueError,FileNotFoundError):self.send_error(404)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        if not self.trusted(True):return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 32768:raise ValueError('잘못된 요청 크기입니다.')
            if self.headers.get_content_type() != 'application/json':raise ValueError('JSON 요청이 필요합니다.')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ValueError('잘못된 요청입니다.')
            path=urlparse(self.path).path
            if path == '/api/shorts/jobs':
                return self.json(self.server.store.enqueue(body.get('urls'),body.get('max_clips',4),'local',body.get('request_key')),202)
            match=re.fullmatch(r'/api/shorts/jobs/([a-f0-9]{32})/retry',path)
            if match:return self.json(self.server.store.retry(match[1],'local'),202)
            self.send_error(404)
        except (ValueError,TypeError,FileNotFoundError) as error:self.json({'error':str(error)},400)

    def do_PUT(self):
        if not self.trusted(True):return
        if urlparse(self.path).path != '/api/shorts/upload':self.send_error(404);return
        try:
            result=uploads.accept(self.server.store,self.rfile,int(self.headers.get('Content-Length','0')),
                unquote(self.headers.get('X-Filename','')),int(self.headers.get('X-Max-Clips','4')),self.headers.get('X-Request-Key',''))
            self.json(result,202)
        except (ValueError,TypeError,subprocess.SubprocessError) as error:
            self.close_connection=True
            self.json({'error':str(error)},400)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8431)
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/studio')
    parser.add_argument('--slots',type=int,default=1)
    parser.add_argument('--no-worker',action='store_true',help='For API checks only; jobs will stay queued.')
    args=parser.parse_args()
    server=Server(('127.0.0.1',args.port),Store(args.root,slots=args.slots))
    worker=None
    if not args.no_worker:
        worker=subprocess.Popen([sys.executable,str(APP/'worker.py'),'--root',str(args.root.resolve()),'--slots',str(args.slots)])
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    print(f'Shorts Studio: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if worker:
            worker.terminate()
            try:worker.wait(timeout=20)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()

if __name__=='__main__':main()
