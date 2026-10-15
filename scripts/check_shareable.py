#!/usr/bin/env python3
"""Reject credentials, machine-specific paths, and runtime files."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
# Keep checks generic; never store private identifiers in a public-facing guard.
PRIVATE_PATHS=[rb'/(?:Users|home)/[A-Za-z0-9_.-]+/', rb'[A-Z]:\\Users\\[A-Za-z0-9_.-]+\\']
SECRETS=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}', rb'AKIA[0-9A-Z]{16}', rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}']

def main():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    issues=[];count=0
    for name in filter(None,paths):
        path=ROOT/name
        if not path.is_file():continue
        count+=1
        if path.is_symlink() or name.startswith(('artifacts/','.tools/','.venv/','notes/','deploy/')):
            issues.append((name,'private/runtime path'))
        data=path.read_bytes()
        if path.suffix=='.ttf':continue
        if any(re.search(pattern,data,re.IGNORECASE) for pattern in PRIVATE_PATHS):issues.append((name,'machine-specific home path'))
        if any(re.search(pattern,data) for pattern in SECRETS):issues.append((name,'credential pattern'))
    for name,reason in issues:print(f'{name}: {reason}')
    if issues:raise SystemExit(1)
    print(f'PASS: {count} files; no machine-specific home paths, runtime files, or credential patterns.')
if __name__=='__main__':main()
