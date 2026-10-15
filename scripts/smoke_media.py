#!/usr/bin/env python3
"""Exercise the real file importer, renderer, verification, and result queue offline."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apps/shorts-lab'))
import media
import uploads
from store import Store

def main():
    out=ROOT/'artifacts/smoke';out.mkdir(parents=True,exist_ok=True)
    source=out/'source.mp4'
    subprocess.run([str(media.ffmpeg()),'-v','error','-y','-f','lavfi','-i','testsrc2=size=1280x720:rate=24:duration=12',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=12','-c:v','libx264','-preset','ultrafast','-c:a','aac','-shortest',str(source)],check=True)
    store=Store(out/('queue-'+uuid.uuid4().hex),minimum_free_bytes=0)
    with source.open('rb') as handle:response=uploads.accept(store,handle,source.stat().st_size,'test.mp4',1,uuid.uuid4().hex)
    job=store.claim()
    if not job:raise RuntimeError('Queue did not claim the synthetic file')
    base=store.root/'jobs'/job['id']/'attempt-1'/'clip-01';(base/'speech/qa').mkdir(parents=True)
    shutil.copyfile(source,base/'speech/edited.mp4');sha=media.timeline.sha256_file(base/'speech/edited.mp4')
    media.write(base/'speech/manifest.json',{'source':{'video_id':job['video_id'],'title':'Synthetic media check'},
        'output':{'media_path':'edited.mp4','sha256':sha,'duration_seconds':12},'verification':{'synthetic':True}})
    # Synthetic transcript and boundary receipts deliberately bypass editorial/ASR calls only in this fixture.
    media.write(base/'speech/transcript.edited.json',{'transcript':{'segments':[{'id':'seg-1','start':0,'end':12,'text':'한국어 자막과 원본 음성을 검증합니다.'}]}})
    media.write(base/'speech/qa/manifest.json',{'status':'ready-for-render','synthetic':True})
    media.write(base/'boundary-qa.json',{'status':'passed','media_sha256':sha,'synthetic':True})
    edited={'title_lines':['영상 한 편에서','짧은 이야기로'],'focus_x':.5,'segments':[{'id':'seg-1','text':'한국어 자막과 원본 음성을 검증합니다.'}],'review_notes':[]}
    result=media.render(base,edited,{'start':0,'end':12},job['video_id'])
    for suffix in ['mp4']:
        shutil.copyfile(base/f'render/reel.{suffix}',out/f'reel.{suffix}')
    shutil.copyfile(base/'render/frame-1.jpg',out/'preview.jpg')
    store.add_clip(job['id'],{'id':'clip-01','title':'Synthetic media check','duration':result['output']['duration_seconds'],'sha256':result['output']['sha256'],
        'source_start':0,'source_end':12,**{key:str((base/'render'/name).relative_to(store.root)) for key,name in [('media','reel.mp4'),('poster','frame-0.jpg'),('manifest','manifest.json')]}})
    store.finish(job['id'])
    receipt={'status':store.job(job['id'])['status'],'synthetic':True,'production_e2e':False,'checks':result['verification'],'output_sha256':result['output']['sha256']}
    media.write(out/'receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
