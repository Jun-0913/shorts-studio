import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apps/shorts-lab'))
from store import Store
import server
import uploads
import producer

PROBE={'streams':[{'codec_type':'video'},{'codec_type':'audio'}], 'format':{'duration':'30','format_name':'mov,mp4'}}

class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name),minimum_free_bytes=0)
    def upload(self,content=b'sample',key=None,probe=PROBE,filename='sample.mp4'):
        with mock.patch.object(uploads.subprocess,'run',return_value=mock.Mock(returncode=0,stdout=json.dumps(probe))):
            return uploads.accept(self.store,io.BytesIO(content),len(content),filename,1,key or uuid.uuid4().hex)
    def test_file_is_durable_hash_verified_and_has_no_exposed_local_path(self):
        job_id=self.upload(filename='../sample.mp4')['jobs'][0]['id']
        job=self.store.claim()
        source,meta=producer.source(self.store,job,Path(self.tmp.name))
        self.assertEqual(source.media_path.read_bytes(),b'sample')
        self.assertTrue(meta['local_file'])
        snapshot=Store(self.store.root,minimum_free_bytes=0).snapshot('local')
        self.assertEqual(snapshot['jobs'][0]['source_url'],'')
        self.assertEqual(snapshot['jobs'][0]['title'],'sample')
        self.assertEqual(job_id,job['id'])
    def test_duplicate_upload_reuses_active_job_and_cleans_temporary(self):
        a=self.upload();b=self.upload()
        self.assertEqual(a['jobs'][0]['id'],b['jobs'][0]['id'])
        self.assertEqual(list((self.store.root/'sources').glob('.upload-*')),[])
    def test_idempotent_upload_after_failed_attempt(self):
        key=uuid.uuid4().hex
        a=self.upload(key=key);self.store.claim();self.store.finish(a['jobs'][0]['id'],'test')
        b=self.upload(key=key);self.assertEqual(a['jobs'][0]['id'],b['jobs'][0]['id'])
    def test_no_audio_or_bad_duration_rejected_before_enqueue(self):
        for probe in [{**PROBE,'streams':[{'codec_type':'video'}]}, {**PROBE,'format':{'duration':'nan','format_name':'mp4'}}, {**PROBE,'format':{'duration':'5','format_name':'mp4'}}, {**PROBE,'format':{'duration':'30','format_name':'hls'}}]:
            with self.subTest(probe=probe),self.assertRaises(ValueError):self.upload(probe=probe)
        self.assertEqual(self.store.snapshot('local')['jobs'],[])
    def test_invalid_extension_and_incomplete_transfer_do_not_leave_sources(self):
        with self.assertRaises(ValueError):self.upload(filename='playlist.m3u8')
        with self.assertRaises(ValueError):uploads.accept(self.store,io.BytesIO(b'x'),100,'file.mp4',1,uuid.uuid4().hex)
        self.assertEqual(list((self.store.root/'sources').iterdir()),[])
    def test_malformed_source_hash_fails_on_read(self):
        self.upload();job=self.store.claim()
        (self.store.root/'sources'/job['video_id']/'source.mp4').write_bytes(b'tamper')
        with self.assertRaises(producer.media.timeline.TimelineEditError):producer.source(self.store,job,self.store.root)


class PipelineIntegrationTests(unittest.TestCase):
    def test_already_short_sources_retain_the_subject_introduction(self):
        rows=[{"start":1,"end":4,"text":"These are the animals."},{"start":5,"end":18,"text":"Their behavior is interesting."}]
        picked={"clips":[{"start_segment":0,"end_segment":1,"title_lines":["A subject","A complete story"],"reason":"Complete context"}]}
        with mock.patch.object(producer.editor,"call",return_value=picked) as call:
            result=producer.editor.select({"segments":rows},{"duration":19},4,Path("unused"))
        schema=call.call_args.args[1]["properties"]["clips"]
        self.assertEqual(schema["maxItems"],1)
        self.assertEqual(schema["items"]["properties"]["start_segment"]["enum"],[0])
        self.assertEqual(result[0]["start_segment"],0)
        picked["clips"][0]["start_segment"]=1
        with mock.patch.object(producer.editor,"call",return_value=picked),self.assertRaises(ValueError):
            producer.editor.select({"segments":rows},{"duration":19},4,Path("unused"))

    def test_sentence_fragments_inside_one_asr_segment_can_be_trimmed_without_word_loss(self):
        words=[{"id": "w1", "text":"the", "start":0., "end":.15},
               {"id":"w2", "text":"animals.", "start":.15, "end":.6},
               {"id":"w3", "text":"Their", "start":2.2, "end":2.5},
               {"id":"w4", "text":"trunks", "start":2.5, "end":3.},
               {"id":"w5", "text":"are", "start":3., "end":3.2},
               {"id":"w6", "text":"long.", "start":3.2, "end":5.}]
        document={"transcript":{"segments":[{"id":"s1","start":0.,"end":5.,"text":"the animals. Their trunks are long.","words":words}]}}
        result=producer.boundary_transcript(document)
        rows=result["transcript"]["segments"]
        self.assertEqual([r["text"] for r in rows],["the animals.","Their trunks are long."])
        self.assertEqual([w for r in rows for w in r["words"]],words)
        self.assertEqual(len(document["transcript"]["segments"]),1)
        limits,start,end=producer.reviewed_cut_ranges(result,{"first_segment":1,"last_segment":1},10,16)
        self.assertGreater(start,10.6)
        self.assertLessEqual(start,12.2)
        self.assertGreaterEqual(end,15)

    def test_boundary_segmentation_preserves_segments_without_word_times(self):
        doc={"segments":[{"id":"s","start":1,"end":2,"text":"Complete sentence."}]}
        self.assertEqual(producer.boundary_transcript(doc)["segments"],doc["segments"])

    def test_caption_failure_falls_back_to_auto_language_asr(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'navigation'
            source=mock.Mock(video_id='no-cache-source',manifest_path=Path(temporary)/'manifest.json')
            transcript={'verification':{'valid':True},'transcript':{'language':'es','segments':[{'text':'Una historia completa.'}]}}
            def run_script(name,*args,**kwargs):
                if name.startswith('youtube-caption-ingestor'):return 1
                self.assertIn('--no-prompt',args)
                self.assertIn('auto',args)
                output=Path(args[args.index('--output-dir')+1]);output.mkdir(parents=True)
                (output/'transcript.whisper-chunked.json').write_text(json.dumps(transcript))
                return 0
            with mock.patch.object(producer,'script',side_effect=run_script):
                result,path,language=producer.navigation(source,{},folder)
            self.assertEqual(language,'es')
            self.assertEqual(result,transcript)

    def test_final_boundary_check_uses_hash_bound_prompt_free_identity_rebind(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'speech/retranscription').mkdir(parents=True)
            path=folder/'speech/retranscription/transcript.source.json';path.write_text('{}')
            speech=mock.Mock(media_sha256='verified-media')
            with mock.patch.object(producer.media.timeline,'load_transcript') as check, mock.patch.object(producer.editor,'audit_boundaries',return_value={'ending_complete':True}):
                producer.verify_boundaries(speech,folder,[])
            check.assert_called_once_with(path,speech,allow_media_bound_video_id_rebind=True)

    def test_chunked_cli_accepts_prompt_free_auto_language_arguments(self):
        import subprocess
        script=ROOT/'.agents/skills/subtitle-generator/scripts/transcribe_chunked.py'
        result=subprocess.run([sys.executable,str(script),'missing.json','--language','auto','--no-prompt'],capture_output=True,text=True)
        self.assertNotIn('unrecognized arguments',result.stderr)
        self.assertIn('Input file does not exist',result.stderr)

class Routes(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name),minimum_free_bytes=0)
        self.server=server.Server(('127.0.0.1',0),self.store)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop)
    def stop(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,method='GET',body=None,headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        c.request(method,path,body=body,headers=headers or {})
        r=c.getresponse();result=(r.status,dict(r.getheaders()),r.read());c.close();return result
    def test_home_and_state_and_absence_of_publishing_routes(self):
        self.assertEqual(self.request('/')[0],200)
        self.assertEqual(json.loads(self.request('/api/shorts/state')[2])['jobs'],[])
        for path in ['/api/upload-ready','/select','/shorts','/artifacts/studio/queue.sqlite3','/static/../server.py']:
            self.assertEqual(self.request(path)[0],404)
    def test_dns_rebinding_and_cross_origin_mutations_are_rejected(self):
        self.assertEqual(self.request('/',headers={'Host':'evil.test'})[0],403)
        status,_,_=self.request('/api/shorts/jobs','POST','{}',{'Content-Type':'application/json','Origin':'https://evil.test'})
        self.assertEqual(status,403)
        self.assertEqual(self.store.snapshot('local')['jobs'],[])
    def test_real_url_enqueue_and_invalid_url_are_atomic(self):
        value={'urls':['https://youtu.be/abcdefghijk'],'max_clips':1,'request_key':uuid.uuid4().hex}
        self.assertEqual(self.request('/api/shorts/jobs','POST',json.dumps(value),{'Content-Type':'application/json'})[0],202)
        value['urls']=['https://youtu.be/abcdefghijk','https://localhost/secret']
        self.assertEqual(self.request('/api/shorts/jobs','POST',json.dumps(value),{'Content-Type':'application/json'})[0],400)
        self.assertEqual(len(self.store.snapshot('local')['jobs']),1)
    def test_upload_api_and_result_range_and_download(self):
        with mock.patch.object(uploads.subprocess,'run',return_value=mock.Mock(returncode=0,stdout=json.dumps(PROBE))):
            response=self.request('/api/shorts/upload','PUT',b'media',{'X-Filename':'file.mp4','X-Max-Clips':'1','X-Request-Key':uuid.uuid4().hex})
        self.assertEqual(response[0],202)
        job=self.store.claim();folder=self.store.root/'jobs'/job['id'];folder.mkdir(parents=True)
        for name in ['out.mp4','poster.jpg','manifest.json']:(folder/name).write_bytes(b'0123456789')
        self.store.add_clip(job['id'],{'id':'clip-1','title':'Title','sha256':'test','duration':20,'source_start':0,'source_end':20,
             'media':f'jobs/{job["id"]}/out.mp4','poster':f'jobs/{job["id"]}/poster.jpg','manifest':f'jobs/{job["id"]}/manifest.json'})
        url=f'/api/shorts/jobs/{job["id"]}/clips/clip-1/video'
        status,headers,data=self.request(url,headers={'Range':'bytes=2-5'})
        self.assertEqual((status,data),(206,b'2345'));self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
        self.assertIn('attachment',self.request(url+'?download=1')[1]['Content-Disposition'])
        self.assertEqual(self.request(url,headers={'Range':'bytes=20-'})[0],416)
        self.assertEqual(self.request(url,headers={'Range':'bytes=-3'})[2],b'789')
    def test_cross_origin_upload_never_reaches_disk(self):
        response=self.request('/api/shorts/upload','PUT',b'file',{'Origin':'https://evil.test'})
        self.assertEqual(response[0],403);self.assertFalse((self.store.root/'sources').exists())

if __name__=='__main__':unittest.main()
