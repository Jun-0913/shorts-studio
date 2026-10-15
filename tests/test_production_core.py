"""Queue races, access controls and source-bound Shorts editorial validation."""
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.util
import json
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import uuid
from pathlib import Path
from unittest import mock
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


STORE = module("shorts_queue_test", "apps/shorts-lab/store.py")
EDITOR = module("shorts_editor_test", "apps/shorts-lab/editor.py")
with mock.patch.object(sys, "path", [str(ROOT / "apps/shorts-lab"), *sys.path]):
    PRODUCER = module("shorts_producer_test", "apps/shorts-lab/producer.py")

URL = "https://www.youtube.com/watch?v=abcdefghijk"


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = STORE.Store(Path(self.temporary.name), minimum_free_bytes=0)

    def enqueue(self, urls=None, key=None):
        return self.store.enqueue(urls or [URL], 4, "shorts-lab", key or uuid.uuid4().hex)["jobs"][0]["id"]

    def test_urls_are_canonicalized_and_cannot_target_other_hosts(self):
        for url in ["https://youtu.be/abcdefghijk?t=3", "https://www.youtube.com/shorts/abcdefghijk", URL+"&list=PL123"]:
            self.assertEqual(STORE.youtube_id(url), "abcdefghijk")
        for url in ["http://youtube.com/watch?v=abcdefghijk", "https://youtube.com.evil.test/watch?v=abcdefghijk",
                    "https://youtube.com@127.0.0.1/watch?v=abcdefghijk", "https://youtube.com:8000/watch?v=abcdefghijk",
                    "https://youtu.be/abcdefghijk/../../secret", "https://youtube.com/playlist?list=123", "https://127.0.0.1/", None]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                STORE.youtube_id(url)

    def test_mixed_invalid_batch_has_no_partial_side_effect(self):
        with self.assertRaises(ValueError):
            self.enqueue([URL, "https://example.com/"])
        self.assertEqual(self.store.snapshot("shorts-lab")["jobs"], [])

    def test_parallel_duplicate_submissions_create_one_job(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(lambda _: self.enqueue(), range(20)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.store.snapshot("shorts-lab")["jobs"]), 1)

    def test_idempotency_survives_completion(self):
        key = uuid.uuid4().hex
        job_id = self.enqueue(key=key)
        self.store.claim()
        self.store.finish(job_id, "Expected test failure")
        self.assertEqual(self.enqueue(key=key), job_id)
        self.assertEqual(self.store.job(job_id)["status"], "failed")

    def test_global_two_slot_claim_is_atomic_across_connections(self):
        for i in range(8):
            self.enqueue([f"https://youtu.be/{i:011d}"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(lambda _: self.store.claim(), range(8)))
        active = [x for x in claims if x]
        self.assertEqual(len(active), 2)
        self.assertEqual(len({x["id"] for x in active}), 2)
        self.assertEqual(self.store.snapshot("shorts-lab")["counts"], {"queued": 6, "running": 2})

    def test_low_disk_waits_without_consuming_attempts(self):
        job_id = self.enqueue()
        self.store.minimum_free_bytes = 10**30
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.job(job_id)["attempts"], 0)

    def test_retry_preserves_id_increments_attempt_and_recovery_is_bounded(self):
        job_id = self.enqueue()
        self.store.claim()
        self.store.finish(job_id, "Failed")
        self.store.retry(job_id, "shorts-lab")
        self.assertEqual(self.store.claim()["attempts"], 2)
        self.store.recover()
        self.assertEqual(self.store.job(job_id)["status"], "queued")
        self.store.claim()
        self.store.recover()
        self.assertEqual(self.store.job(job_id)["status"], "failed")

    def test_unsafe_result_path_and_symlink_are_rejected(self):
        job_id = self.enqueue()
        self.store.claim()
        outside = self.store.root / "outside.mp4"
        outside.write_bytes(b"media")
        folder = self.store.root / "jobs" / job_id
        folder.mkdir(parents=True)
        (folder / "linked.mp4").symlink_to(outside)
        for relative in ["outside.mp4", f"jobs/{job_id}/linked.mp4", "../secret"]:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.store.add_clip(job_id, {"id":"clip-1", "media":relative,"poster":relative,"manifest":relative})

    def test_counts_are_strict_including_booleans(self):
        for value in [0,5,True,1.5,"4"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.enqueue([URL],value,"local",uuid.uuid4().hex)

    def test_revision_preserves_original_and_does_not_inherit_feedback(self):
        import hashlib
        job_id=self.enqueue();self.store.claim()
        folder=self.store.root/"jobs"/job_id;folder.mkdir(parents=True)
        def result(identifier):
            (folder/f"{identifier}.mp4").write_bytes(identifier.encode())
            sha=hashlib.sha256(identifier.encode()).hexdigest()
            (folder/f"{identifier}.json").write_text(json.dumps({"status":"ready-for-human-review","output":{"sha256":sha},"verification":{"valid":True}}))
            return {"id":identifier,"title":"Title","duration":20,"sha256":sha,"source_start":1,"source_end":21,
                "media":f"jobs/{job_id}/{identifier}.mp4","poster":f"jobs/{job_id}/{identifier}.mp4","manifest":f"jobs/{job_id}/{identifier}.json"}
        old,new=result("old"),result("new")
        self.store.add_clip(job_id,old);self.store.finish(job_id)
        self.store.save_feedback(job_id,"old","shorts-lab","edit","Cut the tail")
        self.store.replace_clip(job_id,"old",old["sha256"],new)
        saved=self.store.job(job_id)["clips"]
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]["previous_revision"],old)
        exposed=self.store.snapshot("shorts-lab")["jobs"][0]["clips"][0]
        self.assertEqual(exposed["feedback"]["decision"],"")
        self.assertTrue((folder/"old.mp4").is_file())
        with self.assertRaises(ValueError):
            self.store.replace_clip(job_id,"old",old["sha256"],new)


class EditorialTests(unittest.TestCase):
    def test_user_reported_opening_and_trailing_fragments_are_blocked(self):
        for texts in [["하고 얘기를 했었죠.", "완결된 문장입니다."], ["새로운 이야기입니다.", "많은..."]]:
            rows = [{"start": i*5, "end": i*5+4, "text": text} for i, text in enumerate(texts)]
            with self.subTest(texts=texts), mock.patch.object(EDITOR, "call") as call, self.assertRaises(ValueError):
                EDITOR.audit_boundaries({"transcript": {"segments": rows}}, Path("unused"))
            call.assert_not_called()

    def test_incomplete_final_predicate_fails_editorial_gate(self):
        rows = [{"start":0, "end":20, "text":"제품을 먼저 팔기 시작했는데"}]
        review = {"opening_self_contained":True, "ending_complete":False, "no_neighboring_fragment":True, "reason":"다음 술어가 필요함"}
        with mock.patch.object(EDITOR, "call", return_value=review), self.assertRaises(ValueError):
            EDITOR.audit_boundaries({"segments":rows}, Path("unused"))

    def test_cut_ranges_cannot_include_neighboring_speech(self):
        rows = [{"start":0, "end":.8}, {"start":1.2, "end":16., "words":[{"start":1.2,"end":15.93}]},
                {"start":16.05,"end":17.2}]
        limits, start, end = PRODUCER.reviewed_cut_ranges({"segments":rows}, {"first_segment":1,"last_segment":1}, 100., 120.)
        self.assertGreater(start, 100.8)
        self.assertLessEqual(limits["start"][1], 101.2)
        self.assertGreaterEqual(limits["end"][0], 115.93)
        self.assertLess(limits["end"][1], 116.05)
        self.assertLess(end, 116.05)

    def test_contiguous_word_boundaries_do_not_create_negative_padding(self):
        rows=[{"start":0,"end":1.2},{"start":1.2,"end":16.},{"start":16.,"end":20.}]
        limits,start,end=PRODUCER.reviewed_cut_ranges({"segments":rows},{"first_segment":1,"last_segment":1},0,20)
        self.assertEqual((start,end),(1.2,16.))
        self.assertEqual(limits,{"start":[1.2,1.2],"end":[16.,16.]})

    def test_acoustic_snap_respects_final_word_guard(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(PRODUCER.media, "pcm", return_value=np.zeros(20800)):
            cut, receipt = PRODUCER.media.boundary(SimpleNamespace(media_path=Path("unused"),duration_seconds=60),
                47.14, "end", Path(temporary)/"wave.png", [47.10,47.16])
        self.assertGreaterEqual(cut,47.10)
        self.assertLessEqual(cut,47.16)
        self.assertEqual(receipt["allowed_range"],[47.10,47.16])

    def test_shorts_uses_shared_layout_without_detached_ending_word(self):
        font = ImageFont.truetype(str(ROOT/"assets/fonts/noto-sans-kr/NotoSansKR-VariableFont_wght.ttf"),59)
        text = "사실 회사가 저희가 아무것도 안 하고 있었고"
        pages = PRODUCER.media.subtitle_pages(text,font)
        self.assertEqual(pages,PRODUCER.media.SHORTS.paginate_subtitle_text(text,font=font,max_width=940,max_lines=2,scale_x=100,
            protected_phrases=PRODUCER.media.SHORTS.load_subtitle_regression_corpus()["protected_phrases"]))
        self.assertFalse(PRODUCER.media.subtitle_layout.layout_has_orphans(pages))
        self.assertEqual("".join("".join(p) for p in pages).replace(" ",""),text.replace(" ",""))

    def test_editorial_phrase_hints_protect_negative_verbs_quotes_and_terms(self):
        font=ImageFont.truetype(str(ROOT/"assets/fonts/noto-sans-kr/NotoSansKR-VariableFont_wght.ttf"),59)
        cases=[("그래서 효진이한테 말을 안 하고 그림을 먼저 그렸고", "말을 안 하고"),
               ("CTO인 상호님한테 “이거 가능해요?”라고만 물어놓고", "“이거 가능해요?”라고만"),
               ("마침 상호님이 비슷한 걸 사이드 프로젝트로 했다가 버린 게 있었대요.","사이드 프로젝트로 했다가")]
        for text,phrase in cases:
            with self.subTest(phrase=phrase):
                pages=PRODUCER.media.subtitle_pages(text,font,protected_phrases=[phrase])
                self.assertTrue(any(phrase in line for page in pages for line in page))
                self.assertFalse(PRODUCER.media.subtitle_layout.layout_has_orphans(pages))

    def test_invented_phrase_hints_are_rejected(self):
        rows=[{"id":"seg-1","start":0,"end":20,"text":"완성된 문장입니다."}]
        value={"title_lines":["제목","둘째"],"focus_x":.5,"segments":[{"id":"seg-1","text":"완성된 문장입니다."}],
               "protected_phrases":["없는 문장"],"review_notes":[]}
        with mock.patch.object(EDITOR,"call",return_value=value),self.assertRaises(ValueError):
            EDITOR.captions({"segments":rows},[],{"title_lines":["제목","둘째"]},{},Path("unused"),Path("unused.jpg"))

    def test_render_requires_boundary_receipt_for_exact_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/"speech/qa").mkdir(parents=True)
            (root/"speech/qa/manifest.json").write_text('{"status":"ready-for-render"}')
            (root/"boundary-qa.json").write_text('{"status":"passed","media_sha256":"old"}')
            with mock.patch.object(PRODUCER.media.timeline,"load_media_manifest",return_value=SimpleNamespace(media_sha256="new")), self.assertRaises(ValueError):
                PRODUCER.media.render(root,{}, {}, "abcdefghijk")

    def test_selection_has_real_bounds_and_no_material_overlap(self):
        segments = [{"start":i*3., "end":(i+1)*3., "text":"문장입니다."} for i in range(40)]
        item = {"start_segment":3,"end_segment":12,"title_lines":["한 줄 제목","두 번째 줄"],"reason":"완결된 이야기"}
        output = EDITOR.validate_selection({"clips":[item]},segments,120,4)
        self.assertAlmostEqual(output[0]["start"],8.82)
        for bad in [{"clips":[]},{"clips":[item,item]},{"clips":[{**item,"end_segment":100}]},{"clips":[{**item,"start_segment":True}]},{"clips":[{**item,"title_lines":["ASS{\\pos}","둘"]}]}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EDITOR.validate_selection(bad,segments,120,4)

    def test_canonical_nested_transcript_is_understood(self):
        rows=[{"id":"seg-1","start":0,"end":2,"text":"안녕하세요"}]
        self.assertEqual(EDITOR.transcript_segments({"transcript":{"segments":rows}}),rows)

    def test_boundary_refinement_removes_edge_fragments_only(self):
        rows = [{"start":0., "end":.8, "text":"하고 말했죠."},
                {"start":1.2, "end":16., "text":"완결된 이야기입니다."},
                {"start":16.8, "end":17.2, "text":"많은…"}]
        with mock.patch.object(EDITOR, "call", return_value={"first_segment":1,"last_segment":1,"reason":"양끝 조각"}):
            result = EDITOR.boundaries({"transcript":{"segments":rows}},Path("unused"))
        self.assertEqual((result["start"],result["end"]),(1.2,16.))
        rows[1]["start"] = 9.
        with mock.patch.object(EDITOR, "call", return_value={"first_segment":1,"last_segment":1,"reason":"too much"}), self.assertRaises(ValueError):
            EDITOR.boundaries({"transcript":{"segments":rows}},Path("unused"))

    def test_caption_output_cannot_drop_or_reorder_asr(self):
        rows=[{"id":"seg-1","start":0,"end":2,"text":"첫 문장"},{"id":"seg-2","start":2,"end":5,"text":"다음 문장"}]
        value={"title_lines":["제목","두 번째"],"focus_x":.5,"segments":[{"id":"seg-2","text":"다음 문장"}],"review_notes":[]}
        with mock.patch.object(EDITOR,"call",return_value=value), self.assertRaises(ValueError):
            EDITOR.captions({"transcript":{"segments":rows}},[],{"title_lines":["첫째","둘째"]},{},Path("unused"),Path("unused.jpg"))

    def test_uncertain_silence_edit_falls_back_to_identical_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.mp4"
            original.write_bytes(b"all-original-audio-and-video")
            transcript = root / "transcript.json"
            transcript.write_text('{"transcript":{"segments":[]}}')
            rejected = root / "speech/qa"
            rejected.mkdir(parents=True)
            (rejected / "manifest.json").write_text('{"status":"review-required"}')
            source = SimpleNamespace(media_path=original, media_sha256=PRODUCER.media.timeline.sha256_file(original),
                video_id="abcdefghijk", title="original", manifest_path=root / "source.json",
                manifest_sha256="a"*64, duration_seconds=25.)
            with mock.patch.object(PRODUCER.media.timeline, "load_media_manifest", return_value=source), \
                 mock.patch.object(PRODUCER.media.timeline, "load_transcript") as verify:
                PRODUCER.preserve_unedited_speech(source.manifest_path, transcript, root)
            verify.assert_called_once_with(transcript, source)
            self.assertEqual((root / "speech/edited.mp4").read_bytes(), original.read_bytes())
            self.assertEqual((root / "speech/transcript.edited.json").read_bytes(), transcript.read_bytes())
            self.assertEqual(json.loads((root / "speech-rejected/qa/manifest.json").read_text())["status"], "review-required")
            qa = json.loads((root / "speech/qa/manifest.json").read_text())
            self.assertEqual(qa["agent"], "shorts-lab")
            self.assertFalse(qa["edited"])
