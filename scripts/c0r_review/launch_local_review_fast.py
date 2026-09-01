#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import html
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resolve_authorized_images import append_verdict, flag_counts, load_jsonl, load_roots, remaining_order, resolve_item, sha256_file, verify_all, verify_chain

BIND_HOST = "127.0.0.1"
CONFIDENCE = {"high", "medium", "low"}
RELATION = {"direct_answer", "acceptable_visual_deixis", "acceptable_synonym", "context_dependent", "ambiguous", "mismatch"}
ISSUE = {"none", "question_reference_mismatch", "wrong_source_field", "wrong_entity", "wrong_location", "wrong_count", "wrong_modality", "under_specific", "over_specific", "ambiguous_question", "ambiguous_reference", "duplicate_annotation", "other"}
ACTION = {"retain", "repair", "exclude", "manual_review"}
PRESETS = {
    "1": {"valid": "true", "confidence": "high", "relation": "direct_answer", "issue_type": "none", "recommended_action": "retain", "reason": "The target directly answers the question and is consistent with the displayed image."},
    "2": {"valid": "true", "confidence": "high", "relation": "acceptable_visual_deixis", "issue_type": "none", "recommended_action": "retain", "reason": "The target is a valid image-dependent spatial or deictic answer in the displayed image."},
    "3": {"valid": "true", "confidence": "medium", "relation": "context_dependent", "issue_type": "none", "recommended_action": "retain", "reason": "The target is acceptable when interpreted with the displayed image context."},
    "4": {"valid": "false", "confidence": "high", "relation": "mismatch", "issue_type": "", "recommended_action": "exclude", "reason": ""},
    "5": {"valid": "false", "confidence": "low", "relation": "ambiguous", "issue_type": "", "recommended_action": "manual_review", "reason": ""},
}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def one(values, name):
    return values.get(name, [""])[0].strip()


def validate_verdict(values, review_id):
    preset = one(values, "preset")
    raw_valid = one(values, "valid")
    if preset not in {*PRESETS, "6"} or raw_valid not in {"true", "false"}:
        raise ValueError("explicit verdict selection required")
    verdict = {
        "review_id": review_id,
        "valid": raw_valid == "true",
        "confidence": one(values, "confidence"),
        "relation": one(values, "relation"),
        "issue_type": one(values, "issue_type"),
        "recommended_action": one(values, "recommended_action"),
        "reason": one(values, "reason"),
        "created_at_utc": now_utc(),
    }
    if verdict["confidence"] not in CONFIDENCE or verdict["relation"] not in RELATION or verdict["issue_type"] not in ISSUE or verdict["recommended_action"] not in ACTION:
        raise ValueError("invalid verdict enum")
    minimum = 12 if preset in {"4", "5"} or not verdict["valid"] or verdict["recommended_action"] == "manual_review" else 1
    if len(verdict["reason"]) < minimum or len(verdict["reason"]) > 2000:
        raise ValueError(f"reason must contain {minimum}..2000 characters")
    if preset == "4" and (verdict["issue_type"] == "none" or one(values, "confirmed") != "true"):
        raise ValueError("invalid verdict requires issue and confirmation")
    if preset == "5" and (verdict["issue_type"] not in {"ambiguous_question", "ambiguous_reference", "other"} or one(values, "confirmed") != "true"):
        raise ValueError("uncertain verdict requires ambiguity issue and confirmation")
    return verdict


def accept_review_id(review_id, known_ids, completed):
    if review_id not in known_ids or review_id in completed:
        raise FileExistsError(review_id)


def flag_allowed(count):
    return count < 2


def options(values):
    return '<option value="" selected>Choose…</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in sorted(values)
    )


def page(item, completed, total, flagged, token):
    rid = html.escape(item["review_id"])
    question = html.escape(item["question"])
    target = html.escape(item["target_reference"])
    presets = html.escape(json.dumps(PRESETS, separators=(",", ":")))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Fast local review</title>
<style>
body{{max-width:1200px;margin:1rem auto;padding:0 1rem;font:18px system-ui;background:#111;color:#eee}}.health,.progress,.keys{{position:sticky;top:0;background:#1b1b1b;padding:.65rem;z-index:2}}.health{{color:#7ee787}}.image{{height:58vh;display:flex;align-items:center;justify-content:center;overflow:auto;background:#000}}img{{max-width:100%;max-height:100%;object-fit:contain;transform-origin:center}}h2{{font-size:1.55rem}}.target{{font-size:1.4rem;color:#ffd580}}button{{font-size:1rem;padding:.65rem;margin:.25rem}}button.selected{{outline:4px solid #58a6ff}}label{{display:block;margin:.5rem 0}}select,textarea{{width:100%;font-size:1rem;background:#222;color:#eee}}textarea{{height:5rem}}#submit:disabled{{opacity:.35}}#confirm{{display:none;background:#7c2d12;padding:.7rem}}.keys{{bottom:0;top:auto;font-size:.9rem}}[hidden]{{display:none!important}}
</style></head><body>
<div class="health">package hash PASS · input hash PASS · image hash PASS</div>
<div class="progress">Random review progress only — not formal position: {completed}/{total} completed · {total-completed} remaining · {flagged} flagged-for-later</div>
<div class="image"><img id="image" src="/{token}/image/{rid}" alt="authorized source image" draggable="false"></div>
<h2>Question</h2><p>{question}</p><h2>Target/reference</h2><p class="target">{target}</p>
<form id="form" method="post" action="/{token}/verdict"><input type="hidden" name="review_id" value="{rid}"><input id="preset" type="hidden" name="preset"><input id="confirmed" type="hidden" name="confirmed" value="false">
<div><button type="button" data-preset="1">1 Direct valid</button><button type="button" data-preset="2">2 Visual deixis</button><button type="button" data-preset="3">3 Context valid</button><button type="button" data-preset="4">4 Invalid</button><button type="button" data-preset="5">5 Uncertain</button><button type="button" data-preset="6">6 Custom</button></div>
<div id="fields" hidden><label>Valid<select name="valid">{options({'true','false'})}</select></label><label>Confidence<select name="confidence">{options(CONFIDENCE)}</select></label><label>Relation<select name="relation">{options(RELATION)}</select></label><label>Issue<select name="issue_type">{options(ISSUE)}</select></label><label>Action<select name="recommended_action">{options(ACTION)}</select></label><label>Reason<textarea name="reason" maxlength="2000"></textarea></label></div>
<div id="confirm">Press Enter again or click Confirm to freeze this 4/5 verdict. <button id="confirmButton" type="button">Confirm</button></div><button id="submit" type="submit" disabled>Freeze verdict</button></form>
<form id="flagForm" method="post" action="/{token}/flag"><input type="hidden" name="review_id" value="{rid}"><button type="submit">F Flag for later</button></form>
<div id="help" class="keys">1/2/3 select valid preset · 4 invalid · 5 uncertain · 6 custom · Enter submit · Esc clear · F later · +/- zoom · 0 fit · ? help</div>
<script>
const PRESETS=JSON.parse('{presets}'),form=document.querySelector('#form'),fields=document.querySelector('#fields'),submit=document.querySelector('#submit'),confirmBar=document.querySelector('#confirm'),preset=document.querySelector('#preset'),confirmed=document.querySelector('#confirmed'),image=document.querySelector('#image');let confirmStage=0,zoom=1;
function setField(name,value){{form.elements[name].value=value}}
function valid(){{if(!preset.value)return false;for(const n of ['valid','confidence','relation','issue_type','recommended_action'])if(!form.elements[n].value)return false;const reason=form.elements.reason.value.trim(),need=['4','5'].includes(preset.value)||form.elements.valid.value==='false'||form.elements.recommended_action.value==='manual_review'?12:1;return reason.length>=need}}
function refresh(){{submit.disabled=!valid();if(!valid()){{confirmStage=0;confirmed.value='false';confirmBar.style.display='none'}}}}
function choose(key){{clear(false);preset.value=key;fields.hidden=false;if(key!=='6')for(const [n,v] of Object.entries(PRESETS[key]))setField(n,v);document.querySelector(`[data-preset="${{key}}"]`).classList.add('selected');refresh();if(['4','5','6'].includes(key))form.elements.reason.focus()}}
function clear(hide=true){{form.reset();preset.value='';confirmed.value='false';confirmStage=0;confirmBar.style.display='none';document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('selected'));fields.hidden=hide;refresh()}}
function freeze(){{if(!valid())return;if(['4','5'].includes(preset.value)&&confirmStage===0){{confirmStage=1;confirmBar.style.display='block';return}}confirmed.value='true';form.querySelectorAll('button,select,textarea').forEach(x=>x.disabled=true);form.submit()}}
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>choose(b.dataset.preset));document.querySelector('#confirmButton').onclick=()=>{{confirmStage=1;freeze()}};form.oninput=refresh;form.onsubmit=e=>{{e.preventDefault();freeze()}};
document.addEventListener('keydown',e=>{{const editing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName);if(e.key==='Escape'){{e.preventDefault();clear()}}else if(e.key==='Enter'){{e.preventDefault();freeze()}}else if(!editing&&/^[1-6]$/.test(e.key)){{e.preventDefault();choose(e.key)}}else if(!editing&&e.key.toLowerCase()==='f'){{e.preventDefault();document.querySelector('#flagForm').requestSubmit()}}else if(!editing&&e.key==='?'){{document.querySelector('#help').hidden=!document.querySelector('#help').hidden}}else if(!editing&&['+','-','0'].includes(e.key)){{zoom=e.key==='0'?1:Math.min(3,Math.max(.5,zoom+(e.key==='+'?.25:-.25)));image.style.transform=`scale(${{zoom}})`}}}});clear();
</script></body></html>"""


def make_handler(items, order, roots, token, output_path, event_path):
    by_id = {item["review_id"]: item for item in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def headers(self, content_type):
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=()")
            self.send_header("Content-Disposition", "inline")

        def completed_rows(self):
            if not output_path.exists():
                return []
            if not verify_chain(output_path):
                raise ValueError("review hash chain failure")
            rows = load_jsonl(output_path)
            if len({row["review_id"] for row in rows}) != len(rows):
                raise ValueError("duplicate review ID")
            return rows

        def do_GET(self):
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if not parts or parts[0] != token:
                self.send_error(404); return
            rows = self.completed_rows(); done = {row["review_id"] for row in rows}
            if len(parts) == 2 and parts[1] == "health":
                body = json.dumps({"status": "PASS", "completed": len(done)}).encode()
                self.send_response(200); self.headers("application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if len(parts) == 1:
                queue = remaining_order(order, done, event_path)
                active_flags = sum(rid not in done for rid in flag_counts(event_path))
                body = (f"<!doctype html><p>Review complete: {len(done)}/{len(order)}</p>" if not queue else page(by_id[queue[0]], len(done), len(order), active_flags, token)).encode()
                self.send_response(200); self.headers("text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if len(parts) == 3 and parts[1] == "image" and parts[2] in by_id:
                path = resolve_item(by_id[parts[2]], roots, verify_hash=True); data = b""
                try:
                    data = path.read_bytes(); self.send_response(200); self.headers(mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                finally:
                    data = b""
                return
            self.send_error(404)

        def do_POST(self):
            if self.path not in {f"/{token}/verdict", f"/{token}/flag"}:
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self.send_error(400); return
            values = urllib.parse.parse_qs(self.rfile.read(length).decode()); rid = one(values, "review_id")
            done = {row["review_id"] for row in self.completed_rows()}
            try:
                accept_review_id(rid, by_id, done)
            except FileExistsError:
                self.send_error(409); return
            if self.path.endswith("/flag"):
                if not flag_allowed(flag_counts(event_path).get(rid, 0)):
                    self.send_error(409, "flag limit reached; verdict required"); return
                append_verdict(event_path, {"event": "flagged_for_later", "review_id": rid, "created_at_utc": now_utc()})
            else:
                try:
                    append_verdict(output_path, validate_verdict(values, rid))
                except ValueError as exc:
                    self.send_error(400, str(exc)); return
            self.send_response(303); self.send_header("Location", f"/{token}/"); self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", default="../package")
    parser.add_argument("--package-zip", default="../../../public_metadata_only/M3BENCH_FORMAL_TARGET_REVIEW_RECONSTRUCTION_200.zip")
    parser.add_argument("--session-manifest", default="../../../sessions/Reviewer_A/REVIEW_SESSION_MANIFEST.json")
    parser.add_argument("--roots", default="../../../sessions/Reviewer_A/DATASET_ROOTS.json")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--preverified-images", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); package_dir = Path(args.package_dir).resolve(); session_path = Path(args.session_manifest).resolve(); session = json.load(open(session_path, encoding="utf-8"))
    if session["session_id"] != args.session_id or sha256_file(Path(args.package_zip).resolve()) != session["review_package_sha256"]:
        raise SystemExit("session or package hash mismatch")
    input_path = package_dir / "FORMAL_TARGET_REVIEW_INPUT_200.jsonl"; items = load_jsonl(input_path)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != session["review_input_sha256"]:
        raise SystemExit("review input hash mismatch")
    roots = load_roots(args.roots); report = {"status": "PASS"} if args.preverified_images else verify_all(items, roots)
    if report["status"] != "PASS" or not (len(items) == len(session["review_order"]) == 200):
        raise SystemExit("image or queue verification failed")
    output = session_path.parent / session["reviewer_output_filename"]; events = session_path.parent / "SESSION_EVENTS.jsonl"
    append_verdict(events, {"event": "session_started", "created_at_utc": now_utc(), "session_id": session["session_id"]})
    server = ThreadingHTTPServer((BIND_HOST, args.port), make_handler(items, session["review_order"], roots, session["session_token"], output, events))
    print(f"Local-only console: http://{BIND_HOST}:{args.port}/{session['session_token']}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
