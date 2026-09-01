#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import html
import json
import mimetypes
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resolve_authorized_images import append_verdict, load_jsonl, load_roots, resolve_item, sha256_file, verify_chain


BIND_HOST = "127.0.0.1"
CONFIDENCE = {"high", "medium", "low"}
RELATION = {"direct_answer", "acceptable_visual_deixis", "acceptable_synonym", "context_dependent", "ambiguous", "mismatch"}
ISSUE = {"none", "question_reference_mismatch", "wrong_source_field", "wrong_entity", "wrong_location", "wrong_count", "wrong_modality", "under_specific", "over_specific", "ambiguous_question", "ambiguous_reference", "duplicate_annotation", "other"}
ACTION = {"retain", "repair", "exclude", "manual_review"}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def validate_verdict(values, review_id):
    verdict = {
        "review_id": review_id,
        "valid": values.get("valid", [""])[0] == "true",
        "confidence": values.get("confidence", [""])[0],
        "relation": values.get("relation", [""])[0],
        "issue_type": values.get("issue_type", [""])[0],
        "recommended_action": values.get("recommended_action", [""])[0],
        "reason": values.get("reason", [""])[0].strip(),
        "created_at_utc": now_utc(),
    }
    if verdict["confidence"] not in CONFIDENCE or verdict["relation"] not in RELATION or verdict["issue_type"] not in ISSUE or verdict["recommended_action"] not in ACTION:
        raise ValueError("invalid verdict enum")
    if not verdict["reason"] or len(verdict["reason"]) > 2000:
        raise ValueError("reason must contain 1..2000 characters")
    return verdict


def option(name, values):
    return "".join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in sorted(values))


def page(item, completed, total, token):
    rid = html.escape(item["review_id"])
    question = html.escape(item["question"])
    target = html.escape(item["target_reference"])
    dataset = html.escape(item["source_dataset_identifier"])
    return f"""<!doctype html><meta charset="utf-8"><title>Local review</title>
<style>body{{max-width:1000px;margin:2rem auto;font:16px system-ui}}img{{max-width:100%;max-height:65vh}}label{{display:block;margin:.7rem 0}}textarea{{width:100%;height:7rem}}</style>
<p>Progress: {completed}/{total}</p><p>Review ID: {rid}</p><p>Dataset: {dataset}</p>
<img src="/{token}/image/{rid}" alt="authorized source image">
<h3>Question</h3><p>{question}</p><h3>Target/reference</h3><p>{target}</p>
<form method="post" action="/{token}/verdict">
<input type="hidden" name="review_id" value="{rid}">
<label>Valid <select name="valid"><option value="true">true</option><option value="false">false</option></select></label>
<label>Confidence <select name="confidence">{option('confidence', CONFIDENCE)}</select></label>
<label>Relation <select name="relation">{option('relation', RELATION)}</select></label>
<label>Issue <select name="issue_type">{option('issue', ISSUE)}</select></label>
<label>Action <select name="recommended_action">{option('action', ACTION)}</select></label>
<label>Reason <textarea name="reason" required maxlength="2000"></textarea></label><button type="submit">Freeze verdict</button></form>"""


def make_handler(items, order, roots, token, output_path, event_path):
    by_id = {item["review_id"]: item for item in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def headers(self, content_type):
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")

        def completed(self):
            if not output_path.exists():
                return set()
            if not verify_chain(output_path):
                raise ValueError("review hash chain failure")
            return {row["review_id"] for row in load_jsonl(output_path)}

        def do_GET(self):
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if not parts or parts[0] != token:
                self.send_error(404)
                return
            done = self.completed()
            if len(parts) == 1:
                remaining = [rid for rid in order if rid not in done]
                if not remaining:
                    body = f"<!doctype html><p>Review complete: {len(done)}/{len(order)}</p>".encode()
                else:
                    body = page(by_id[remaining[0]], len(done), len(order), token).encode()
                self.send_response(200); self.headers("text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            if len(parts) == 3 and parts[1] == "image" and parts[2] in by_id:
                path = resolve_item(by_id[parts[2]], roots, verify_hash=True)
                data = b""
                try:
                    data = path.read_bytes()
                    self.send_response(200); self.headers(mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                finally:
                    data = b""
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != f"/{token}/verdict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self.send_error(400)
                return
            values = urllib.parse.parse_qs(self.rfile.read(length).decode())
            rid = values.get("review_id", [""])[0]
            if rid not in by_id or rid in self.completed():
                self.send_error(409)
                return
            append_verdict(output_path, validate_verdict(values, rid))
            self.send_response(303); self.send_header("Location", f"/{token}/"); self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", default="package")
    parser.add_argument("--package-zip", default="../public_metadata_only/M3BENCH_FORMAL_TARGET_REVIEW_RECONSTRUCTION_200.zip")
    parser.add_argument("--session-manifest", default="../sessions/Reviewer_A/REVIEW_SESSION_MANIFEST.json")
    parser.add_argument("--roots", default="../sessions/Reviewer_A/DATASET_ROOTS.json")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()
    session_path = Path(args.session_manifest).resolve()
    session = json.load(open(session_path, encoding="utf-8"))
    if sha256_file(Path(args.package_zip).resolve()) != session["review_package_sha256"]:
        raise SystemExit("package hash mismatch")
    items = load_jsonl(package_dir / "FORMAL_TARGET_REVIEW_INPUT_200.jsonl")
    if hashlib.sha256((package_dir / "FORMAL_TARGET_REVIEW_INPUT_200.jsonl").read_bytes()).hexdigest() != session["review_input_sha256"]:
        raise SystemExit("review input hash mismatch")
    roots = load_roots(args.roots)
    output = session_path.parent / session["reviewer_output_filename"]
    events = session_path.parent / "SESSION_EVENTS.jsonl"
    append_verdict(events, {"event": "session_started", "created_at_utc": now_utc(), "session_id": session["session_id"]})
    server = ThreadingHTTPServer((BIND_HOST, args.port), make_handler(items, session["review_order"], roots, session["session_token"], output, events))
    print(f"Local-only console: http://{BIND_HOST}:{args.port}/{session['session_token']}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
