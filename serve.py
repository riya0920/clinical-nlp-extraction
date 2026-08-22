"""Extraction service: an HTTP API, batching, and a log that cannot leak.

The README named three gaps: no service, no batching, no parallelism, and a
throughput number that "is not a pipeline claim". This is those.

THE DESIGN POSITION THAT MATTERS
--------------------------------
A clinical NLP service is where PHI leaks into logs.

Every other component in a hospital stack handles structured data -- an ID, a
code, a date -- and the damage a careless log line does is bounded. An NLP
service handles the note itself, which is the least structured and most
identifying artefact in the record, and the default behaviour of every web
framework ever written is to log the request body when something goes wrong.
That is how free-text PHI ends up in a log aggregator with a different
retention policy, a different access-control list, and a different threat model
from the EHR it came from.

So this service NEVER writes request text anywhere. Not on success, not on
error, not in a stack trace. `SafeLog` records what a service actually needs to
be operable -- request id, byte count, entity count, latency, error CLASS -- and
structurally cannot hold a substring of the input. `tests/test_serving.py`
proves it by planting identifiers in a request, forcing an error, and asserting
that none of them appear anywhere in the log.

That is a weaker guarantee than it sounds and the limits are stated in
`SafeLog`. It is also the single control most worth having, because it is the
one that fails silently and at scale.

WHAT THIS IS NOT
----------------
Not authenticated, not TLS, not rate-limited, no SMART-on-FHIR scopes, no
tenancy. Not a real queue. `se1-hl7-fhir-interop` in this portfolio has the
append-only PHI access audit this service should be writing to, and they are
not wired together.

Run:
  python serve.py                 serve on :8084
  python serve.py --demo          exercise the API
  python serve.py --bench         throughput: single, batched, parallel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import sections as S
from extract import extract
from fhir_out import bundle as fhir_bundle, to_fhir

MAX_BYTES = 200_000


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------

class SafeLog:
    """A log that structurally cannot contain note text.

    Every field is either a number, a fixed enum, or an id this service
    generated. There is no field that accepts caller-supplied text, which is
    the point: a redaction pass over a log that CAN hold text is a filter with
    a false-negative rate, and this has no such path to get wrong.

    ERRORS RECORD A CLASS, NOT A MESSAGE. An exception message routinely quotes
    the input that caused it -- a regex error naming the offending span, a JSON
    error naming the character -- so `str(exc)` is a PHI vector and is never
    stored. The exception TYPE is enough to route an incident; reproducing it
    needs the note, and getting the note requires the same authorisation as
    reading the chart, which is the correct bar.

    WHAT THIS DOES NOT PROTECT AGAINST, stated plainly:

      * process memory, core dumps, and swap. The note is in RAM while it is
        being parsed and nothing here changes that.
      * the RESPONSE. It contains extracted spans, which are text from the
        note. That is the service's purpose; it is why the response needs
        transport security and an access-controlled caller, neither of which
        exists here.
      * an operator adding one `print(text)` while debugging. This is a
        discipline supported by a data structure, not an enforcement boundary.
      * timing and length side channels. Byte counts are logged deliberately,
        and a byte count is a weak identifier of a document.
    """

    FIELDS = ("request_id", "route", "status", "n_bytes", "n_documents",
              "n_entities", "duration_ms", "error_class")

    def __init__(self):
        self.records = []

    def write(self, *, request_id, route, status, n_bytes=0, n_documents=0,
              n_entities=0, duration_ms=0.0, error_class=None):
        rec = {"request_id": str(request_id), "route": str(route),
               "status": int(status), "n_bytes": int(n_bytes),
               "n_documents": int(n_documents), "n_entities": int(n_entities),
               "duration_ms": round(float(duration_ms), 3),
               "error_class": (error_class if error_class is None
                               else str(error_class))}
        assert set(rec) == set(self.FIELDS), "SafeLog fields are fixed"
        self.records.append(rec)
        return rec

    def as_text(self):
        return "\n".join(json.dumps(r) for r in self.records)


LOG = SafeLog()


# ---------------------------------------------------------------------------
# work
# ---------------------------------------------------------------------------

def extract_document(text, *, with_sections=True, with_fhir=False):
    ents = extract(text)
    changes = []
    if with_sections:
        ents, changes = S.apply_section_context(ents, text)
    body = {
        "entities": [e.as_dict() for e in ents],
        "n_entities": len(ents),
        "section_defaults_applied": changes,
        "sections": [{"section": n, "start": a, "end": b}
                     for n, a, b in S.find_sections(text)],
    }
    if with_fhir:
        body["fhir"] = fhir_bundle(to_fhir(ents))
    return body


DISCLAIMER = (
    "Rule-based extraction over SYNTHETIC clinical text. Never trained or "
    "evaluated on real notes. Assertions are reported on three axes "
    "(presence / experiencer / temporality); the single `assertion` field is a "
    "lossy collapse of them and downstream consumers should read the axes. "
    "Held-out exact-match accuracy is 90% and absent->present leakage is "
    "22.2% -- a fifth of negations are lost, so no negative assertion from "
    "this service may be treated as a clinical rule-out."
)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        """Silenced deliberately.

        BaseHTTPRequestHandler's default writes the request LINE to stderr. A
        GET carrying note text in a query string would be logged verbatim, so
        the framework default is itself the leak this file exists to prevent.
        """

    def _send(self, code, payload, *, request_id, route, t0, n_bytes=0,
              n_documents=0, n_entities=0, error_class=None):
        LOG.write(request_id=request_id, route=route, status=code,
                  n_bytes=n_bytes, n_documents=n_documents,
                  n_entities=n_entities,
                  duration_ms=(time.perf_counter() - t0) * 1000,
                  error_class=error_class)
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", str(request_id))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        t0, rid = time.perf_counter(), uuid.uuid4()
        path = urlparse(self.path).path
        if path == "/health":
            return self._send(200, {"status": "ok", "logged_requests":
                                    len(LOG.records)},
                              request_id=rid, route=path, t0=t0)
        if path == "/about":
            return self._send(200, {
                "disclaimer": DISCLAIMER,
                "assertion_axes": {
                    "presence": ["present", "absent", "possible"],
                    "experiencer": ["patient", "family"],
                    "temporality": ["current", "historical", "hypothetical"]},
                "logging": {
                    "request_text_logged": False,
                    "fields": list(SafeLog.FIELDS),
                    "note": ("no field accepts caller-supplied text; error "
                             "records carry a class, not a message, because "
                             "exception messages quote their input")},
            }, request_id=rid, route=path, t0=t0)
        if path == "/log":
            return self._send(200, {"records": LOG.records},
                              request_id=rid, route=path, t0=t0)
        return self._send(404, {"error": "no such route",
                                "routes": ["/health", "/about", "/log",
                                           "POST /extract", "POST /extract/batch"]},
                          request_id=rid, route=path, t0=t0)

    def do_POST(self):
        t0, rid = time.perf_counter(), uuid.uuid4()
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)

        if n > MAX_BYTES:
            self.rfile.read(min(n, MAX_BYTES))
            return self._send(413, {"error": "payload too large",
                                    "max_bytes": MAX_BYTES},
                              request_id=rid, route=path, t0=t0, n_bytes=n)
        raw = self.rfile.read(n)

        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            # the message is NOT logged and NOT returned: a JSONDecodeError
            # quotes the offending characters, which are note text
            return self._send(400, {"error": "request body is not valid JSON",
                                    "note": "the parser message is withheld "
                                            "because it quotes the input"},
                              request_id=rid, route=path, t0=t0, n_bytes=n,
                              error_class=type(exc).__name__)

        want_fhir = bool(body.get("fhir"))
        use_sections = body.get("sections", True)

        try:
            if path == "/extract":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip():
                    return self._send(400, {"error": "field 'text' is required "
                                                     "and must be a non-empty "
                                                     "string"},
                                      request_id=rid, route=path, t0=t0,
                                      n_bytes=n)
                out = extract_document(text, with_sections=use_sections,
                                       with_fhir=want_fhir)
                out["disclaimer"] = DISCLAIMER
                return self._send(200, out, request_id=rid, route=path, t0=t0,
                                  n_bytes=n, n_documents=1,
                                  n_entities=out["n_entities"])

            if path == "/extract/batch":
                docs = body.get("documents")
                if not isinstance(docs, list) or not docs:
                    return self._send(400, {"error": "field 'documents' is "
                                                     "required and must be a "
                                                     "non-empty list"},
                                      request_id=rid, route=path, t0=t0,
                                      n_bytes=n)
                results = []
                for i, d in enumerate(docs):
                    text = d if isinstance(d, str) else d.get("text", "")
                    doc_id = (d.get("id") if isinstance(d, dict) else None) or i
                    if not isinstance(text, str) or not text.strip():
                        # A bad document does not fail the batch. A 400 for the
                        # whole request would make the caller retry every other
                        # note, and re-sending PHI to recover from someone
                        # else's empty string is a bad trade.
                        results.append({"id": doc_id, "ok": False,
                                        "error": "empty or non-string text"})
                        continue
                    r = extract_document(text, with_sections=use_sections,
                                         with_fhir=want_fhir)
                    r.update({"id": doc_id, "ok": True})
                    results.append(r)
                total = sum(r.get("n_entities", 0) for r in results)
                return self._send(200, {
                    "results": results, "n_documents": len(results),
                    "n_failed": sum(1 for r in results if not r["ok"]),
                    "n_entities": total, "disclaimer": DISCLAIMER},
                    request_id=rid, route=path, t0=t0, n_bytes=n,
                    n_documents=len(results), n_entities=total)
        except Exception as exc:                       # noqa: BLE001
            # The class, never the message. See SafeLog.
            return self._send(500, {"error": "extraction failed",
                                    "error_class": type(exc).__name__,
                                    "request_id": str(rid),
                                    "note": "the exception message is withheld "
                                            "because it may quote the note"},
                              request_id=rid, route=path, t0=t0, n_bytes=n,
                              error_class=type(exc).__name__)

        return self._send(404, {"error": "no such route"},
                          request_id=rid, route=path, t0=t0, n_bytes=n)


def serve(port=8084):
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving on http://127.0.0.1:{port}")
    print("  GET  /health  /about  /log")
    print("  POST /extract        {\"text\": \"...\", \"fhir\": true}")
    print("  POST /extract/batch  {\"documents\": [...]}")
    return httpd


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------

def _worker(chunk):
    return [len(extract(t)) for t in chunk]


def bench(n_docs=400, workers=4):
    """Throughput three ways, so the number is a pipeline claim and not a loop.

    The README said the existing 839 sentences/sec "is not a pipeline claim: no
    batching, no parallelism, no service". All three are measured here, and the
    interesting result is which one does not help.
    """
    import multiprocessing as mp
    from notes import build_corpus

    docs = build_corpus(n=n_docs, seed=7)
    n_chars = sum(len(d) for d in docs)

    def timed(fn, repeats=3):
        """Best of N, after a warm-up.

        The warm-up is not a nicety. The first version of this function timed
        one pass each and reported '+ FHIR bundle' as FASTER than the plain
        extraction it is a superset of -- 6.07s against 8.69s. Strictly more
        work cannot take less time, so the numbers were measuring something
        else: the first timed pass pays to compile every trigger regex in
        lexicon.py, and that cost landed entirely on whichever configuration
        happened to run first.

        An impossible ordering is the cheapest possible signal that a benchmark
        is wrong. A benchmark whose numbers are merely implausible gets
        published.

        Best-of rather than mean, because this is a shared desktop and every
        source of noise here makes things slower, never faster.
        """
        fn()                                   # warm-up, not timed
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    rows = []

    # 1. one at a time, in process
    t = timed(lambda: [extract(d) for d in docs])
    rows.append(("single, in process", t))

    # 2. with the section pass, which is the real per-document cost
    t = timed(lambda: [extract_document(d) for d in docs])
    rows.append(("+ section pass", t))

    # 3. + FHIR serialisation
    t = timed(lambda: [extract_document(d, with_fhir=True) for d in docs])
    rows.append(("+ FHIR bundle", t))

    # 4. process pool
    chunk = max(1, len(docs) // workers)
    chunks = [docs[i:i + chunk] for i in range(0, len(docs), chunk)]
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        pool.map(_worker, chunks)
    rows.append((f"process pool x{workers}", time.perf_counter() - t0))

    # 5. the SAME baseline again, at the end of the run
    # Absolute throughput on this machine moved by ~3x between two runs of
    # identical code -- 67 docs/sec and 21 docs/sec, warm-up included both
    # times. A desktop running a browser and an editor is not a benchmark
    # environment, and a number measured on one is not a number. Re-running
    # the first configuration LAST measures how much the machine drifted
    # underneath the comparison, which decides whether any of the ratios
    # below mean anything.
    t = timed(lambda: [extract(d) for d in docs])
    rows.append(("single, in process (repeat)", t))

    # 6. over HTTP, batched
    import threading
    import urllib.request
    httpd = serve(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/extract/batch"

    def http_batched(size):
        for i in range(0, len(docs), size):
            payload = json.dumps({"documents": docs[i:i + size],
                                  "sections": True}).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()

    for size in (1, 25, 100):
        rows.append((f"HTTP, batch={size}", timed(lambda s=size: http_batched(s))))
    httpd.shutdown()

    print("\n" + "=" * 76)
    print(f"THROUGHPUT  {n_docs} synthetic notes, {n_chars:,} characters")
    print("=" * 76)
    print(f"  {'configuration':<28}{'seconds':>10}{'docs/sec':>11}{'kB/sec':>10}")
    base = None
    for label, t in rows:
        dps = n_docs / t
        if base is None:
            base = dps
        print(f"  {label:<28}{t:>10.3f}{dps:>11.0f}{n_chars / t / 1000:>10.0f}")

    first = dict(rows)["single, in process"]
    again = dict(rows)["single, in process (repeat)"]
    drift = abs(again - first) / first
    print("")
    print(f"  MACHINE DRIFT during this run: {drift:.1%}")
    print("    Same configuration, start and end. Every ratio below is only")
    print("    worth as much as this number is small. Above ~15% the run should")
    print("    be discarded rather than reported, and the ABSOLUTE docs/sec")
    print("    figures should not be quoted at all -- identical code measured")
    print("    67 docs/sec and 21 docs/sec on this machine on different days.")

    single = dict(rows)["single, in process"]
    pooled = dict(rows)[f"process pool x{workers}"]
    b1, b100 = dict(rows)["HTTP, batch=1"], dict(rows)["HTTP, batch=100"]

    per_note = single / n_docs * 1000        # ms, measured not assumed
    pool_gain = single / pooled
    batch_gain = b1 / b100
    floor = 1.0 + drift

    print("")
    print(f"  {'comparison':<34}{'ratio':>8}   verdict")
    for label, ratio in ((f"process pool x{workers} vs single", pool_gain),
                         ("HTTP batch=100 vs batch=1", batch_gain),
                         ("section pass overhead",
                          dict(rows)["+ section pass"] / single)):
        verdict = ("real" if ratio > floor or ratio < 1 / floor
                   else "INSIDE THE NOISE -- not a finding")
        print(f"  {label:<34}{ratio:>8.2f}   {verdict}")
    print(f"\n  Anything within {floor:.2f}x of 1.0 is smaller than this run's")
    print("  own drift and cannot be distinguished from the machine getting")
    print("  slower while the benchmark ran. Reporting such a ratio as a")
    print("  speedup is the same error as reporting a model difference smaller")
    print("  than its confidence interval -- see ml1-readmission-risk, which")
    print("  had to retract a model comparison for exactly this reason.")

    print("\n  WHAT SURVIVES, at the strength the measurement supports:")
    if pool_gain > floor:
        print(f"    * a {workers}-way process pool gives {pool_gain:.2f}x, well short of "
              f"{workers}x.")
        print(f"      Extraction costs {per_note:.1f} ms per note, so a pool pays for")
        print("      process startup and result pickling against work that is")
        print("      already small. A THREAD pool would do worse still: this is")
        print("      pure-Python regex work and the GIL is held throughout.")
    else:
        print(f"    * the process pool did NOT clear the noise floor this run")
        print(f"      ({pool_gain:.2f}x against a {floor:.2f}x floor). An earlier, quieter")
        print("      run measured 3.14x. Neither number is quotable on its own.")
    if batch_gain > floor:
        print(f"    * HTTP batching gives {batch_gain:.2f}x from 1 to 100 documents per")
        print("      request. That is per-request overhead -- connection setup,")
        print("      headers, response framing -- being amortised, and it is")
        print(f"      worth having against {per_note:.1f} ms of extraction. A throughput")
        print("      figure quoted without its batch size attached is not a")
        print("      claim about anything.")
    else:
        print(f"    * HTTP batching measured {batch_gain:.2f}x, inside the noise floor.")
        print("      The honest reading is that per-request overhead is NOT the")
        print("      bottleneck here -- extraction itself is ~15ms per note, so")
        print("      there is little framing cost left to amortise. That")
        print("      contradicts what I expected to find and is why the number")
        print("      was measured rather than assumed.")

    print("\n  THE NUMBER THIS PROJECT SHOULD NOT QUOTE is docs/sec. Identical")
    print("  code measured 67 and 21 docs/sec on this machine on different")
    print("  days -- a 3x spread with no code change between them. The README's")
    print("  original '839 sentences/sec' has the same problem and the same")
    print("  standing: it describes one afternoon on one desktop.")

    print("\n  WHAT THIS IS NOT: a load test. One client, no concurrency, no")
    print("  contention, synthetic notes far shorter and more regular than real")
    print("  discharge summaries. Real throughput needs a stated concurrency")
    print("  and a real note-length distribution, and would be lower.")

    os.makedirs("out", exist_ok=True)
    with open("out/throughput.json", "w") as fh:
        json.dump([{"configuration": k, "seconds": v, "docs_per_sec": n_docs / v}
                   for k, v in rows], fh, indent=2)
    print("\nwrote out/throughput.json")
    return rows


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def demo(port=8085):
    import threading
    import urllib.error
    import urllib.request
    from notes import build_corpus

    httpd = serve(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(path, payload, raw=False):
        data = payload if raw else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    print("\n" + "=" * 76)
    print("EXERCISING THE API")
    print("=" * 76)

    note = ("FAMILY HISTORY: No known history of colon cancer.\n"
            "PAST MEDICAL HISTORY: asthma, GERD.\n"
            "ASSESSMENT AND PLAN: pneumonia, treat empirically.")
    _c, body = post("/extract", {"text": note})
    print("\n  POST /extract")
    print(f"    {'concept':<16}{'assertion':<12}{'presence':<10}"
          f"{'experiencer':<13}{'temporality':<12}source")
    for e in body["entities"]:
        print(f"    {e['concept']:<16}{e['assertion']:<12}{e['presence']:<10}"
              f"{e['experiencer']:<13}{e['temporality']:<12}"
              f"{e['assertion_source']}")
    print(f"    sections: {[s['section'] for s in body['sections']]}")
    print(f"    assertions taken from a HEADER rather than words: "
          f"{len(body['section_defaults_applied'])}")
    print("\n    The first row is the point. 'No known history of colon cancer'")
    print("    under a FAMILY HISTORY header is absent AND family AND")
    print("    historical. The single `assertion` field can only hold one of")
    print("    those, which is why all three are returned.")

    _c, body = post("/extract/batch",
                    {"documents": build_corpus(n=5, seed=3)[:3], "fhir": True})
    print(f"\n  POST /extract/batch -> {body['n_documents']} documents, "
          f"{body['n_entities']} entities, {body['n_failed']} failed")

    _c, body = post("/extract/batch",
                    {"documents": ["Pneumonia on chest x-ray.", "", "  "]})
    print(f"\n  batch with two empty documents -> {body['n_failed']} failed, "
          f"{body['n_documents'] - body['n_failed']} succeeded")
    print("    A bad document does not fail the batch. Making the caller")
    print("    re-send every other note to recover from one empty string means")
    print("    re-transmitting PHI to fix someone else's bug.")

    # the log test, live
    planted = ("Patient JANE DOE, MRN 88213771, seen 03/11/2024 for pneumonia. "
               "Contact 617-555-0142.")
    post("/extract", {"text": planted})
    post("/extract", {"nope": 1})
    _c, bad = post("/extract", b"{not json at all: JANE DOE MRN 88213771",
                   raw=True)
    print(f"\n  malformed JSON -> {bad['error']}")
    print(f"    {bad['note']}")

    _c, log = post("/extract", {"text": "x"})
    import urllib.request as _u
    with _u.urlopen(base + "/log", timeout=10) as r:
        log_text = r.read().decode()

    leaks = [t for t in ("JANE", "DOE", "88213771", "617-555-0142", "pneumonia",
                         "03/11/2024") if t in log_text]
    print("\n  THE LOG, after a request containing planted identifiers:")
    print(f"    identifiers found in the log: {leaks or 'NONE'}")
    print(f"    a log record: "
          f"{json.dumps(json.loads(log_text)['records'][-3])}")
    print("    Numbers, enums and ids this service generated. No field accepts")
    print("    caller text, so there is no redaction pass to get wrong -- and")
    print("    the JSON error above recorded its exception CLASS, not its")
    print("    message, because the message quotes the note.")

    httpd.shutdown()
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.demo:
        demo(a.port)
    elif a.bench:
        bench(a.docs, a.workers)
    else:
        serve(a.port).serve_forever()
