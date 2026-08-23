"""Tests for document-level reconciliation and the service's auth.

The reconciliation tests are about what the pipeline REFUSES to decide. A note
that contradicts itself is the ordinary shape of a clinical document, and the
failure this guards against is a pipeline that quietly picks one mention and
destroys the evidence that the note disagreed with itself.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from http.server import HTTPServer

import reconcile as R
import sections as S
import serve as SV
from extract import extract


def _reconcile(note):
    ents = extract(note)
    ents, _c = S.apply_section_context(ents, note)
    return R.reconcile(ents, note, S.find_sections)


def _by_concept(result):
    return {d["concept"]: d for d in result["decisions"]}


# --------------------------------------------------------------------------
# the case the module exists for
# --------------------------------------------------------------------------

def test_a_note_that_contradicts_itself_is_unresolved_not_collapsed():
    """Denied in the HPI, asserted in the assessment. Resolving automatically
    would either assert a finding a clinician denied or drop one they
    asserted."""
    note = ("HISTORY OF PRESENT ILLNESS: Patient denies chest pain.\n"
            "ASSESSMENT AND PLAN: chest pain, likely musculoskeletal.")
    d = _by_concept(_reconcile(note))["chest_pain"]
    assert d["confidence"] == "unresolved"
    assert d["presence"] is None
    assert d["conflict"] is True


def test_an_unresolved_conflict_is_omitted_from_flat_rows_by_default():
    """Omitted rather than guessed. A consumer that wants everything has to
    ask, which makes accepting a guess explicit at the call site."""
    note = ("HISTORY OF PRESENT ILLNESS: Patient denies chest pain.\n"
            "ASSESSMENT AND PLAN: chest pain, likely musculoskeletal.")
    res = _reconcile(note)
    assert "chest_pain" not in [r["concept"] for r in R.to_rows(res)]
    assert "chest_pain" in [r["concept"]
                            for r in R.to_rows(res, include_unresolved=True)]


def test_a_single_mention_needs_no_reconciliation():
    d = _by_concept(_reconcile("ASSESSMENT AND PLAN: pneumonia."))["pneumonia"]
    assert d["confidence"] == "single" and d["n_mentions"] == 1


def test_agreeing_mentions_are_not_a_conflict():
    note = ("HISTORY OF PRESENT ILLNESS: pneumonia suspected.\n"
            "ASSESSMENT AND PLAN: pneumonia.")
    res = _reconcile(note)
    assert res["n_unresolved"] == 0


# --------------------------------------------------------------------------
# section authority
# --------------------------------------------------------------------------

def test_the_assessment_outranks_the_history_when_presence_agrees():
    """The clinician's conclusion outranks the patient's report -- a defensible
    ordering, which is why it lives in a table a site can disagree with."""
    a = R.SECTION_AUTHORITY["assessment_and_plan"]
    for lower in ("history_of_present_illness", "review_of_systems",
                  "chief_complaint", "physical_exam"):
        assert a > R.SECTION_AUTHORITY[lower]


def test_a_higher_authority_mention_wins_when_it_is_not_a_negation_conflict():
    mentions = [
        {"concept": "c", "experiencer": "patient", "presence": "possible",
         "temporality": "current", "assertion": "possible", "start": 0,
         "end": 1, "text": "c", "section": "history_of_present_illness",
         "authority": 40},
        {"concept": "c", "experiencer": "patient", "presence": "present",
         "temporality": "current", "assertion": "present", "start": 50,
         "end": 51, "text": "c", "section": "assessment_and_plan",
         "authority": 100},
    ]
    d = R.reconcile_group(mentions)
    assert d["confidence"] == "authoritative"
    assert d["presence"] == "present"


def test_a_negation_disagreement_is_flagged_even_when_a_section_outranks():
    """Being wrong in this direction puts a condition on a record that a
    clinician denied. No automatic rule is trusted with it."""
    mentions = [
        {"concept": "c", "experiencer": "patient", "presence": "absent",
         "temporality": "current", "assertion": "absent", "start": 0, "end": 1,
         "text": "c", "section": "history_of_present_illness", "authority": 40},
        {"concept": "c", "experiencer": "patient", "presence": "present",
         "temporality": "current", "assertion": "present", "start": 50,
         "end": 51, "text": "c", "section": "assessment_and_plan",
         "authority": 100},
    ]
    d = R.reconcile_group(mentions)
    assert d["confidence"] == "unresolved"
    assert d["presence"] is None


def test_last_mention_wins_would_be_wrong_on_a_return_precaution():
    """"ASSESSMENT: Pneumonia. Return if cough worsens." Last-wins would mark
    the presenting cough hypothetical. Position is a bad proxy for authority."""
    note = ("HISTORY OF PRESENT ILLNESS: productive cough for three days.\n"
            "ASSESSMENT AND PLAN: pneumonia. Return if cough worsens.")
    d = _by_concept(_reconcile(note))["cough"]
    last = max(d["mentions"], key=lambda m: m["start"])
    assert last["temporality"] == "hypothetical"
    # the reconciled answer is not simply the last mention
    assert d["presence"] != "present" or d["confidence"] != "single"


def test_family_and_patient_mentions_are_never_reconciled_together():
    """"Mother had breast cancer" must not resolve a question about the
    patient -- a different experiencer is a different claim."""
    note = ("FAMILY HISTORY: mother had breast cancer.\n"
            "ASSESSMENT AND PLAN: patient denies breast cancer.")
    res = _reconcile(note)
    experiencers = {d["experiencer"] for d in res["decisions"]
                    if d["concept"] == "breast_cancer"}
    assert experiencers == {"family", "patient"}
    assert res["n_unresolved"] == 0        # two separate claims, not a conflict


def test_the_policy_is_stated_in_the_result():
    """A consumer reading only the rows should still be able to find out that
    the document was not collapsed by default."""
    res = _reconcile("ASSESSMENT AND PLAN: pneumonia.")
    assert "contradicts itself" in res["policy"]


def test_conflicts_are_counted_even_when_resolved():
    mentions = [
        {"concept": "c", "experiencer": "patient", "presence": "possible",
         "temporality": "current", "assertion": "possible", "start": 0,
         "end": 1, "text": "c", "section": "chief_complaint", "authority": 20},
        {"concept": "c", "experiencer": "patient", "presence": "present",
         "temporality": "current", "assertion": "present", "start": 9,
         "end": 10, "text": "c", "section": "assessment_and_plan",
         "authority": 100},
    ]
    d = R.reconcile_group(mentions)
    assert d["conflict"] is True and d["confidence"] == "authoritative"


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------

@pytest.fixture
def api():
    SV.LOG.records.clear()
    SV.AUTH_TOKEN = "s3cret"
    httpd = HTTPServer(("127.0.0.1", 0), SV.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    SV.AUTH_TOKEN = None


def _post(base, path, payload, token=None):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(payload).encode(),
                                 method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_an_unauthenticated_extraction_is_refused(api):
    """An extraction endpoint takes the NOTE as input, so an unauthenticated
    one is an open channel for submitting PHI to a service you do not
    control."""
    code, body = _post(api, "/extract", {"text": "Pneumonia."})
    assert code == 401
    assert "clinical text as input" in body["why"]


def test_a_wrong_token_is_refused(api):
    code, _b = _post(api, "/extract", {"text": "Pneumonia."}, "wrong")
    assert code == 401


def test_the_right_token_is_accepted(api):
    code, body = _post(api, "/extract", {"text": "Pneumonia on x-ray."},
                       "s3cret")
    assert code == 200 and body["n_entities"] >= 1


def test_a_refused_request_still_logs_no_note_text(api):
    planted = "Patient JANE DOE MRN 88213771 with pneumonia."
    _post(api, "/extract", {"text": planted})
    log = SV.LOG.as_text()
    for token in ("JANE", "DOE", "88213771", "pneumonia"):
        assert token not in log


def test_the_service_returns_reconciliation_when_asked(api):
    code, body = _post(api, "/extract", {
        "text": ("HISTORY OF PRESENT ILLNESS: denies chest pain.\n"
                 "ASSESSMENT AND PLAN: chest pain."),
        "reconcile": True}, "s3cret")
    assert code == 200
    rec = body["reconciliation"]
    assert rec["n_unresolved"] == 1
    assert any(d["confidence"] == "unresolved" for d in rec["decisions"])


def test_reconciliation_is_opt_in(api):
    code, body = _post(api, "/extract", {"text": "Pneumonia."}, "s3cret")
    assert code == 200 and "reconciliation" not in body
