"""Tests for section context, the three assertion axes, and the service.

The log tests are the ones that matter. Everything else here checks that the
extractor got an answer right; those check that a failure mode with no visible
symptom did not happen, which is the only kind worth writing a test for.
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

import sections as S
import serve as SV
from extract import collapse_assertion, extract


def _by_concept(text):
    return {e.concept: e for e in extract(text)}


# --------------------------------------------------------------------------
# the negation-loss bug
# --------------------------------------------------------------------------

def test_negation_survives_a_history_trigger_in_the_same_sentence():
    """THE BUG. `historical` outranked `absent` in the old priority list, so a
    negated statement carrying a history trigger came out positive. "PMH:
    Denies asthma" reported the patient as having asthma."""
    e = _by_concept("PMH: Denies asthma.")["asthma"]
    assert e.assertion == "absent"
    assert e.presence == "absent"
    assert e.temporality == "historical"      # both facts, not one


def test_negation_and_family_are_both_recorded():
    """"No family history of colon cancer" is absent AND family. A single
    label can only hold one of them, which is the argument for the axes."""
    e = _by_concept("No family history of colon cancer.")["colon_cancer"]
    assert e.presence == "absent"
    assert e.experiencer == "family"
    # the collapsed label follows ANNOTATION_GUIDELINE.md Rule 3
    assert e.assertion == "family"


def test_collapse_never_reports_present_for_a_negated_finding():
    """The property that matters more than any specific label: no combination
    of axes may collapse to `present` when presence is absent."""
    for exp in ("patient", "family"):
        for temp in ("current", "historical", "hypothetical"):
            assert collapse_assertion("absent", exp, temp) != "present"
            assert collapse_assertion("possible", exp, temp) != "present"


def test_collapse_is_documented_rule_3_not_an_accident():
    assert collapse_assertion("absent", "family", "historical") == "family"
    assert collapse_assertion("absent", "patient", "historical") == "absent"


# --------------------------------------------------------------------------
# scope bounding
# --------------------------------------------------------------------------

def test_negation_stops_at_the_next_clause_trigger():
    """The second bug, which the first was hiding. A comma-chained telegraphic
    sentence: each clause owns its own assertion."""
    got = _by_concept("No fever, denies chest pain, history of MI, "
                      "mother had breast cancer.")
    assert got["fever"].assertion == "absent"
    assert got["chest_pain"].assertion == "absent"
    assert got["myocardial_infarction"].assertion == "historical"
    assert got["breast_cancer"].assertion == "family"


def test_negation_still_carries_across_a_plain_list():
    """The case that stops the fix from being 'split on commas': with no second
    trigger, negation genuinely does scope over the whole list."""
    got = _by_concept("No fever, chills, or cough.")
    assert got["fever"].assertion == "absent"
    assert got["cough"].assertion == "absent"


def test_an_adjacent_trigger_does_not_truncate_the_scope():
    """Regression on the fix's own first version. "no" and "family history of"
    are consecutive words of one predicate; truncating the negation at the
    family trigger reinstated the inversion the fix had just removed."""
    e = _by_concept("No family history of breast cancer.")["breast_cancer"]
    assert e.presence == "absent"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def test_a_header_supplies_a_default_only_where_the_trigger_cannot_reach():
    """THE SECTION LAYER DOES LESS THAN IT LOOKS LIKE IT DOES, and this test
    is where that became visible.

    "PMH: asthma, obesity." already resolves to `historical` WITHOUT any
    section layer, because "PMH" is a ConText trigger in lexicon.py that scopes
    forward over the rest of the sentence. The header and the trigger are the
    same string, so the sentence layer gets there first.

    The section layer only earns its place across SENTENCE boundaries, where
    the trigger's scope has ended and the header's has not."""
    ents, changes = S.extract_with_sections("PMH: asthma, obesity.", extract)
    assert all(e.assertion == "historical" for e in ents)
    assert changes == [], "the sentence trigger already covered this"

    text = "PMH: asthma.\nAlso carries a diagnosis of GERD."
    ents, changes = S.extract_with_sections(text, extract)
    got = {e.concept: e.assertion for e in ents}
    assert got["gerd"] == "historical"
    assert [c["concept"] for c in changes] == ["gerd"]


def test_an_explicit_trigger_beats_the_section_default():
    """The precedence rule. A section that OVERRODE would turn a negated
    patient statement under FAMILY HISTORY into a positive family history."""
    ents, _c = S.extract_with_sections(
        "FAMILY HISTORY: Mother with breast cancer. Patient denies breast "
        "cancer herself.", extract)
    assertions = sorted(e.assertion for e in ents)
    assert assertions == ["absent", "family"]


def test_sections_run_to_the_next_header_not_the_next_line():
    text = ("PAST MEDICAL HISTORY: asthma.\nAlso carries a diagnosis of GERD.\n"
            "ASSESSMENT AND PLAN: pneumonia.")
    got = {e.concept: e.assertion
           for e in S.extract_with_sections(text, extract)[0]}
    assert got["gerd"] == "historical"        # three lines below the header
    assert got["pneumonia"] == "present"      # after the next header


def test_a_header_like_phrase_mid_sentence_is_not_a_header():
    """"family history of stroke" mid-sentence is a ConText trigger. Treating
    it as a header would put the REST OF THE DOCUMENT into family context."""
    text = "The patient has a family history of stroke. Pneumonia on x-ray."
    assert [n for n, _a, _b in S.find_sections(text)] == ["preamble"]
    got = {e.concept: e.assertion
           for e in S.extract_with_sections(text, extract)[0]}
    assert got["pneumonia"] == "present"


def test_sections_with_no_default_supply_nothing():
    """ASSESSMENT is the most `present` thing in a note. A default there could
    only ever demote it."""
    ents, changes = S.extract_with_sections(
        "ASSESSMENT AND PLAN: pneumonia, treat empirically.", extract)
    assert ents[0].assertion == "present"
    assert changes == []


def test_an_unknown_header_is_not_treated_as_a_section():
    text = "HOSPITAL COURSE: pneumonia treated with antibiotics."
    assert [n for n, _a, _b in S.find_sections(text)] == ["preamble"]


def test_text_before_the_first_header_is_preamble():
    names = [n for n, _a, _b in S.find_sections("Seen today.\nPMH: diabetes.")]
    assert names == ["preamble", "past_medical_history"]


def test_section_changes_are_reported_not_applied_silently():
    """"Which of these came from the header rather than the words" is the first
    question a reviewer asks. A layer that cannot answer it is unauditable."""
    # "Father" is itself a FAMILY trigger, so a sentence containing it
    # never reaches the section layer. The second sentence deliberately
    # has no trigger of any kind -- that is the only situation in which
    # this layer does any work at all.
    text = "FAMILY HISTORY: mother had a stroke.\nAlso lung cancer."
    _e, changes = S.extract_with_sections(text, extract)
    assert [c["concept"] for c in changes] == ["lung_cancer"]
    assert changes[0]["section"] == "family_history"
    assert changes[0]["from"] == "present" and changes[0]["to"] == "family"


def test_the_section_layer_never_touches_a_medication():
    """Notes list ACTIVE home medications under history headers constantly, so
    a header must not be allowed to mark a drug historical.

    Scoped to what this layer controls. The medication below still comes out
    `historical` -- from the SENTENCE trigger "PAST MEDICAL HISTORY", which is
    a separate decision made before this code runs and is not the section
    layer's to override. What is asserted here is that `apply_section_context`
    changed nothing about it."""
    text = "PAST MEDICAL HISTORY: asthma.\nTaking metformin 500 mg daily."
    ents, changes = S.extract_with_sections(text, extract)
    meds = [e for e in ents if e.kind == "medication"]
    assert meds, "fixture should contain a medication"
    assert all(c["concept"] != "metformin" for c in changes)
    assert all(e.kind == "condition"
               for c in changes
               for e in ents if e.concept == c["concept"])


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------

@pytest.fixture
def api():
    SV.LOG.records.clear()
    httpd = HTTPServer(("127.0.0.1", 0), SV.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(base, path, payload, raw=False):
    data = payload if raw else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


PLANTED = ("Patient JANE DOE, MRN 88213771, born 04/11/1963, seen for "
           "pneumonia. Call 617-555-0142 or jane.doe@example.com.")
IDENTIFIERS = ["JANE", "DOE", "88213771", "04/11/1963", "617-555-0142",
               "jane.doe@example.com", "pneumonia"]


def test_the_log_contains_no_request_text(api):
    """THE TEST THIS SERVICE EXISTS FOR. Plant identifiers, exercise every
    path including the error paths, and assert none of them reach the log."""
    _post(api, "/extract", {"text": PLANTED})
    _post(api, "/extract/batch", {"documents": [PLANTED, ""]})
    _post(api, "/extract", {"wrong_field": PLANTED})
    _post(api, "/extract", ("{not json " + PLANTED).encode(), raw=True)
    _get(api, "/health")

    log_text = SV.LOG.as_text()
    assert log_text, "nothing was logged, so the test proves nothing"
    for token in IDENTIFIERS:
        assert token not in log_text, f"{token!r} leaked into the log"


def test_log_records_have_only_the_fixed_fields(api):
    """No field accepts caller text, so there is no redaction pass to get
    wrong. This test is what keeps that true as fields get added."""
    _post(api, "/extract", {"text": "Pneumonia on x-ray."})
    for rec in SV.LOG.records:
        assert set(rec) == set(SV.SafeLog.FIELDS)


def test_a_malformed_json_error_records_a_class_not_a_message(api):
    """A JSONDecodeError message quotes the offending characters, which are
    note text. The class routes an incident; the message is a PHI vector."""
    code, body = _post(api, "/extract",
                       ("{broken " + PLANTED).encode(), raw=True)
    assert code == 400
    assert SV.LOG.records[-1]["error_class"] == "JSONDecodeError"
    for token in IDENTIFIERS:
        assert token not in json.dumps(body)


def test_the_log_still_records_what_operations_needs(api):
    """A log that leaks nothing and says nothing is not a win."""
    _post(api, "/extract", {"text": "Pneumonia on chest x-ray."})
    rec = SV.LOG.records[-1]
    assert rec["status"] == 200
    assert rec["n_bytes"] > 0
    assert rec["n_entities"] >= 1
    assert rec["duration_ms"] >= 0
    assert rec["request_id"]


def test_extract_returns_all_three_axes(api):
    _c, body = _post(api, "/extract",
                     {"text": "No family history of colon cancer."})
    e = body["entities"][0]
    assert e["presence"] == "absent"
    assert e["experiencer"] == "family"
    assert e["temporality"] == "historical"


def test_every_response_carries_the_leakage_rate(api):
    """A fifth of negations are lost. A consumer treating a negative assertion
    as a clinical rule-out has to be told that in the response, not a README."""
    _c, body = _post(api, "/extract", {"text": "Pneumonia on x-ray."})
    assert "22.2%" in body["disclaimer"]
    assert "rule-out" in body["disclaimer"]


def test_empty_text_is_refused(api):
    code, _b = _post(api, "/extract", {"text": "   "})
    assert code == 400


def test_missing_text_field_is_refused(api):
    code, body = _post(api, "/extract", {"note": "x"})
    assert code == 400 and "text" in body["error"]


def test_oversized_payload_is_refused(api):
    code, body = _post(api, "/extract", {"text": "x" * (SV.MAX_BYTES + 100)})
    assert code == 413
    assert body["max_bytes"] == SV.MAX_BYTES


def test_one_bad_document_does_not_fail_the_batch(api):
    """Making the caller re-send every other note to recover from one empty
    string means re-transmitting PHI to fix someone else's bug."""
    _c, body = _post(api, "/extract/batch",
                     {"documents": ["Pneumonia on x-ray.", "", "  "]})
    assert body["n_failed"] == 2
    assert any(r["ok"] for r in body["results"])


def test_batch_preserves_caller_supplied_ids(api):
    _c, body = _post(api, "/extract/batch", {"documents": [
        {"id": "note-a", "text": "Pneumonia."},
        {"id": "note-b", "text": "No fever."}]})
    assert [r["id"] for r in body["results"]] == ["note-a", "note-b"]


def test_batch_can_be_asked_for_fhir(api):
    _c, body = _post(api, "/extract/batch",
                     {"documents": ["Pneumonia on x-ray."], "fhir": True})
    assert body["results"][0]["fhir"]["resourceType"] == "Bundle"


def test_sections_can_be_switched_off(api):
    _c, on = _post(api, "/extract", {"text": "PMH: asthma."})
    _c, off = _post(api, "/extract", {"text": "PMH: asthma.",
                                      "sections": False})
    assert on["entities"][0]["assertion"] == "historical"
    assert off["section_defaults_applied"] == []


def test_about_route_states_the_logging_policy(api):
    _c, body = _get(api, "/about")
    assert body["logging"]["request_text_logged"] is False
    assert set(body["assertion_axes"]) == {"presence", "experiencer",
                                           "temporality"}


def test_unknown_route_lists_the_real_ones(api):
    code, body = _get(api, "/parse")
    assert code == 404 and "POST /extract" in body["routes"]
