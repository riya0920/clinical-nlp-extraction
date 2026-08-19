"""Tests for the assertion layer and the FHIR mapping.

The one that matters most is
`test_family_history_never_becomes_a_patient_condition`. Everything else here
is ordinary correctness; that one is the case a screener checks by hand,
because getting it wrong puts a disease the patient does not have onto their
problem list, where a clinician, a risk-adjustment process, and possibly a
care-gap workflow will all read it as fact.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "gold"))

import extract as E
import fhir_out
import notes
from lexicon import ASSERTIONS


def assertions_for(text, concept):
    return [e.assertion for e in E.extract(text) if e.concept == concept]


# ---------------------------------------------------------------------------
# The instant-fail case
# ---------------------------------------------------------------------------
def test_family_history_never_becomes_a_patient_condition():
    for text in [
        "Mother had breast cancer at age 52.",
        "Family history of colon cancer.",
        "Father with myocardial infarction at 55.",
        "FH: type 2 diabetes.",
        "Her sister has type 2 diabetes.",
    ]:
        res, _ = fhir_out.to_fhir(E.extract(text))
        patient_conditions = fhir_out.patient_conditions(res)
        assert patient_conditions == [], (
            f"{text!r} produced patient Condition(s) {patient_conditions}")
        assert any(r["resourceType"] == "FamilyMemberHistory" for r in res)


def test_negated_family_history_is_still_family_not_absent():
    """Guideline rule 3. Arguable, and the guideline picks a side: the negation
    applies to the family history, not to a patient condition. What must NEVER
    happen either way is a patient Condition."""
    text = "No family history of breast cancer."
    assert assertions_for(text, "breast_cancer") == ["family"]
    res, _ = fhir_out.to_fhir(E.extract(text))
    assert fhir_out.patient_conditions(res) == []


def test_hypothetical_findings_are_not_emitted_as_conditions():
    """'Return if fever develops' is an instruction, not a finding."""
    res, dropped = fhir_out.to_fhir(E.extract("Return to the ED if fever develops."))
    assert fhir_out.patient_conditions(res) == []
    assert len(dropped) == 1
    assert dropped[0]["assertion"] == "hypothetical"
    assert "reason" in dropped[0]


# ---------------------------------------------------------------------------
# ConText behaviour
# ---------------------------------------------------------------------------
def test_the_canonical_sentence():
    text = "No fever, denies chest pain, history of MI, mother had breast cancer."
    got = {(e.concept, e.assertion) for e in E.extract(text)}
    assert got == {("fever", "absent"), ("chest_pain", "absent"),
                   ("myocardial_infarction", "historical"),
                   ("breast_cancer", "family")}


def test_pseudo_negation_does_not_negate():
    assert assertions_for("There is no increase in chest pain.", "chest_pain") == ["present"]
    assert assertions_for("No new pneumonia identified.", "pneumonia") == ["present"]


def test_pseudo_negation_still_allows_the_uncertainty_axis():
    """Regression test. 'cannot be ruled out' must be blocked from NEGATING and
    must still fire UNCERTAINTY -- it is an expression of doubt. Masking it
    globally silenced both and turned it into a confident diagnosis."""
    assert assertions_for("Pneumonia cannot be ruled out.", "pneumonia") == ["possible"]
    assert assertions_for("Cannot exclude early heart failure.", "heart_failure") == ["possible"]


def test_termination_closes_the_scope():
    text = "No chest pain but reports dyspnea on exertion."
    assert assertions_for(text, "chest_pain") == ["absent"]
    assert assertions_for(text, "dyspnea") == ["present"]


def test_negation_trigger_outranks_the_historical_trigger_it_contains():
    """Regression test. 'no history of diabetes' literally contains the
    historical trigger 'history of', which outranks absent in
    ASSERTION_PRIORITY and won until negation matches began consuming their
    spans."""
    text = "She has no history of diabetes but her mother does."
    assert assertions_for(text, "diabetes_unspec") == ["absent"]


def test_trailing_modifiers_scope_backward():
    """A forward-only ConText misses these, and misses them silently by
    returning `present`."""
    assert assertions_for("Asthma since childhood.", "asthma") == ["historical"]
    assert assertions_for("NIDDM noted on prior admission.", "t2dm") == ["historical"]
    assert assertions_for("Albuterol inhaled prn.", "albuterol") == ["hypothetical"]


def test_backward_negation():
    assert assertions_for("Sepsis ruled out.", "sepsis") == ["absent"]
    assert assertions_for("Chest pain is absent.", "chest_pain") == ["absent"]


# ---------------------------------------------------------------------------
# Concept normalisation
# ---------------------------------------------------------------------------
def test_synonyms_and_abbreviations_normalise_to_one_concept():
    """'DM2' and 'type 2 diabetes' must be the SAME concept. Emitting both as
    distinct problems is the classic downstream bug: the problem list grows a
    duplicate, and any count of diabetic members double-counts the patient."""
    for surface in ["DM2 controlled on metformin.",
                    "Type 2 diabetes controlled on metformin.",
                    "Type II diabetes with neuropathy.",
                    "T2DM on insulin glargine."]:
        concepts = [e.concept for e in E.extract(surface) if e.kind == "condition"]
        assert "t2dm" in concepts, surface
        assert "diabetes_unspec" not in concepts, (
            f"{surface!r}: the specific concept must beat the unspecified one")


def test_longest_match_wins():
    ents = E.extract("Community acquired pneumonia treated.")
    assert [e.concept for e in ents if e.kind == "condition"] == ["pneumonia"]


def test_brand_names_map_to_the_generic_concept():
    assert [e.concept for e in E.extract("Lipitor 40 mg nightly.")] == ["atorvastatin"]
    assert [e.concept for e in E.extract("Lasix 20 mg po daily.")] == ["furosemide"]


# ---------------------------------------------------------------------------
# Medication attributes
# ---------------------------------------------------------------------------
def test_dose_route_frequency_extraction():
    ents = E.extract("Metformin 500 mg po bid.")
    med = next(e for e in ents if e.kind == "medication")
    assert med.attrs == {"dose": "500 mg", "route": "po", "frequency": "bid"}


def test_prn_medication_is_marked_as_needed_in_fhir():
    res, _ = fhir_out.to_fhir(E.extract("Oxycodone 5 mg q4h prn for pain."))
    ms = next(r for r in res if r["resourceType"] == "MedicationStatement")
    assert ms["status"] == "active", "a PRN drug is prescribed, not 'not-taken'"
    assert ms["dosage"][0]["asNeededBoolean"] is True


def test_negated_medication_is_not_taken():
    res, _ = fhir_out.to_fhir(E.extract("Not currently on warfarin."))
    ms = next(r for r in res if r["resourceType"] == "MedicationStatement")
    assert ms["status"] == "not-taken"


# ---------------------------------------------------------------------------
# FHIR shape
# ---------------------------------------------------------------------------
def test_absent_condition_is_refuted_without_a_clinical_status():
    """FHIR: clinicalStatus is meaningless for a refuted Condition -- a
    condition the patient does not have is not 'active'."""
    res, _ = fhir_out.to_fhir(E.extract("No evidence of pneumonia."))
    cond = next(r for r in res if r["resourceType"] == "Condition")
    assert cond["verificationStatus"]["coding"][0]["code"] == "refuted"
    assert "clinicalStatus" not in cond


def test_assertion_to_status_mapping():
    cases = {"The patient has pneumonia.": ("active", "confirmed"),
             "History of pneumonia.": ("resolved", "confirmed"),
             "Findings concerning for pneumonia.": ("active", "provisional")}
    for text, (clin, ver) in cases.items():
        res, _ = fhir_out.to_fhir(E.extract(text))
        cond = next(r for r in res if r["resourceType"] == "Condition")
        assert cond["clinicalStatus"]["coding"][0]["code"] == clin, text
        assert cond["verificationStatus"]["coding"][0]["code"] == ver, text


def test_codeable_concept_keeps_original_text_even_when_unmapped():
    """Dropping the text on a mapping miss destroys the only evidence a human
    reviewer could use to repair it."""
    res, _ = fhir_out.to_fhir(E.extract("Acute pancreatitis."))
    cond = next(r for r in res if r["resourceType"] == "Condition")
    assert cond["code"]["text"]
    assert "coding" not in cond["code"], "pancreatitis is deliberately unmapped"


def test_mapped_concepts_carry_a_system_uri():
    res, _ = fhir_out.to_fhir(E.extract("The patient has pneumonia."))
    coding = next(r for r in res if r["resourceType"] == "Condition")["code"]["coding"][0]
    assert coding["system"] == "http://snomed.info/sct"
    assert coding["code"] == "233604007"


def test_shape_lint_is_clean_on_generated_notes():
    problems = []
    for note in notes.build_corpus(40):
        res, _ = fhir_out.to_fhir(E.extract(note))
        problems.extend(fhir_out.validate_shape(res))
    assert problems == []


def test_shape_lint_actually_catches_a_bad_resource():
    """A lint nobody has seen fail is not evidence of anything."""
    bad = [{"resourceType": "Condition", "id": "x", "subject": {},
            "code": {"text": "pneumonia"},
            "verificationStatus": {"coding": [{"code": "refuted"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]}}]
    problems = fhir_out.validate_shape(bad)
    assert any("refuted" in p for p in problems)


# ---------------------------------------------------------------------------
def test_every_assertion_value_is_known():
    for note in notes.build_corpus(30):
        for e in E.extract(note):
            assert e.assertion in ASSERTIONS


def test_generated_notes_produce_extractions():
    corpus = notes.build_corpus(20)
    total = sum(len(E.extract(n)) for n in corpus)
    assert total > 60, "generator and extractor have drifted apart"
