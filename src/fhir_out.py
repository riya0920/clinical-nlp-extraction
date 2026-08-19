"""Extraction output -> FHIR R4 resources.

This is the handshake that makes the project a healthcare project rather than
an NLP project: the output is something an EHR could ingest, and the assertion
axis maps onto real FHIR status fields rather than being flattened away.

THE MAPPING, AND WHY EACH CHOICE
--------------------------------
| assertion   | resource                | clinicalStatus | verificationStatus |
|-------------|-------------------------|----------------|--------------------|
| present     | Condition               | active         | confirmed          |
| historical  | Condition               | resolved       | confirmed          |
| possible    | Condition               | active         | provisional        |
| absent      | Condition               | (omitted)      | refuted            |
| family      | **FamilyMemberHistory** | n/a            | n/a                |
| hypothetical| **nothing**             | n/a            | n/a                |

Three of those rows are the ones that get checked in an interview:

**family -> FamilyMemberHistory, never Condition.** "Mother had breast cancer"
describes a different person. Emitting it as a patient Condition puts breast
cancer on the patient's problem list, where it will be read by a clinician, a
risk-adjustment process, and possibly an underwriting or care-gap workflow. It
is the single most consequential error this kind of pipeline can make, and it
is a one-line mistake to make.

**absent -> Condition with verificationStatus=refuted, and clinicalStatus
omitted.** FHIR requires this: clinicalStatus SHALL NOT be present when
verificationStatus is `entered-in-error`, and it is meaningless for a refuted
condition -- a condition the patient does not have is not "active". Recording
negatives at all is a deliberate choice: "no chest pain" is clinically
informative, and pertinent negatives are why the assertion axis exists.

**hypothetical -> nothing.** "Return if fever develops" is an instruction, not
a finding. Emitting it as a Condition would put fever on the problem list of a
patient who does not have a fever. These are counted and reported by
`run_eval.py` rather than dropped silently.

MEDICATIONS use MedicationStatement.status, which has its own vocabulary:
active for current, not-taken for negated ("not currently on warfarin"), and
active with `asNeeded=true` for PRN. Mapping "prn" to `not-taken` -- tempting,
since the patient may not be taking it today -- would be wrong: the medication
IS prescribed.

SCOPE NOTE: these are hand-built resource dicts, not validated against the FHIR
R4 StructureDefinitions. The official validator is a Java application and is not
run here. `validate_shape()` below checks the handful of invariants this code
could plausibly violate; it is a lint, not a conformance statement, and calling
it validation would be the overclaim.
"""

from __future__ import annotations

SNOMED_URL = "http://snomed.info/sct"
RXNORM_URL = "http://www.nlm.nih.gov/research/umls/rxnorm"
COND_CLIN = "http://terminology.hl7.org/CodeSystem/condition-clinical"
COND_VER = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
ALLERGY_CLIN = "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"

ASSERTION_TO_STATUS = {
    "present": ("active", "confirmed"),
    "historical": ("resolved", "confirmed"),
    "possible": ("active", "provisional"),
    "absent": (None, "refuted"),
}


def _coding(system, code, display):
    return {"system": system, "code": code, "display": display}


def _codeable(ent):
    """A CodeableConcept ALWAYS carries the original text, even when the
    terminology lookup fails. Dropping the text on a mapping miss destroys the
    only evidence a human reviewer could use to fix it."""
    cc = {"text": ent.text}
    if ent.code:
        url = SNOMED_URL if ent.code_system == "SNOMED-CT" else RXNORM_URL
        cc["coding"] = [_coding(url, ent.code, ent.code_display)]
    return cc


def _simple(system, code):
    return {"coding": [{"system": system, "code": code}]}


def to_fhir(entities, patient_id="example", note_id="note-1"):
    """Return (resources, dropped) where dropped explains every omission."""
    resources, dropped = [], []
    fam_conditions = []
    subject = {"reference": f"Patient/{patient_id}"}
    n = 0

    for ent in entities:
        n += 1
        rid = f"{note_id}-{n}"
        prov = [{"reference": f"DocumentReference/{note_id}"}]

        if ent.kind == "condition":
            if ent.assertion == "family":
                fam_conditions.append(ent)
                continue
            if ent.assertion == "hypothetical":
                dropped.append({
                    "text": ent.text, "assertion": ent.assertion,
                    "reason": "conditional instruction, not a finding; emitting "
                              "it would put a condition the patient does not "
                              "have on the problem list"})
                continue
            clinical, verification = ASSERTION_TO_STATUS[ent.assertion]
            res = {"resourceType": "Condition", "id": rid, "subject": subject,
                   "code": _codeable(ent),
                   "verificationStatus": _simple(COND_VER, verification),
                   "evidence": [{"detail": prov}]}
            if clinical is not None:
                res["clinicalStatus"] = _simple(COND_CLIN, clinical)
            resources.append(res)

        elif ent.kind == "medication":
            status = {"present": "active", "historical": "completed",
                      "absent": "not-taken", "possible": "unknown",
                      "hypothetical": "active", "family": "unknown"}[ent.assertion]
            res = {"resourceType": "MedicationStatement", "id": rid,
                   "subject": subject, "status": status,
                   "medicationCodeableConcept": _codeable(ent),
                   "derivedFrom": prov}
            dosage = {}
            if "dose" in ent.attrs:
                amount, unit = ent.attrs["dose"].split(" ", 1)
                dosage["doseAndRate"] = [{"doseQuantity": {
                    "value": float(amount), "unit": unit}}]
            if "route" in ent.attrs:
                dosage["route"] = {"text": ent.attrs["route"]}
            if "frequency" in ent.attrs:
                dosage["text"] = ent.attrs["frequency"]
            if ent.assertion == "hypothetical" or \
                    ent.attrs.get("frequency") in ("prn", "as needed"):
                dosage["asNeededBoolean"] = True
            if dosage:
                res["dosage"] = [dosage]
            resources.append(res)

        elif ent.kind == "allergy":
            res = {"resourceType": "AllergyIntolerance", "id": rid,
                   "patient": subject,
                   "clinicalStatus": _simple(ALLERGY_CLIN, "active"),
                   "verificationStatus": _simple(
                       "http://terminology.hl7.org/CodeSystem/"
                       "allergyintolerance-verification", "confirmed"),
                   "code": _codeable(ent)}
            resources.append(res)

    if fam_conditions:
        resources.append({
            "resourceType": "FamilyMemberHistory", "id": f"{note_id}-fmh",
            "status": "completed", "patient": subject,
            "relationship": {"text": "unspecified family member"},
            "condition": [{"code": _codeable(e)} for e in fam_conditions],
        })
    return resources, dropped


def bundle(resources, bundle_id="b1"):
    return {"resourceType": "Bundle", "id": bundle_id, "type": "collection",
            "entry": [{"resource": r} for r in resources]}


# ---------------------------------------------------------------------------
def validate_shape(resources):
    """A lint, NOT FHIR validation. Checks the invariants this code could break.

    The official HL7 validator is a Java application resolving real
    StructureDefinitions and terminology servers; it is not run here and no
    claim of conformance is made. What this catches is the specific class of
    error a hand-built mapper produces.
    """
    problems = []
    for r in resources:
        rt = r.get("resourceType")
        if not rt:
            problems.append("resource with no resourceType")
            continue
        if rt == "Condition":
            if "subject" not in r:
                problems.append(f"{r.get('id')}: Condition.subject is required")
            code = r.get("code", {})
            if not code.get("text") and not code.get("coding"):
                problems.append(f"{r.get('id')}: Condition.code is empty")
            ver = (r.get("verificationStatus", {}).get("coding") or [{}])[0].get("code")
            if ver == "refuted" and "clinicalStatus" in r:
                problems.append(
                    f"{r.get('id')}: clinicalStatus present on a refuted "
                    f"Condition -- a condition the patient does not have "
                    f"cannot be 'active'")
            if ver not in {"unconfirmed", "provisional", "differential",
                           "confirmed", "refuted", "entered-in-error"}:
                problems.append(f"{r.get('id')}: bad verificationStatus {ver!r}")
        elif rt == "MedicationStatement":
            if r.get("status") not in {"active", "completed", "entered-in-error",
                                       "intended", "stopped", "on-hold",
                                       "not-taken", "unknown"}:
                problems.append(f"{r.get('id')}: bad MedicationStatement.status "
                                f"{r.get('status')!r}")
            if "medicationCodeableConcept" not in r:
                problems.append(f"{r.get('id')}: medication[x] is required")
        elif rt == "FamilyMemberHistory":
            if r.get("status") not in {"partial", "completed",
                                       "entered-in-error", "health-unknown"}:
                problems.append(f"{r.get('id')}: bad FamilyMemberHistory.status")
            if "relationship" not in r:
                problems.append(f"{r.get('id')}: relationship is required")
    return problems


def patient_conditions(resources):
    """Every condition asserted OF THE PATIENT. Used by the test that proves
    family history never lands here."""
    out = []
    for r in resources:
        if r.get("resourceType") == "Condition":
            out.append(r["code"].get("text", "").lower())
    return out
