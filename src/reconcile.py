"""Document-level reconciliation: what to do when a note contradicts itself.

WHAT THE README SAID WAS MISSING
--------------------------------
"No document-level reconciliation. If a note says 'no chest pain' in the HPI
and 'chest pain' in the assessment, both are annotated as written. Which one is
true is a clinical judgement the annotator is not qualified to make, and a real
pipeline needs an explicit document-level policy."

THE POLICY, AND WHY IT IS NOT "PICK THE LATER ONE"
---------------------------------------------------
A note contradicting itself is not a parsing failure. It is the ordinary shape
of a clinical document: the HPI records what the patient reported on arrival,
the exam records what was found, and the assessment records what the clinician
concluded. Those can disagree for good reasons, and the disagreement is
information.

So the default here is NOT to resolve. `reconcile()` returns a decision with a
`confidence`, and the two most common outcomes are:

    AUTHORITATIVE   one mention sits in a section that outranks the others.
                    The ASSESSMENT is the clinician's conclusion and outranks
                    the HPI, which is the patient's report. This is a defensible
                    ordering, not a fact, and SECTION_AUTHORITY says so.

    UNRESOLVED      the mentions conflict with no authority to separate them.
                    Reported as a conflict for a human, NOT silently collapsed.

The failure mode this exists to prevent is a pipeline that quietly picks one
mention, emits a single Condition, and destroys the evidence that the note
disagreed with itself. Downstream, nobody can tell a clean note from a
contradictory one -- and a contradictory note is exactly the one a reviewer
should see.

WHY LAST-MENTION-WINS IS WRONG
-------------------------------
It is the obvious rule and it inverts on the most common real pattern:

    "ASSESSMENT: Pneumonia. Return if cough worsens."

The last mention of cough is hypothetical, in a return-precaution sentence, and
last-wins would mark the patient's presenting cough as hypothetical. Position
is a proxy for authority and a bad one; the section is the actual signal.

NEGATION NEVER LOSES QUIETLY
-----------------------------
When mentions disagree about PRESENCE specifically -- one says absent, another
says present -- the result is always flagged even if a section outranks. Being
wrong in that direction puts a condition on a patient's record that a clinician
denied, and no automatic rule should be trusted with it.
"""

from __future__ import annotations

from collections import defaultdict

# Higher outranks lower. A DEFENSIBLE ORDERING, NOT A FACT -- and the reason it
# is a table rather than an `if` chain is so a site can disagree with it.
#
# assessment_and_plan is the clinician's conclusion, so it outranks the exam
# (a finding) and the HPI (the patient's report). review_of_systems sits low
# because a ROS negative is a screening question, not an examination.
SECTION_AUTHORITY = {
    "assessment_and_plan": 100,
    "physical_exam": 70,
    "labs": 70,
    "history_of_present_illness": 40,
    "past_medical_history": 40,
    "review_of_systems": 20,
    "chief_complaint": 20,
    "family_history": 0,        # a different EXPERIENCER; never reconciled
    "preamble": 10,
}

PRESENCE_RANK = {"present": 2, "possible": 1, "absent": 0}


def _section_of(entity, sections_fn, text):
    for name, a, b in sections_fn(text):
        if a <= entity.start < b:
            return name
    return "preamble"


def group_mentions(entities, text, sections_fn, kinds=("condition",)):
    """Mentions of the same concept, tagged with their section.

    Grouped by (concept, experiencer). A family-history mention and a
    patient-history mention of the same concept are NOT the same claim and must
    never be reconciled against each other -- doing so would let "mother had
    breast cancer" resolve a question about the patient.
    """
    groups = defaultdict(list)
    for e in entities:
        if e.kind not in kinds:
            continue
        section = _section_of(e, sections_fn, text)
        groups[(e.concept, e.experiencer)].append({
            "concept": e.concept, "experiencer": e.experiencer,
            "presence": e.presence, "temporality": e.temporality,
            "assertion": e.assertion, "start": e.start, "end": e.end,
            "text": e.text, "section": section,
            "authority": SECTION_AUTHORITY.get(section, 10),
        })
    return {k: sorted(v, key=lambda m: m["start"]) for k, v in groups.items()}


def reconcile_group(mentions):
    """One decision for one (concept, experiencer), with its confidence."""
    if len(mentions) == 1:
        m = mentions[0]
        return {"concept": m["concept"], "experiencer": m["experiencer"],
                "presence": m["presence"], "temporality": m["temporality"],
                "n_mentions": 1, "conflict": False, "confidence": "single",
                "basis": "only one mention", "mentions": mentions}

    presences = {m["presence"] for m in mentions}
    conflict = len(presences) > 1
    top = max(m["authority"] for m in mentions)
    winners = [m for m in mentions if m["authority"] == top]
    winner_presences = {m["presence"] for m in winners}

    # NEGATION DISAGREEMENT IS ALWAYS FLAGGED, even when a section outranks.
    # Being wrong here puts a condition on a record that a clinician denied.
    negation_conflict = conflict and ("absent" in presences)

    if len(winners) == 1 and not negation_conflict:
        w = winners[0]
        return {"concept": w["concept"], "experiencer": w["experiencer"],
                "presence": w["presence"], "temporality": w["temporality"],
                "n_mentions": len(mentions), "conflict": conflict,
                "confidence": "authoritative",
                "basis": (f"the {w['section']} mention outranks the others "
                          f"(authority {top})"),
                "mentions": mentions}

    if len(winner_presences) == 1 and not negation_conflict:
        w = winners[0]
        return {"concept": w["concept"], "experiencer": w["experiencer"],
                "presence": w["presence"], "temporality": w["temporality"],
                "n_mentions": len(mentions), "conflict": conflict,
                "confidence": "agreed",
                "basis": "the highest-authority mentions agree",
                "mentions": mentions}

    # UNRESOLVED. Reported, never silently collapsed.
    return {
        "concept": mentions[0]["concept"],
        "experiencer": mentions[0]["experiencer"],
        "presence": None, "temporality": None,
        "n_mentions": len(mentions), "conflict": True,
        "confidence": "unresolved",
        "basis": (
            "mentions disagree about whether the finding is present and no "
            "section authority separates them"
            if negation_conflict and len(winners) > 1 else
            "mentions disagree about PRESENCE. Resolving automatically would "
            "either assert a finding a clinician denied, or drop one they "
            "asserted; both are decisions for a human"),
        "disagreement": sorted(
            {(m["section"], m["presence"]) for m in mentions}),
        "mentions": mentions,
    }


def reconcile(entities, text, sections_fn, kinds=("condition",)):
    groups = group_mentions(entities, text, sections_fn, kinds)
    decisions = [reconcile_group(v) for v in groups.values()]
    return {
        "decisions": sorted(decisions, key=lambda d: d["concept"]),
        "n_concepts": len(decisions),
        "n_conflicts": sum(1 for d in decisions if d["conflict"]),
        "n_unresolved": sum(1 for d in decisions
                            if d["confidence"] == "unresolved"),
        "policy": (
            "The document is NOT collapsed to one row per concept by default. "
            "A note that contradicts itself is the ordinary shape of a "
            "clinical document, and a pipeline that quietly picks one mention "
            "destroys the evidence that it did -- after which nobody "
            "downstream can tell a clean note from a contradictory one, which "
            "is exactly the note a reviewer should see."),
    }


def to_rows(result, include_unresolved=False):
    """Flatten to one row per concept, for a consumer that insists.

    `include_unresolved` DEFAULTS TO FALSE, and that is the safe direction: an
    unresolved conflict is omitted rather than guessed. A consumer that wants
    everything has to ask, and asking is the point -- it makes the decision to
    accept a guess explicit at the call site instead of buried in this module.
    """
    rows = []
    for d in result["decisions"]:
        if d["confidence"] == "unresolved" and not include_unresolved:
            continue
        rows.append({"concept": d["concept"], "experiencer": d["experiencer"],
                     "presence": d["presence"],
                     "temporality": d["temporality"],
                     "confidence": d["confidence"],
                     "n_mentions": d["n_mentions"],
                     "conflict": d["conflict"]})
    return rows
