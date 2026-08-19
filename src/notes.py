"""Synthetic clinical note generator.

NO REAL CLINICAL TEXT. Every sentence below is a template I wrote. Real notes
are PHI: they cannot go in a repository, cannot be pasted into a hosted API,
and cannot be shown to a colleague for a second opinion. A clinical-NLP project
that quietly uses real text has failed before it has started, which is why this
file exists rather than a data loader.

THE GENERATION PROMPT AND ITS BIASES
------------------------------------
These are templates, not an LLM, so the "generation prompt" is the template set
itself, and its biases are legible and worth stating because they bound what
any evaluation on generated notes can show:

  * VOCABULARY IS CLOSED. Every condition and drug emitted here is in
    `lexicon.py`. The extractor therefore cannot suffer an out-of-vocabulary
    miss on generated notes, which is the single most common real failure. This
    is why the honest evaluation lives on the hand-written held-out set and
    generated notes are used only for throughput and smoke testing.
  * PHRASING IS REGULAR. Real notes have copy-forward, telegraphic fragments,
    inconsistent abbreviation, dictation errors, misspellings, and section
    headers that vary by author and by EHR template. Almost none of that is
    here.
  * ASSERTIONS ARE CLEAN. One assertion per clause, no nested scopes, no
    "denies chest pain although reports pressure that may represent angina".
  * NEGATIVES ARE UNDER-REPRESENTED relative to real notes, which are dense
    with pertinent negatives.

So: F1 on generated notes is not evidence of capability. It is a regression
signal and a throughput fixture, and `run_eval.py` treats it as one.
"""

from __future__ import annotations

import random

SECTIONS = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY",
            "FAMILY HISTORY", "ALLERGIES", "MEDICATIONS", "ASSESSMENT AND PLAN"]

CHIEF = ["Shortness of breath.", "Chest pain.", "Cough and fever.",
         "Productive cough.", "Leg swelling.", "Fatigue."]

HPI = [
    "The patient is a {age} year old {sex} presenting with {sx} for {days} days.",
    "{age} year old {sex} with {days} days of {sx}.",
    "Patient reports {sx} beginning {days} days ago.",
]

HPI_NEG = ["Denies {sx2}.", "No {sx2}.", "Reports no {sx2}.",
           "Review of systems negative for {sx2}."]

PMH = ["PMH: {c1}, {c2}.", "History of {c1} and {c2}.",
       "Past medical history significant for {c1}."]

FH = ["Family history of {c3} in her mother.", "Father with {c3}.",
      "FH: {c3}.", "Mother had {c3} at age 60."]

ALLERGY = ["Allergies: {al}.", "Allergic to {al}.", "No known drug allergies."]

MEDS = ["{m1} {d1} po daily.", "{m1} {d1} bid.", "Home meds include {m1} {d1} daily.",
        "{m1} {d1} daily and {m2} {d2} bid."]

AP = ["Assessment: likely {c4}.", "Impression: {c4}, treat empirically.",
      "Concerning for {c4}.", "{c4} confirmed on imaging.",
      "No evidence of {c5}."]

SX = ["dyspnea", "chest pain", "cough", "fever"]
CONDS = ["copd", "hypertension", "type 2 diabetes", "CHF", "CKD", "asthma",
         "atrial fibrillation", "GERD"]
FAM_CONDS = ["breast cancer", "colon cancer", "stroke", "type 2 diabetes"]
DX = ["pneumonia", "COPD exacerbation", "urinary tract infection", "sepsis"]
RULEOUT = ["pulmonary embolism", "myocardial infarction", "sepsis"]
DRUGS = [("metformin", "500 mg"), ("lisinopril", "10 mg"), ("atorvastatin", "40 mg"),
         ("furosemide", "20 mg"), ("metoprolol", "25 mg"), ("aspirin", "81 mg"),
         ("omeprazole", "20 mg"), ("sertraline", "50 mg")]
ALLERGENS = ["penicillin", "sulfa", "codeine", "latex"]


def make_note(rng):
    (m1, d1), (m2, d2) = rng.sample(DRUGS, 2)
    fill = {
        "age": rng.randint(28, 88),
        "sex": rng.choice(["male", "female"]),
        "sx": rng.choice(SX), "sx2": rng.choice(SX),
        "days": rng.randint(1, 10),
        "c1": rng.choice(CONDS), "c2": rng.choice(CONDS),
        "c3": rng.choice(FAM_CONDS), "c4": rng.choice(DX),
        "c5": rng.choice(RULEOUT),
        "al": rng.choice(ALLERGENS),
        "m1": m1, "d1": d1, "m2": m2, "d2": d2,
    }
    lines = [
        "CHIEF COMPLAINT: " + rng.choice(CHIEF),
        "HISTORY OF PRESENT ILLNESS: " + rng.choice(HPI).format(**fill)
        + " " + rng.choice(HPI_NEG).format(**fill),
        "PAST MEDICAL HISTORY: " + rng.choice(PMH).format(**fill),
        "FAMILY HISTORY: " + rng.choice(FH).format(**fill),
        "ALLERGIES: " + rng.choice(ALLERGY).format(**fill),
        "MEDICATIONS: " + rng.choice(MEDS).format(**fill),
        "ASSESSMENT AND PLAN: " + rng.choice(AP).format(**fill)
        + " Return if " + rng.choice(SX) + " worsens.",
    ]
    return "\n".join(lines)


def build_corpus(n=200, seed=13):
    rng = random.Random(seed)
    return [make_note(rng) for _ in range(n)]


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extract as E
    import fhir_out
    import json

    notes = build_corpus(3)
    for i, note in enumerate(notes[:1]):
        print(note)
        print("-" * 70)
        ents = E.extract(note)
        for e in ents:
            code = f"{e.code_system} {e.code}" if e.code else "UNMAPPED"
            print(f"  {e.kind:<11} {e.concept:<22} {e.assertion:<12} {code}")
        res, dropped = fhir_out.to_fhir(ents, note_id=f"demo{i}")
        print("-" * 70)
        print(json.dumps(fhir_out.bundle(res), indent=2)[:1200])
        for d in dropped:
            print(f"  DROPPED: {d['text']!r} ({d['assertion']}) -- {d['reason']}")
