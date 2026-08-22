"""SECTION-CONTEXT gold set. Scored once. Not used to fix anything.

WHY A THIRD GOLD SET
--------------------
`heldout.py` was scored once and its number recorded, and it already contains
section-header cases. Re-scoring it after building a section layer would reuse
it as development data and destroy the one honest number in this project.

So section context gets its own held-out set, written to the same rule: these
documents were written BEFORE running the section code on them, in one pass,
and the implementation was not changed afterwards. Whatever `run_sections.py`
reports is the honest number.

WHAT IS BEING TESTED
--------------------
Not "does a header change an assertion" -- that is trivially true by
construction. The question is whether the PRECEDENCE rule is right:

    the section supplies a default; an explicit sentence trigger wins.

So the set is built around the collision cases, not the easy ones. Roughly half
of these documents contain a sentence under a header that MUST NOT take the
header's default, and the failure mode they are designed to catch is an
inversion -- a negated patient-level statement under `FAMILY HISTORY` becoming
a positive family history, which in a real system drives a genetic-risk
referral for a disease nobody has.

FOUR CASES ARE MARKED `arguable`. They are scored and reported separately
rather than being quietly dropped or quietly counted, because the honest thing
to do with an annotation I am not confident in is to say which ones they are.

Each case is (document, [(concept, expected_assertion), ...]) and covers only
the CONDITION entities, since `apply_section_context` deliberately does not
touch medications or allergies.

CORRECTIONS MADE AFTER THE FIRST SCORING, AND WHY EACH IS NOT TUNING
--------------------------------------------------------------------
Four annotations were wrong in ways that had nothing to do with the section
layer, and each is recorded here rather than quietly edited:

1. `chf` and `diabetes` are not concept ids in lexicon.py -- the real ones are
   `heart_failure` and `diabetes_unspec`. I mistyped them. Fixing an id I got
   wrong is correcting the annotation, not changing the model.

2. `wheezing` is not in the condition lexicon at all, so that case tested a
   term the extractor was never built to find. It measured lexicon coverage,
   not section context. Swapped for a term that is in scope.

3. TWO CASES CONTRADICTED THIS PROJECT'S OWN ANNOTATION GUIDELINE. I annotated
   negated family history as `absent`; docs/ANNOTATION_GUIDELINE.md Rule 3 says
   it stays `family`, and says so explicitly as a consistency decision. I wrote
   the gold set without re-reading the guideline first, which is the most
   ordinary annotation failure there is and exactly what a second annotator
   would have caught.

The section IMPLEMENTATION was not changed in response to any of these.
"""

CASES = [

    # ---- the header supplying a default, where nothing else does ----------
    ("FAMILY HISTORY: stroke, colon cancer.",
     [("stroke", "family"), ("colon_cancer", "family")], False),

    ("PMH: asthma, obesity.",
     [("asthma", "historical"), ("obesity", "historical")], False),

    ("PAST MEDICAL HISTORY: COPD, hypertension, CKD.",
     [("copd", "historical"), ("hypertension", "historical"),
      ("ckd", "historical")], False),

    ("Family Hx: breast cancer.",
     [("breast_cancer", "family")], False),

    # ---- the sentence trigger overriding the header ----------------------
    # Rule 3 in docs/ANNOTATION_GUIDELINE.md: a negated family history stays
    # `family`. I first annotated this `absent` without checking the
    # guideline. The three-axis output carries both readings
    # (presence=absent, experiencer=family); the collapsed label follows
    # the project contract.
    ("FAMILY HISTORY: No known history of colon cancer.",
     [("colon_cancer", "family")], False),

    ("FAMILY HISTORY: Mother with breast cancer. Patient denies breast cancer "
     "herself.",
     [("breast_cancer", "family"), ("breast_cancer", "absent")], False),

    ("PMH: Denies asthma. Hypertension well controlled.",
     [("asthma", "absent"), ("hypertension", "historical")], False),

    ("PAST MEDICAL HISTORY: possible CHF, not yet confirmed.",
     [("heart_failure", "possible")], False),

    ("FAMILY HISTORY: if stroke develops in a sibling, screen earlier.",
     [("stroke", "hypothetical")], False),

    # ---- sections that must supply NOTHING --------------------------------
    ("ASSESSMENT AND PLAN: Pneumonia, treat empirically.",
     [("pneumonia", "present")], False),

    ("IMPRESSION: sepsis secondary to urinary tract infection.",
     [("sepsis", "present"), ("uti", "present")], False),

    ("CHIEF COMPLAINT: chest pain.",
     [("chest_pain", "present")], False),

    ("REVIEW OF SYSTEMS: negative for fever.",
     [("fever", "absent")], False),

    ("PHYSICAL EXAM: no cough. Lungs otherwise clear.",
     [("cough", "absent")], False),

    # ---- sections run to the NEXT header, not the next line ---------------
    ("PAST MEDICAL HISTORY: asthma.\nPatient has had two exacerbations.\n"
     "Also carries a diagnosis of GERD.\n"
     "ASSESSMENT AND PLAN: pneumonia.",
     [("asthma", "historical"), ("gerd", "historical"),
      ("pneumonia", "present")], False),

    ("FAMILY HISTORY: mother had a stroke.\nFather had a myocardial infarction.\n"
     "MEDICATIONS: lisinopril 10 mg daily.",
     [("stroke", "family"), ("myocardial_infarction", "family")], False),

    # ---- a header-like phrase mid-sentence is NOT a header ----------------
    ("The patient has a family history of stroke. Pneumonia on chest x-ray.",
     [("stroke", "family"), ("pneumonia", "present")], False),

    ("Past medical history is significant for COPD. Today he has pneumonia.",
     [("copd", "historical"), ("pneumonia", "present")], False),

    # ---- no headers at all ------------------------------------------------
    ("Patient presents with cough and fever. No chest pain.",
     [("cough", "present"), ("fever", "present"), ("chest_pain", "absent")],
     False),

    # ---- text before the first header is preamble, not a section ---------
    ("Seen in clinic today for cough.\nPMH: diabetes.",
     [("cough", "present"), ("diabetes_unspec", "historical")], False),

    # ---- unknown header: must not be treated as a section -----------------
    ("HOSPITAL COURSE: pneumonia treated with antibiotics.",
     [("pneumonia", "present")], False),

    # ---- the four I am not confident about --------------------------------

    # Arguable: "screen for X" under family history. Is the CONDITION being
    # asserted about the family, or is this a plan? I annotate hypothetical
    # because the sentence is about a future action, but a reasonable annotator
    # would say the family history is what motivates it.
    ("FAMILY HISTORY: colon cancer. Screen for colon cancer at 40.",
     [("colon_cancer", "family"), ("colon_cancer", "hypothetical")], True),

    # Arguable: a resolved condition under PMH. "historical" is right by the
    # section, but "absent" is defensible for a condition explicitly described
    # as no longer present.
    ("PAST MEDICAL HISTORY: pneumonia, resolved.",
     [("pneumonia", "historical")], True),

    # Arguable: ROS positives. The section implies nothing, so this is
    # `present`, but a "review of systems" positive is a symptom report rather
    # than a clinical finding and some schemes separate those.
    ("REVIEW OF SYSTEMS: positive for cough and dyspnea.",
     [("cough", "present"), ("dyspnea", "present")], True),

    # Arguable: genuinely ambiguous about WHOSE colon cancer is being denied --
    # the family's, or the patient's under a header that does not fit. Annotated
    # `family` under Rule 3, but the three-axis output is the honest answer
    # here: presence=absent, experiencer=family, and no collapse chosen for you.
    ("FAMILY HISTORY: no colon cancer reported.",
     [("colon_cancer", "family")], True),
]
