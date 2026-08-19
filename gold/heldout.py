"""HELD-OUT gold set. Scored once. Not used to fix anything.

WHY THIS FILE HAD TO EXIST
--------------------------
The adversarial set in `adversarial.py` scores 75/75, and that number is
worthless as an evaluation, because I obtained it by fixing the extractor until
it passed. Six sentences failed on the first run; I added backward-scoping
lexicons and a trigger-consumption rule until they did not. That is development,
and the resulting score measures how well the rules fit the cases used to write
them -- the rule-based equivalent of reporting training accuracy.

So these sentences were written afterwards, in one pass, without running the
extractor on them, and deliberately using constructions the adversarial set does
not contain: different negation phrasings, section headers, list syntax, tapers
and holds, telegraphic note style, and two clauses whose correct annotation I
consider genuinely arguable.

**The extractor was not modified after this file was scored.** Whatever
`run_eval.py` reports for `heldout` is the honest number, and it is lower than
the adversarial number. That gap is the point of the file.
"""

CASES = [
    # different negation phrasings than the dev set used
    ("Patient reports no dyspnea at rest or with exertion.",
     [("dyspnea", "condition", "absent")]),
    ("There were no findings of pneumonia.",
     [("pneumonia", "condition", "absent")]),
    ("Review of systems is negative for fever and cough.",
     [("fever", "condition", "absent"), ("cough", "condition", "absent")]),
    ("Workup unremarkable for pulmonary embolism.",
     [("pulmonary_embolism", "condition", "absent")]),

    # section-header style, telegraphic
    ("ASSESSMENT: Sepsis secondary to urinary tract infection.",
     [("sepsis", "condition", "present"), ("uti", "condition", "present")]),
    ("PMH: asthma, GERD, obesity.",
     [("asthma", "condition", "historical"), ("gerd", "condition", "historical"),
      ("obesity", "condition", "historical")]),
    ("Social history noncontributory. Family history: stroke in mother.",
     [("stroke", "condition", "family")]),
    ("IMPRESSION: probable community acquired pneumonia.",
     [("pneumonia", "condition", "possible")]),

    # medication list syntax
    ("Home meds: lisinopril 20 mg daily, metformin 1000 mg bid, aspirin 81 mg daily.",
     [("lisinopril", "medication", "present"),
      ("metformin", "medication", "present"),
      ("aspirin", "medication", "present")]),
    ("Discharge medications include apixaban 5 mg twice daily and metoprolol 25 mg bid.",
     [("apixaban", "medication", "present"),
      ("metoprolol", "medication", "present")]),
    ("Prednisone taper starting at 40 mg daily.",
     [("prednisone", "medication", "present")]),
    ("Levothyroxine 88 mcg po daily on an empty stomach.",
     [("levothyroxine", "medication", "present")]),
    ("Oxycodone 5 mg q4h prn for pain.",
     [("oxycodone", "medication", "hypothetical")]),

    # temporal / historical variants not in the dev set
    ("Stroke two years ago with full recovery.",
     [("stroke", "condition", "historical")]),
    ("Remote history of deep vein thrombosis.",
     [("dvt", "condition", "historical")]),
    ("Status post myocardial infarction, now on aspirin and atorvastatin.",
     [("myocardial_infarction", "condition", "historical"),
      ("aspirin", "medication", "present"),
      ("atorvastatin", "medication", "present")]),

    # uncertainty variants
    ("Imaging equivocal for pneumonia.",
     [("pneumonia", "condition", "possible")]),
    ("Presentation worrisome for sepsis.",
     [("sepsis", "condition", "possible")]),
    ("Questionable atrial fibrillation on telemetry.",
     [("atrial_fibrillation", "condition", "possible")]),

    # family variants
    ("Paternal uncle with colon cancer diagnosed at 60.",
     [("colon_cancer", "condition", "family")]),
    ("Her sister has type 2 diabetes.",
     [("t2dm", "condition", "family")]),

    # conditional / discharge instruction variants
    ("Advised to return for worsening chest pain.",
     [("chest_pain", "condition", "hypothetical")]),
    ("Watch for signs of cellulitis at the IV site.",
     [("cellulitis", "condition", "hypothetical")]),

    # allergies
    ("Allergies: penicillin (hives), shellfish.",
     [("penicillin", "allergy", "present"), ("shellfish", "allergy", "present")]),
    ("Intolerance to metformin with GI upset.",
     [("metformin", "allergy", "present")]),

    # multi-clause with termination
    ("No acute kidney injury, but chronic kidney disease is stable.",
     [("aki", "condition", "absent"), ("ckd", "condition", "present")]),
    ("Afebrile and denies cough, though dyspnea persists.",
     [("cough", "condition", "absent"), ("dyspnea", "condition", "present")]),

    # constructions I expect to be hard
    ("Neither pneumonia nor heart failure explains the hypoxia.",
     [("pneumonia", "condition", "absent"),
      ("heart_failure", "condition", "absent")]),
    ("Hypertension, poorly controlled despite amlodipine and HCTZ.",
     [("hypertension", "condition", "present"),
      ("amlodipine", "medication", "present"), ("hctz", "medication", "present")]),
    ("Diabetes is well controlled; no evidence of end-organ damage.",
     [("diabetes_unspec", "condition", "present")]),
]

MED_ATTRS = {
    (8, "lisinopril"): {"dose": "20 mg", "frequency": "daily"},
    (11, "levothyroxine"): {"dose": "88 mcg", "route": "po", "frequency": "daily"},
    (12, "oxycodone"): {"dose": "5 mg", "frequency": "q4h"},
}
