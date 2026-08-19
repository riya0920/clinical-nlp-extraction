"""Hand-authored adversarial gold standard.

WHY THIS EXISTS SEPARATELY FROM THE GENERATED NOTES
---------------------------------------------------
`src/notes.py` generates notes from templates and can therefore emit its own
ground truth for free. Evaluating the extractor on that is nearly circular: the
generator and the extractor were written by the same person on the same day,
so the templates and the lexicon agree by construction, and a high F1 measures
that agreement rather than any capability.

So there are two evaluation sets and they answer different questions:

  GENERATED  -- large, free labels, tells you about throughput and regression.
                Its F1 is an upper bound with no external validity.
  ADVERSARIAL (this file) -- 74 sentences I wrote deliberately to BREAK the
                rule layer, annotated by hand, several of which I expected to
                fail when I wrote them. Its F1 is the honest number.

ANNOTATION GUIDELINE
--------------------
See `docs/ANNOTATION_GUIDELINE.md`. The rules that decide the hard cases:

  1. Assertion is about THE PATIENT AND NOW. A condition the patient does not
     currently have takes a non-present assertion, and which one depends on
     why: absent (negated), historical (had it, resolved or past), family
     (someone else has it), hypothetical (conditional on a future event),
     possible (asserted with uncertainty).
  2. Family history is never a patient condition. "Mother had breast cancer"
     yields breast_cancer/family, never breast_cancer/present.
  3. A negated family history is still family, not absent. "No family history
     of colon cancer" is a statement about the family, and the negation applies
     to the family history, not to a patient condition. This one is genuinely
     arguable and the guideline picks a side rather than leaving it undefined.
  4. Symptoms mentioned as the reason for a medication ("albuterol PRN for
     wheezing") are hypothetical, because the note is describing a conditional
     trigger, not asserting current wheezing.
  5. Medications are extracted regardless of assertion; a discontinued or
     held medication is still a MedicationStatement with a status, which the
     FHIR layer handles rather than the extractor dropping it.

LIMITATIONS OF THIS GOLD SET, STATED PLAINLY
--------------------------------------------
Single annotator. No second annotator, therefore no inter-annotator agreement.
The spec asks for a repeated annotation pass weeks later to compute
intra-annotator kappa; that is not fakeable in a single session and it is NOT
reported here rather than being invented. What a kappa would have caught --
guideline drift, my own inconsistency on the arguable cases such as rule 3 --
is therefore unmeasured, and any F1 below inherits that uncertainty.
"""

# Each case: (text, [(concept, kind, assertion), ...])
# Only entities the extractor is EXPECTED to find are listed; the eval treats
# anything else it emits as a false positive.
CASES = [
    # ---- the canonical interview sentence ---------------------------------
    ("No fever, denies chest pain, history of MI, mother had breast cancer.",
     [("fever", "condition", "absent"),
      ("chest_pain", "condition", "absent"),
      ("myocardial_infarction", "condition", "historical"),
      ("breast_cancer", "condition", "family")]),

    # ---- straightforward negation -----------------------------------------
    ("No evidence of pneumonia on chest radiograph.",
     [("pneumonia", "condition", "absent")]),
    ("Patient denies shortness of breath.",
     [("dyspnea", "condition", "absent")]),
    ("Chest pain is absent.",
     [("chest_pain", "condition", "absent")]),
    ("Pulmonary embolism was ruled out by CT angiography.",
     [("pulmonary_embolism", "condition", "absent")]),
    ("Negative for fever or cough.",
     [("fever", "condition", "absent"), ("cough", "condition", "absent")]),
    ("The patient is without dyspnea at rest.",
     [("dyspnea", "condition", "absent")]),

    # ---- pseudo-negation: contains a negation phrase, asserts the opposite --
    ("There is no increase in chest pain since admission.",
     [("chest_pain", "condition", "present")]),
    ("Pneumonia cannot be ruled out.",
     [("pneumonia", "condition", "possible")]),
    ("No significant interval change in the known heart failure.",
     [("heart_failure", "condition", "present")]),
    ("Gram negative sepsis is being treated.",
     [("sepsis", "condition", "present")]),
    ("No new pneumonia identified, prior consolidation stable.",
     [("pneumonia", "condition", "present")]),

    # ---- termination terms close the scope early --------------------------
    ("No chest pain but reports dyspnea on exertion.",
     [("chest_pain", "condition", "absent"), ("dyspnea", "condition", "present")]),
    ("Denies fever, however cough persists.",
     [("fever", "condition", "absent"), ("cough", "condition", "present")]),
    ("No cough although dyspnea is worsening.",
     [("cough", "condition", "absent"), ("dyspnea", "condition", "present")]),

    # ---- family history ---------------------------------------------------
    ("Family history of colon cancer in her father.",
     [("colon_cancer", "condition", "family")]),
    ("Mother had breast cancer at age 52.",
     [("breast_cancer", "condition", "family")]),
    ("FH: type 2 diabetes and hypertension.",
     [("t2dm", "condition", "family"), ("hypertension", "condition", "family")]),
    ("Brother with atrial fibrillation.",
     [("atrial_fibrillation", "condition", "family")]),
    # guideline rule 3: negated family history stays family
    ("No family history of breast cancer.",
     [("breast_cancer", "condition", "family")]),
    ("Maternal grandmother with stroke in her 70s.",
     [("stroke", "condition", "family")]),

    # ---- historical -------------------------------------------------------
    ("History of myocardial infarction in 2019.",
     [("myocardial_infarction", "condition", "historical")]),
    ("PMH: COPD, hypertension, type 2 diabetes.",
     [("copd", "condition", "historical"),
      ("hypertension", "condition", "historical"),
      ("t2dm", "condition", "historical")]),
    ("s/p stroke with residual weakness.",
     [("stroke", "condition", "historical")]),
    ("h/o deep vein thrombosis three years ago.",
     [("dvt", "condition", "historical")]),
    ("Asthma since childhood.",
     [("asthma", "condition", "historical")]),

    # ---- hypothetical -----------------------------------------------------
    ("Return to the emergency department if fever develops.",
     [("fever", "condition", "hypothetical")]),
    ("Call the clinic if chest pain recurs.",
     [("chest_pain", "condition", "hypothetical")]),
    ("Instructed to seek care if dyspnea worsens.",
     [("dyspnea", "condition", "hypothetical")]),

    # ---- uncertainty ------------------------------------------------------
    ("Findings concerning for pneumonia.",
     [("pneumonia", "condition", "possible")]),
    ("Possible urinary tract infection, culture pending.",
     [("uti", "condition", "possible")]),
    ("Chest pain, rule out myocardial infarction.",
     [("chest_pain", "condition", "present"),
      ("myocardial_infarction", "condition", "possible")]),
    ("Likely COPD exacerbation.",
     [("copd_exacerbation", "condition", "possible")]),
    ("Suspicious for pulmonary embolism.",
     [("pulmonary_embolism", "condition", "possible")]),

    # ---- present ----------------------------------------------------------
    ("The patient has community acquired pneumonia.",
     [("pneumonia", "condition", "present")]),
    ("Admitted with acute kidney injury and sepsis.",
     [("aki", "condition", "present"), ("sepsis", "condition", "present")]),
    ("Known CHF with reduced ejection fraction.",
     [("heart_failure", "condition", "present")]),

    # ---- synonymy: same concept, different surface form -------------------
    ("DM2 controlled on metformin.",
     [("t2dm", "condition", "present"), ("metformin", "medication", "present")]),
    ("Type 2 diabetes controlled on metformin.",
     [("t2dm", "condition", "present"), ("metformin", "medication", "present")]),
    ("NIDDM noted on prior admission.",
     [("t2dm", "condition", "historical")]),
    # specific must beat unspecified
    ("Type II diabetes with neuropathy.",
     [("t2dm", "condition", "present")]),

    # ---- medications with dose / route / frequency ------------------------
    ("Metformin 500 mg po bid.",
     [("metformin", "medication", "present")]),
    ("Lisinopril 10 mg daily.",
     [("lisinopril", "medication", "present")]),
    ("Furosemide 40 mg IV twice daily.",
     [("furosemide", "medication", "present")]),
    ("Started on ceftriaxone 1 g IV daily and azithromycin 500 mg po daily.",
     [("ceftriaxone", "medication", "present"),
      ("azithromycin", "medication", "present")]),
    ("Insulin glargine 22 units subcutaneous at bedtime.",
     [("insulin_glargine", "medication", "present")]),
    ("Albuterol inhaled prn.",
     [("albuterol", "medication", "hypothetical")]),
    ("Atorvastatin 40 mg qhs.",
     [("atorvastatin", "medication", "present")]),
    ("Apixaban 5 mg bid for atrial fibrillation.",
     [("apixaban", "medication", "present"),
      ("atrial_fibrillation", "condition", "present")]),
    # brand names map to the same concept as generics
    ("Lipitor 40 mg nightly.",
     [("atorvastatin", "medication", "present")]),
    ("Lasix 20 mg po daily.",
     [("furosemide", "medication", "present")]),
    ("Coumadin held prior to procedure.",
     [("warfarin", "medication", "present")]),

    # ---- negated / conditional medications --------------------------------
    ("Not currently on warfarin.",
     [("warfarin", "medication", "absent")]),
    ("Patient denies taking aspirin.",
     [("aspirin", "medication", "absent")]),

    # ---- allergies --------------------------------------------------------
    ("Allergic to penicillin, rash.",
     [("penicillin", "allergy", "present")]),
    ("Allergies: sulfa drugs and codeine.",
     [("sulfonamide", "allergy", "present"), ("codeine", "allergy", "present")]),
    ("Adverse reaction to contrast dye.",
     [("iodinated_contrast", "allergy", "present")]),
    ("No known drug allergies.", []),
    ("Latex allergy documented.",
     [("latex", "allergy", "present")]),

    # ---- mixed, multi-clause ---------------------------------------------
    ("PMH significant for COPD; presents with likely pneumonia; no fever.",
     [("copd", "condition", "historical"),
      ("pneumonia", "condition", "possible"),
      ("fever", "condition", "absent")]),
    ("Father with myocardial infarction at 55; patient denies chest pain.",
     [("myocardial_infarction", "condition", "family"),
      ("chest_pain", "condition", "absent")]),
    ("Known CKD stage 3; no evidence of acute kidney injury.",
     [("ckd", "condition", "present"), ("aki", "condition", "absent")]),
    ("Hypertension on amlodipine 5 mg daily; hyperlipidemia on Lipitor.",
     [("hypertension", "condition", "present"),
      ("amlodipine", "medication", "present"),
      ("hyperlipidemia", "condition", "present"),
      ("atorvastatin", "medication", "present")]),
    ("Denies cough or fever; reports fatigue.",
     [("cough", "condition", "absent"), ("fever", "condition", "absent")]),
    ("Sepsis ruled out; cellulitis treated with antibiotics.",
     [("sepsis", "condition", "absent"), ("cellulitis", "condition", "present")]),

    # ---- cases I expect to be hard, included because they are ------------
    ("Pneumonia versus atelectasis on imaging.",
     [("pneumonia", "condition", "possible")]),
    ("Differential includes pulmonary embolism and pneumonia.",
     [("pulmonary_embolism", "condition", "possible"),
      ("pneumonia", "condition", "possible")]),
    ("She has no history of diabetes but her mother does.",
     [("diabetes_unspec", "condition", "absent")]),
    ("Cannot exclude early heart failure.",
     [("heart_failure", "condition", "possible")]),
    ("Presented with chest pain; MI was ruled out; discharged on aspirin.",
     [("chest_pain", "condition", "present"),
      ("myocardial_infarction", "condition", "absent"),
      ("aspirin", "medication", "present")]),
    ("Obesity and obstructive sleep apnea, on CPAP.",
     [("obesity", "condition", "present"),
      ("sleep_apnea", "condition", "present")]),
    ("GERD symptoms improved on omeprazole 20 mg daily.",
     [("gerd", "condition", "present"),
      ("omeprazole", "medication", "present")]),
    ("Anemia, likely secondary to chronic kidney disease.",
     [("anemia", "condition", "present"), ("ckd", "condition", "present")]),
    ("Depression, on sertraline 50 mg daily.",
     [("depression", "condition", "present"),
      ("sertraline", "medication", "present")]),
    ("Acute pancreatitis; no evidence of sepsis.",
     [("pancreatitis", "condition", "present"),
      ("sepsis", "condition", "absent")]),
]

# Medication attribute gold, keyed by (case index, concept).
MED_ATTRS = {
    (42, "metformin"): {"dose": "500 mg", "route": "po", "frequency": "bid"},
    (43, "lisinopril"): {"dose": "10 mg", "frequency": "daily"},
    (44, "furosemide"): {"dose": "40 mg", "route": "iv", "frequency": "twice daily"},
    (46, "insulin_glargine"): {"dose": "22 units", "route": "subcutaneous",
                               "frequency": "at bedtime"},
    (48, "atorvastatin"): {"dose": "40 mg", "frequency": "qhs"},
}
