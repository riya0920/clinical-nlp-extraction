# Annotation guideline — conditions, medications, allergies, assertions

Version 1.0. Single annotator. This document is a deliverable, not a note to
self: in a real clinical-NLP team the guideline is the thing that makes two
annotators produce the same labels, and it is written *before* annotation
because writing it afterwards means describing what you happened to do.

---

## 1. What gets annotated

Three entity kinds:

| kind | annotate | do not annotate |
|---|---|---|
| **condition** | diagnoses, symptoms, findings | procedures, lab values, anatomy, severity words |
| **medication** | drug names, generic or brand | drug classes ("a statin"), devices, IV fluids |
| **allergy** | the allergen, inside an allergy/intolerance context | the reaction ("rash", "hives") |

Annotate the **concept**, not the string. `DM2`, `type 2 diabetes`, and
`T2DM` are all `t2dm`. Brand names take the generic concept: `Lipitor` is
`atorvastatin`.

**Longest match wins.** `type 2 diabetes` is `t2dm`, never `diabetes_unspec`.
The specific and unspecified concepts are different codes with different
downstream consequences — risk adjustment prices them differently, and a
care-gap measure for diabetics may or may not include the unspecified concept
depending on its value set.

---

## 2. The assertion axis

Every condition takes exactly one assertion. The question the annotator asks
is always the same: **does this patient have this, right now?**

| assertion | meaning | example |
|---|---|---|
| `present` | patient has it now | "The patient has pneumonia." |
| `absent` | explicitly negated for this patient | "No evidence of pneumonia." |
| `possible` | asserted with uncertainty | "Concerning for pneumonia." |
| `historical` | had it; past or resolved | "History of MI in 2019." |
| `family` | someone else has it | "Mother had breast cancer." |
| `hypothetical` | conditional on a future event | "Return if fever develops." |

### Precedence when more than one applies

`family` > `historical` > `hypothetical` > `absent` > `possible` > `present`

This ordering is a clinical judgement rather than an implementation
convenience, and the top of it is the important part: whenever a finding could
be read as belonging to a relative, it belongs to the relative.

---

## 3. The rules that decide the hard cases

**Rule 1 — Assertion is about the patient and about now.**
A condition the patient does not currently have never takes `present`. Which
non-present value it takes depends on *why* it is not present.

**Rule 2 — Family history is never a patient condition.**
"Mother had breast cancer" is `breast_cancer`/`family`. Never `present`, never
`historical`. `historical` would be wrong in a specific and dangerous way: it
asserts the patient once had breast cancer.

**Rule 3 — A negated family history stays `family`.**
"No family history of colon cancer" is `colon_cancer`/`family`, not `absent`.

*This rule is genuinely arguable* and is the one most likely to cause
annotator disagreement. The reasoning for the side taken: the negation scopes
over *the family history*, which is a statement about relatives, so the
finding is still not about the patient. The opposing reading — that a negated
family history should be `absent` because nothing is being asserted of anyone
— is defensible. What matters is that the downstream consumer must never see a
patient `Condition` either way, and both readings agree on that. Rule 3 exists
so the annotation is *consistent*, not because it is provably correct.

**Rule 4 — A symptom given as a medication trigger is `hypothetical`.**
"Albuterol PRN for wheezing" does not assert current wheezing; it describes a
condition under which the drug is taken.

**Rule 5 — Medications are annotated regardless of assertion.**
A held, discontinued, or refused drug is still a medication entity. The
assertion carries the status, and the FHIR layer maps it
(`absent` → `not-taken`, `historical` → `completed`). Dropping the entity loses
the information that the drug was discussed at all.

**Rule 6 — PRN is not "not taken".**
A PRN drug is prescribed. It maps to an active MedicationStatement with
`asNeededBoolean: true`.

**Rule 7 — Allergen only inside an allergy context.**
"Penicillin 500 mg qid" is a medication. "Allergic to penicillin" is an
allergy. Context decides the kind.

**Rule 8 — Do not annotate the reaction.**
In "allergic to penicillin, rash", annotate `penicillin` only.

---

## 4. Worked example

> "No fever, denies chest pain, history of MI, mother had breast cancer."

| span | kind | concept | assertion | why |
|---|---|---|---|---|
| fever | condition | `fever` | `absent` | negated by "no" |
| chest pain | condition | `chest_pain` | `absent` | negated by "denies" |
| MI | condition | `myocardial_infarction` | `historical` | "history of" |
| breast cancer | condition | `breast_cancer` | `family` | "mother had" — rule 2 |

Four entities, four different assertions, one sentence. This is why the
assertion axis cannot be an afterthought.

---

## 5. Known limitations of this guideline

Stated because a guideline that claims to be complete is lying:

- **One annotator.** No inter-annotator agreement was computed. The spec asks
  for a repeated pass weeks later to estimate intra-annotator kappa; that is
  not achievable in a single session and **no kappa is reported** rather than
  a number being invented. Rule 3 is where drift would most likely show.
- **No severity, no temporality beyond historical/present, no anatomical
  site, no laterality, no experiencer beyond patient/family.** Real
  risk-adjustment and RWE work need several of these.
- **No document-level reconciliation.** If a note says "no chest pain" in the
  HPI and "chest pain" in the assessment, both are annotated as written. Which
  one is true is a clinical judgement the annotator is not qualified to make,
  and a real pipeline needs an explicit document-level policy.
- **Section context is not used.** A finding under "FAMILY HISTORY:" should
  arguably default to `family` regardless of phrasing. The guideline does not
  do this, and neither does the extractor.
