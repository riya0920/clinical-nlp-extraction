# ML-3 — Clinical NLP: note to structured (~50% build)

**No real clinical text is used anywhere in this project.** Every sentence is
either generated from templates in `src/notes.py` or hand-authored by me in
`gold/`. Real notes are PHI; they cannot go in a repository, cannot be pasted
into a hosted API, and cannot be shown to a colleague for a second opinion.

A ConText assertion layer, terminology mapping with its failures shown, and
FHIR output where family history never becomes a patient condition.

```bash
python run_eval.py            # entity F1, assertion confusion matrix, mapping, FHIR
python compare_assertion.py   # rules vs a learned classifier -- the bakeoff
python write_report.py        # -> docs/EVALUATION.md
python src/notes.py           # one generated note -> entities -> FHIR bundle
python -m pytest tests -q     # 29 tests
```

---

## The five things worth reading

### 1. Two evaluation sets, and only one of them is the number

| set | cases | exact-case | F1 (exact) | relaxed F1 |
|---|---|---|---|---|
| **HELD-OUT** | 30 | **90%** | **0.976** | 0.978 |
| adversarial (dev) | 75 | 100% | 1.000 | 1.000 |

The adversarial set scores 100% and that number is worthless. Six of its cases
failed on the first run and I changed the extractor — adding backward-scoping
lexicons and a trigger-consumption rule — until they passed. That is
development, and the score measures how well the rules fit the cases used to
write them: the rule-based equivalent of reporting training accuracy.

So `gold/heldout.py` was written afterwards, in one pass, using constructions
the dev set does not contain, and **scored once**. Nothing was fixed
afterwards. 90% is the honest number, and the 10-point gap is what a
rule-based clinical NLP system loses on phrasing nobody enumerated.

Exact matching requires concept + kind + assertion; relaxed ignores assertion.
Reporting both separates the ConText layer from the dictionary, which fail for
different reasons.

### 2. The assertion confusion matrix, and why its cells are not interchangeable

| true \ predicted | present | absent | possible | historical | family | hypothetical |
|---|---|---|---|---|---|---|
| **present** | 17 | 0 | 0 | 2 | 0 | 0 |
| **absent** | 2 | 7 | 0 | 0 | 0 | 0 |
| **possible** | 0 | 0 | 4 | 0 | 0 | 0 |
| **historical** | 0 | 0 | 0 | 6 | 0 | 0 |
| **family** | 0 | 0 | 0 | 0 | 3 | 0 |
| **hypothetical** | 0 | 0 | 0 | 0 | 0 | 3 |

- absent → present leakage: **2/9 (22.2%)**
- family → present leakage: **0**

A false `present` in a risk-adjustment pipeline becomes a submitted diagnosis,
and a submitted diagnosis unsupported by the record is potential false-claims
exposure — a legal problem, not a metrics problem. A false `absent` is a missed
condition: worse for the patient clinically, worse for the plan financially, and
not a compliance event.

**Which system consumes the output decides which error you tune against.** A
single F1 averages them as though they were interchangeable, which is how a
model gets shipped to a risk-adjustment team optimised for the wrong error.
That is also the frame for *"your transformer beats rules overall but loses on
negation precision — what ships?"* — and §5 now **measures** it rather than
answering from received wisdom. The measurement partly contradicts the premise.

### 3. Terminology mapping, with the bleed shown

50 distinct concepts extracted, 41 mapped, **hit rate 82.0%**. The misses are
enumerated rather than summarised away:

| kind | concept | n |
|---|---|---|
| condition | `aki` | 3 |
| condition | `colon_cancer` | 2 |
| condition | `cellulitis` | 2 |
| medication | `hctz` | 1 |
| condition | `copd_exacerbation` | 1 |
| medication | `ceftriaxone` | 1 |
| … | | |

Every one of those was **recognised** and could not be coded. The
`CodeableConcept` still carries the original text — deliberately, in
`fhir_out._codeable` — so a human can repair it, but any query filtering on a
code will not see these patients. That is silent data loss rather than an error
anyone gets paged about, which is exactly why concept normalisation is the
unglamorous 90% of production clinical NLP.

The hit rate is a property of the vendored subset in `lexicon.py`, **not** of
SNOMED CT. SNOMED is licensed and RxNorm is a download; neither is vendored.

### 4. FHIR output, where family history is routed correctly

| assertion | resource | clinicalStatus | verificationStatus |
|---|---|---|---|
| present | Condition | active | confirmed |
| historical | Condition | resolved | confirmed |
| possible | Condition | active | provisional |
| absent | Condition | *(omitted)* | refuted |
| family | **FamilyMemberHistory** | — | — |
| hypothetical | **nothing emitted** | — | — |

Three of those rows are the ones that get checked:

**family → FamilyMemberHistory, never Condition.** 12 family-history findings
in the corpus, **0** emitted as a patient Condition. Asserted in
`tests/test_nlp.py`, not merely printed.

**absent → refuted, with clinicalStatus omitted.** A condition the patient does
not have is not "active". Negatives are recorded at all because pertinent
negatives are clinically informative — that is what the assertion axis is for.

**hypothetical → nothing.** "Return if fever develops" is an instruction, not a
finding. 5 such findings were dropped, each with a logged reason rather than
silently.

The shape check in `validate_shape()` is a **lint, not FHIR validation** — the
official HL7 validator is a Java application resolving real StructureDefinitions
and is not run here. Calling it validation would be the overclaim. It has a test
proving it actually catches a bad resource, because a lint nobody has seen fail
is not evidence.

---

### 5. Rules vs a learned model, measured instead of argued

The earlier build could not answer the spec's own question — *"your transformer
beats rules overall but loses on negation precision; what ships to a
risk-adjustment team?"* — because only the rules existed. `compare_assertion.py`
runs the bakeoff. Both systems label the **same entity spans**, so the
comparison isolates the assertion decision.

The learned model is a supervised **n-gram logistic regression** over separated
left/right context. It is **not** a pretrained clinical transformer — no
HuggingFace or BERT weights are available offline, and a transformer trained
from scratch on templated sentences would be a strawman. It is labelled as what
it is everywhere it appears.

Three evaluation sets, split by **contamination**, because the rules were tuned
on one of them:

| set | learned | rules | note |
|---|---|---|---|
| in-distribution (generated) | 97.9% | 100% | rules win **by construction** — the generator's triggers are a subset of the rule lexicon. Circular; shown only to prove the model learned the task. |
| **held-out (clean for both)** | **60.0%** | **93.3%** | **the row that counts** |
| adversarial (rules tuned on it) | 61.0% | 100% | upper bound on the rules' advantage, not a measurement of it |

On the clean held-out set:

| metric | learned | rules |
|---|---|---|
| accuracy | 0.600 | **0.933** |
| precision on `absent` | 0.667 | **1.000** |
| recall on `absent` | 0.667 | **0.778** |

**But the headline hides a reversal that matters more.** Counting the error that
actually costs money — a negated finding reported as `present`:

```
negated finding reported as PRESENT   learned 0/9   rules 2/9
```

**The learned model leaks fewer negations than the rules do.** The rules' two
misses are the `neither X nor Y` construction, absent from their lexicon. So
*"rules win on negation precision"* is **not a law**: rules win when the
phrasing is in the lexicon and fail hard when it is not, while the learned model
degrades more evenly. That is the real shape of the trade-off, and it is the
opposite of what the interview question presumes — which is exactly why the
question is worth measuring rather than answering from received wisdom.

**What ships still depends on the consumer.** To a risk-adjustment team: the
rules, because they fail *legibly* — a missed negation is a lexicon entry
someone can add and a regression test someone can pin, whereas a learned model's
error is a retraining cycle with no guarantee that specific case is fixed. To a
research cohort-building team the answer flips: recall matters more, a human
reviews the cohort anyway, and coverage of unseen phrasing is worth more than
precision. Same measurements, different consumer, different answer.

Honest limits, printed by the script itself: the learned model trains on
templated text so its OOD drop is inflated by the generator's regularity; an
n-gram model is not a transformer and a real clinical transformer would narrow
the gap; and the adversarial row is excluded from the headline for contamination.

## Bugs this evaluation caught

- **Pseudo-negation masking silenced the uncertainty axis.** "Cannot be ruled
  out" was masked to stop it negating, which also stopped it firing `possible`
  — turning *"pneumonia cannot be ruled out"* into a confident diagnosis. The
  mask must apply to the negation axis only.
- **A negation trigger lost to the historical trigger inside it.** "No history
  of diabetes" contains "history of", which outranks `absent` in the priority
  order, so it resolved to `historical`. Fixed by consuming matched negation
  spans before the other axes run.
- **Forward-only scoping missed every trailing modifier.** "Asthma since
  childhood" and "NIDDM noted on prior admission" returned `present` — silently,
  which is the worst possible default.
- **The family-history leak check was corpus-wide, not per-note.** It reported
  7 leaks against a mapper that has none: one note saying "mother had breast
  cancer" and a *different* note saying the patient has breast cancer is two
  patients agreeing, not a bug.
- **Assertion-only errors never reached the error listing.** They are consumed
  by the relaxed-matching pass, so the report showed 1 of the 3 held-out
  failures — hiding the majority of the extractor's mistakes, since finding the
  entity is the easy half.

## Held-out errors, left failing on purpose

Fixing them would convert them into development data and the held-out score
would stop meaning anything. Categorised in
[`docs/EVALUATION.md`](docs/EVALUATION.md):

1. **scope termination** — `s/p MI, now on aspirin` marks active drugs
   historical; `now`/`currently` are not termination terms. Highest severity: it
   turns a live medication list into a historical one.
2. **dictionary partition** — "intolerance to metformin" resolves to a
   medication because allergens are a separate lexicon. Allergen is a *role*, not
   a vocabulary; fixing it is a design change.
3. **lexicon coverage** — "neither X nor Y" is not a trigger. The characteristic
   rule-based failure: unenumerated phrasing, failing silently to `present`.

## What is still missing

- **Still no pretrained transformer.** The bakeoff uses an n-gram logistic
  regression, which is a legitimate pre-transformer baseline but not
  ClinicalBERT. The OOD gap would narrow with real pretraining, and no claim
  about transformer performance is made.
- **No medspaCy / scispaCy.** Not installed; ConText and the NER are
  hand-rolled. Fine for the rule layer, but there is no real linguistic
  parsing — no POS tags, no dependency parse, so medication attribute
  attribution is window-based and provably wrong on coordinated lists.
- **No kappa.** Single annotator, no second pass. Not fakeable in one session,
  so it is absent rather than invented. Rule 3 in the guideline is where drift
  would show.
- **No section-context handling.** A finding under a `FAMILY HISTORY:` header
  should arguably default to `family` regardless of phrasing.
- **No real terminology.** Hand-built SNOMED/RxNorm subsets; no VSAC, no
  version pinning, no inactivated-concept handling.
- **No document-level reconciliation**, no severity, no laterality, no
  temporality beyond historical/present, no coreference.
- **Throughput is 839 sentences/sec single-core** and is not a pipeline claim:
  no batching, no parallelism, no service.

## Files

| path | what |
|---|---|
| `src/lexicon.py` | ConText triggers, clinical dictionaries, terminology subsets |
| `src/extract.py` | sentence splitting, dictionary NER, the ConText algorithm |
| `src/fhir_out.py` | assertion → FHIR resource/status mapping, shape lint |
| `src/notes.py` | synthetic note generator, with its biases documented |
| `gold/adversarial.py` | 75 hand-authored dev sentences |
| `gold/heldout.py` | 30 hand-authored held-out sentences, scored once |
| `run_eval.py` | F1, confusion matrix, mapping, FHIR emission, throughput |
| `src/assertion_ml.py` | labelled-corpus generator + the learned baseline |
| `compare_assertion.py` | the rules-vs-learned bakeoff, split by contamination |
| `docs/ANNOTATION_GUIDELINE.md` | the guideline, written as a deliverable |
| `tests/test_nlp.py` | 29 tests |
