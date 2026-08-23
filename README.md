# ML-3 — Clinical NLP: note to structured — complete

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
python run_sections.py        # section context, scored once on a third gold set
python serve.py --demo        # the extraction API, and the log that cannot leak
python serve.py --bench       # throughput: batching, parallelism, machine drift
python -m pytest tests -q     # 77 tests
```

---

## The eight things worth reading

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

### 6. A negation-loss bug that two gold sets could not see

Building the section layer meant writing a third gold set — one aimed
deliberately at *collision* cases, sentences where two assertion axes fire at
once. It found this immediately:

| sentence | old output | correct |
|---|---|---|
| `FAMILY HISTORY: No known history of colon cancer.` | **family** | absent + family |
| `PMH: Denies asthma.` | **historical** | absent + historical |

Both are **inversions, not misses**. The second reports a patient as asthmatic
from a sentence saying they are not. The first feeds a positive family history
of colon cancer — a genetic-risk referral input — from a sentence denying one.

It survived a 75/75 adversarial set and a 90% held-out set because **neither
contained a sentence where two axes collide.** A gold set only tests the
distinctions its author thought to include.

The cause was representational, not a bad constant. `ASSERTION_PRIORITY` ranked
`family` and `historical` above `absent`, forcing a precedence between things
that are not alternatives. ConText defines three *independent* axes; this had
collapsed them into one label:

```
presence      present | absent | possible              does it exist?
experiencer   patient | family                         whose is it?
temporality   current | historical | hypothetical      when?
```

"No family history of colon cancer" is not absent *or* family. It is absent
**and** family, and any single label loses half of it. Entities now carry all
three, and `assertion` is *derived*, so every existing caller and gold number
still works.

**The axes also dissolve an argument the project had already had with itself.**
`docs/ANNOTATION_GUIDELINE.md` Rule 3 — a negated family history stays `family`
— is documented there as *genuinely arguable*, chosen "so the annotation is
consistent, not because it is provably correct". Rule 3 is a tie-break invented
to manage a collapse that should never have been forced. With the axes carried
separately there is no tie to break, and no consumer has to accept my choice.
(I first wrote the collapse the other way, broke that gold case, and found the
guideline had thought about it harder than I just had.)

### 7. The second bug the first one was hiding

Fixing the priority immediately broke a case that had always passed:

```
"No fever, denies chest pain, history of MI, mother had breast cancer."
   -> absent,  absent,          absent,       absent          (all four)
```

`TERMINATION` holds conjunctions, not commas — correctly, since negation *does*
carry across commas in a list ("no fever, chills, or cough"). So the `absent`
scope ran to the end of the sentence and covered every entity. It had always
done that. The old priority list was silently repairing it, because
`historical` and `family` outranked `absent`. **Two wrongs were cancelling, and
no gold set could tell** — the output was right and the reason was not.

The rule that separates the two cases without special-casing commas: *a scope
ends where the next trigger begins, if a clause separator sits between them.*
No separator means one predicate (`No` + `family history of`); a comma means
different clauses. Getting this wrong in the obvious way — truncating at any
adjacent trigger — reinstated the exact inversion the priority fix had just
removed, which has its own regression test.

### 8. Section context, and how little it turned out to be worth

Scored once on `gold/sections.py`, 25 documents written before running the
code, half of them collision cases:

| set | sentence layer only | + sections | delta |
|---|---|---|---|
| all (25) | 84.0% | **92.0%** | +8.0% |
| core (21) | 85.7% | **95.2%** | +9.5% |
| arguable (4) | 75.0% | 75.0% | +0.0% |

**Cases broken by the section layer: zero.** That is bought entirely by the
precedence rule — *the section supplies a default, an explicit sentence trigger
wins*. A layer that **overrode** instead of defaulting would invert every
negated statement under a `FAMILY HISTORY` header, which is what the gold set
was built to catch.

But the honest headline is smaller than the table looks. **The section layer
fired on 3 entities across all 25 documents**, because the header text *is
itself a ConText trigger*: `PMH:` and `FAMILY HISTORY:` are in `lexicon.py` and
scope forward over their own sentence. The sentence layer gets there first. The
section pass only earns its place across sentence *boundaries* — a finding two
lines below the header, out of the trigger's reach:

```
PAST MEDICAL HISTORY: asthma.
Also carries a diagnosis of GERD.     <- only the section layer reaches this
```

Most of the +9.5% is the axis fix from § 6, not the section layer. Both are
reported rather than letting one take credit for the other.

Four cases are marked `arguable` in the gold file and scored separately,
because the honest thing to do with an annotation I am not confident in is to
say which ones they are. Four annotation errors of my own — two mistyped
concept ids, one term absent from the lexicon, and two cases I annotated
against this project's own guideline without re-reading it — are listed in the
gold file's header rather than quietly edited.

### 9. A service whose log cannot leak

`serve.py` — `POST /extract`, `POST /extract/batch`, `GET /about`, `GET /log`.

**A clinical NLP service is where PHI leaks into logs.** Every other component
in a hospital stack handles structured data, where a careless log line does
bounded damage. This one handles the note — the least structured and most
identifying artefact in the record — and the default behaviour of every web
framework ever written is to log the request body when something goes wrong.
That is how free-text PHI reaches a log aggregator with a different retention
policy, ACL and threat model from the EHR it came from.

`SafeLog` has a **fixed field list**, every entry a number, an enum, or an id
this service generated. There is no field that accepts caller text — so there
is no redaction pass with a false-negative rate to get wrong. Errors record an
exception **class, never a message**, because a `JSONDecodeError` quotes the
characters that broke it and those characters are note text. Even
`BaseHTTPRequestHandler.log_message` is overridden: its default writes the
request *line*, so the framework default is itself the leak.

Tested by planting identifiers in a request, exercising every path including
the error paths, and asserting none of them reach the log:

```
identifiers found in the log: NONE
{"request_id": "13cb4a95…", "route": "/extract", "status": 400,
 "n_bytes": 39, "n_documents": 0, "n_entities": 0,
 "duration_ms": 0.217, "error_class": "JSONDecodeError"}
```

What it does **not** protect: process memory and core dumps, the *response*
(which contains note spans by design), or an operator adding one `print()`.
This is a discipline supported by a data structure, not an enforcement
boundary.

Every response also carries the leakage rate — *22.2% of negations are lost, so
no negative assertion from this service may be treated as a clinical rule-out*
— because a caveat that lives only in a README is one the integrating team
reads once.

### 10. Throughput, and why this project should not quote a docs/sec number

The README's gap list said the old "839 sentences/sec" was "not a pipeline
claim: no batching, no parallelism, no service". All three are now measured,
and the result is mostly negative.

The benchmark re-runs its **first configuration last** and reports the drift:

```
MACHINE DRIFT during this run: 32.6%

comparison                        ratio   verdict
process pool x4 vs single          1.68   real
HTTP batch=100 vs batch=1          0.78   INSIDE THE NOISE -- not a finding
section pass overhead              0.96   INSIDE THE NOISE -- not a finding
```

**A ratio smaller than the run's own drift is not a speedup**, and reporting
one is the same error as reporting a model difference smaller than its
confidence interval — which `ml1-readmission-risk` had to retract for exactly
this reason.

What survives: a 4-way process pool gives **1.68×, not 4×** — extraction costs
14.2 ms per note, so the pool pays for process startup and result pickling
against work that is already small, and a *thread* pool would do worse because
this is pure-Python regex under the GIL. HTTP batching came out **inside the
noise**, contradicting what I expected: at 14 ms of extraction per note there
is little per-request framing cost left to amortise.

**The number this project should not quote is docs/sec.** Identical code
measured **67 and 21 docs/sec** on this machine on different days — a 3× spread
with no code change. The original "839 sentences/sec" has the same standing: it
describes one afternoon on one desktop.

Three earlier versions of this benchmark were wrong in ways worth naming. The
first reported `+ FHIR bundle` as *faster* than the extraction it is a superset
of — an impossible ordering, caused by the first timed pass paying to compile
every trigger regex. An impossible ordering is the cheapest signal a benchmark
is broken; one that is merely implausible gets published.


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

## Document-level reconciliation — and what it refuses to decide

`src/reconcile.py`. The gap list said: *"If a note says 'no chest pain' in the
HPI and 'chest pain' in the assessment, both are annotated as written. Which one
is true is a clinical judgement the annotator is not qualified to make, and a
real pipeline needs an explicit document-level policy."*

**The policy is that most of the time there is no answer, and saying so is the
answer.** A note contradicting itself is not a parsing failure — it is the
ordinary shape of a clinical document. The HPI records what the patient
reported, the exam what was found, the assessment what the clinician concluded,
and those disagree for good reasons.

```
HISTORY OF PRESENT ILLNESS: Patient denies chest pain.
ASSESSMENT AND PLAN: chest pain, likely musculoskeletal.

  chest_pain   presence=None   confidence=unresolved
```

`to_rows()` **omits** unresolved concepts by default. A consumer that wants them
must pass `include_unresolved=True` — which makes accepting a guess explicit at
the call site instead of buried in this module.

**Last-mention-wins is the obvious rule and it inverts on the commonest real
pattern.** `"ASSESSMENT: Pneumonia. Return if cough worsens."` — the last
mention of cough is hypothetical, in a return-precaution sentence, and last-wins
would mark the patient's presenting cough hypothetical. Position is a proxy for
authority and a bad one; the **section** is the signal, which is why
`SECTION_AUTHORITY` is a table a site can disagree with rather than an `if`
chain.

Two rules that are not negotiable:

- **Negation never loses quietly.** When mentions disagree about *presence*
  specifically, the result is flagged even if a section outranks. Being wrong in
  that direction puts a condition on a record that a clinician denied.
- **A different experiencer is a different claim.** "Mother had breast cancer"
  and "patient denies breast cancer" are two facts, not a conflict, and are
  never reconciled against each other.

## The service authenticates

An extraction endpoint takes the **note** as input, so an unauthenticated one is
an open channel for submitting PHI to a service the submitter does not control —
a different and larger problem than an unauthenticated read. The request body is
**not read before refusing**: reading it would pull an unauthenticated caller's
note into this process's memory, which is exactly what the token exists to
prevent. `SafeLog` still holds no note text on the refusal path.

## What is still missing, and why it cannot be closed here

Everything below needs something this environment does not have. The closeable
gaps are closed; these are named with their specific blocker.

- **No pretrained transformer.** ClinicalBERT and friends are not installed and
  there is no network. The bakeoff uses an n-gram logistic regression, which is
  a legitimate pre-transformer baseline and is labelled as one. The OOD gap
  would narrow with real pretraining, and no claim about transformer
  performance is made anywhere.
- **No medspaCy / scispaCy.** Not installed; ConText and the NER are
  hand-rolled. There is no POS tagging and no dependency parse, so medication
  attribute attribution is window-based and **provably wrong on coordinated
  lists** — "metformin 500 mg and lisinopril 10 mg daily" cannot be attributed
  correctly without a parse.
- **No kappa, and it is not fakeable.** Inter-annotator agreement needs a
  second annotator. A second pass by the same person measures their
  consistency, not the guideline's clarity, and reporting it as kappa would be
  inventing a number. Rule 3 in the guideline is where drift would show, and it
  is flagged there as the most likely disagreement.
- **No real terminology.** Hand-built SNOMED/RxNorm subsets. VSAC needs a UMLS
  licence and a network; there is no version pinning and no inactivated-concept
  handling.
- **Section detection cannot be improved with this data.** It is a regex over a
  curated header list, and the honest replacement is a sequence labeller — but
  training one on `src/notes.py` would learn the seven headers the generator
  emits and report near-perfect recall on a problem it had not solved. Real
  notes carry headers this table has never seen, carry none at all, or carry
  template headers stamped in by a system that did not mean them. Recall on any
  real corpus is unknown and probably poor.
- **The section layer's marginal value is small** (3 entities across 25 gold
  documents) because header text doubles as a ConText trigger — see § 8. It
  earns its place only on cross-sentence cases, and that is measured rather
  than argued.
- **Throughput is not measurable on this machine.** Identical code gave 67 and
  21 docs/sec on different days, and the benchmark's own drift check reports
  15–37% within one run. Batching and pooling are implemented and measured; the
  absolute numbers are explicitly not quotable.
- **No severity, laterality, or coreference.** Real risk-adjustment and RWE
  work need several of these; they are scope rather than blocker, and adding
  them without a gold set to score them against would be adding untested
  surface.
- **The service is single-node.** No TLS, no rate limiting, no tenancy, no
  queue. The bearer token authenticates a caller, not a user, and there is no
  wiring to the append-only PHI access audit that `se1-hl7-fhir-interop`
  already has.
- **`SafeLog` is a discipline, not a boundary.** It cannot stop process memory,
  core dumps, or one `print()` added while debugging.

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
| `src/sections.py` | section detection; a default that an explicit trigger overrides |
| `gold/sections.py` | 25 collision-case documents, scored once |
| `run_sections.py` | the section evaluation, and whether the layer helped or hurt |
| `serve.py` | extraction API, batching, `SafeLog`, throughput bench |
| `tests/test_serving.py` | 31 tests: the axes, sections, and the log that cannot leak |
| `src/reconcile.py` | document-level policy; what it refuses to decide |
| `tests/test_reconcile.py` | 17 tests: conflicts, section authority, auth |
| `tests/test_nlp.py` | 29 tests |
