"""Entity extraction plus a ConText-style assertion layer.

medspaCy and scispaCy are not installed offline, so both the dictionary NER and
the ConText algorithm are implemented here. That is the right trade for this
project anyway: the spec calls the rule layer "interview gold", and the rule
layer is precisely the part you cannot demonstrate understanding of by importing
it.

THE ALGORITHM
-------------
ConText (Chapman et al.) generalises NegEx to several axes. For each sentence:

  1. find PSEUDO-triggers first and mask them, because "no increase in pain" and
     "cannot be ruled out" contain negation phrases while asserting the
     opposite. Skipping this step is the single most common way a negation
     detector goes quietly wrong;
  2. find real triggers and open a scope, forward or backward;
  3. propagate each scope until a TERMINATION term ("but", "however",
     "however") or the sentence boundary;
  4. any entity inside a scope takes that scope's assertion value.

Precedence when scopes overlap is explicit and ordered (see ASSERTION_PRIORITY):
family beats historical beats hypothetical beats absent beats possible beats
present. The ordering is a clinical judgement, not an implementation detail --
"no family history of breast cancer" must not become a patient condition, and
"mother had breast cancer" must not either, for different reasons.
"""

from __future__ import annotations

import re

from lexicon import (ALLERGEN_TERMS, CONDITION_TERMS, FAMILY, HISTORICAL,
                     HISTORICAL_BACKWARD, HYPOTHETICAL, HYPOTHETICAL_BACKWARD,
                     MEDICATION_TERMS, NEGATION_BACKWARD, NEGATION_FORWARD,
                     POSSIBLE, PSEUDO_NEGATION, RXNORM, SNOMED, TERMINATION)

# Later entries win when two scopes cover the same entity.
#
# THIS ORDERING WAS A BUG, AND IT WAS THE WORST KIND. `family` and `historical`
# outranked `absent`, so any negated statement that also carried a history or
# family trigger came out POSITIVE:
#
#     "FAMILY HISTORY: No known history of colon cancer."  ->  family
#     "PMH: Denies asthma."                                ->  historical
#
# That is negation loss -- an inversion, not a miss -- and in the family-history
# section it is the input to a genetic-risk referral for a disease nobody has.
# It survived a 75/75 adversarial set and a 90% held-out set because neither
# contained a sentence where the two axes collide. gold/sections.py was written
# around those collisions and found it immediately.
#
# The root cause is representational, not a bad constant. ConText defines three
# INDEPENDENT axes and this collapsed them into one label, which forces an
# arbitrary precedence between things that are not alternatives:
#
#     presence      present | absent | possible               does it exist?
#     experiencer   patient | family                          whose is it?
#     temporality   current | historical | hypothetical       when?
#
# "No family history of colon cancer" is not absent OR family. It is absent AND
# family, and any single label loses half of it. Entities now carry all three,
# and `assertion` is DERIVED from them (see collapse_assertion) so every
# existing caller and gold set keeps working.
ASSERTION_PRIORITY = ["present", "possible", "absent", "hypothetical",
                      "historical", "family"]

PRESENCE_LABELS = ("possible", "absent")
TEMPORALITY_LABELS = ("hypothetical", "historical")


def collapse_assertion(presence, experiencer, temporality):
    """One label from three axes, for callers that want a single string.

    THE ORDER IS experiencer > presence > temporality, and the first two are
    not in the order I first wrote them.

    My initial collapse put presence first, on the reasoning that negation is
    the clinically actionable fact, so "no family history of breast cancer"
    should be `absent`. That broke a gold case, and the gold case turned out to
    be right: docs/ANNOTATION_GUIDELINE.md Rule 3 says a negated family history
    stays `family`, and the guideline had already thought about it harder than
    I just had. Its reasoning:

        the negation scopes over THE FAMILY HISTORY, which is a statement about
        relatives, so the finding is still not about the patient ... the
        downstream consumer must never see a patient Condition either way, and
        both readings agree on that. Rule 3 exists so the annotation is
        CONSISTENT, not because it is provably correct.

    That last sentence is the important one, and it is what the three-axis
    representation is for. Rule 3 is a tie-break invented to manage a
    collapse that should never have been forced. With the axes carried
    separately there is no tie to break:

        "No family history of breast cancer"
            presence=absent  experiencer=family  temporality=historical

    Both readings are in there, and no consumer has to accept my choice between
    them. The collapse honours Rule 3 so that every existing gold set, and the
    once-scored held-out number, remain comparable -- but fhir_out and any
    other consumer should read the axes.

    TEMPORALITY IS LAST, and that part of the original reasoning stands.
    Nothing in the guideline protects a negated historical statement, and
    "PMH: Denies asthma" collapsing to `historical` was straightforward
    negation loss about the patient's own condition. Presence beats temporality.
    """
    if experiencer == "family":
        return "family"                    # ANNOTATION_GUIDELINE.md Rule 3
    if presence in ("absent", "possible"):
        return presence
    if temporality in ("historical", "hypothetical"):
        return temporality
    return "present"


class Entity:
    __slots__ = ("text", "start", "end", "kind", "concept", "assertion",
                 "attrs", "code_system", "code", "code_display",
                 "presence", "experiencer", "temporality", "assertion_source")

    def __init__(self, text, start, end, kind, concept):
        self.text, self.start, self.end = text, start, end
        self.kind, self.concept = kind, concept
        self.assertion = "present"
        self.presence = "present"
        self.experiencer = "patient"
        self.temporality = "current"
        self.assertion_source = "default"
        self.attrs = {}
        self.code_system = self.code = self.code_display = None

    def __repr__(self):
        return (f"Entity({self.text!r}, {self.kind}, {self.concept}, "
                f"{self.assertion})")

    def as_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end,
                "kind": self.kind, "concept": self.concept,
                "assertion": self.assertion,
                "presence": self.presence, "experiencer": self.experiencer,
                "temporality": self.temporality,
                "assertion_source": self.assertion_source,
                "attrs": dict(self.attrs),
                "code_system": self.code_system, "code": self.code,
                "code_display": self.code_display}


# ---------------------------------------------------------------------------
def split_sentences(text):
    """Sentence boundaries, clinical-note flavoured.

    Notes are not prose. Newlines and section headers are boundaries, and
    abbreviations with periods are not. A general-purpose splitter that breaks
    on 'q.d.' or 'Dr.' will scatter assertion scopes across fragments, which is
    a silent accuracy loss nobody attributes to the splitter.
    """
    protected = text
    for abbr in ["q.d.", "b.i.d.", "t.i.d.", "q.i.d.", "p.o.", "p.r.n.",
                 "Dr.", "Mr.", "Mrs.", "Ms.", "a.m.", "p.m.", "y.o.", "i.e.",
                 "e.g.", "q.h.s.", "s.p."]:
        protected = protected.replace(abbr, abbr.replace(".", "\x00"))
    parts, spans, pos = [], [], 0
    for chunk in re.split(r"(?<=[.;!?])\s+|\n+", protected):
        if not chunk.strip():
            pos = protected.find(chunk, pos) + len(chunk) if chunk else pos
            continue
        start = protected.find(chunk, pos)
        if start < 0:
            start = pos
        parts.append(chunk.replace("\x00", "."))
        spans.append((start, start + len(chunk)))
        pos = start + len(chunk)
    return list(zip(parts, spans))


def _find_terms(sent_lower, table):
    """Longest-match-first dictionary lookup returning (start, end, concept).

    Longest-first matters clinically: 'type 2 diabetes' must not be matched as
    'diabetes', because the specific concept and the unspecified one are
    different codes with different downstream meaning (risk adjustment pays
    differently for them, for one thing).
    """
    hits = []
    for surface in sorted(table, key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])",
                             sent_lower):
            if any(m.start() < h[1] and h[0] < m.end() for h in hits):
                continue     # already covered by a longer match
            hits.append((m.start(), m.end(), table[surface]))
    return sorted(hits)


DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?|ml|iu)\b", re.I)
ROUTE = re.compile(r"\b(po|by mouth|orally|iv|intravenous(?:ly)?|im|sublingual|"
                   r"sl|subcutaneous(?:ly)?|subq|sc|inhaled|nebulized|topical|"
                   r"pr|rectally|transdermal)\b", re.I)
FREQ = re.compile(r"\b(qd|q\.?d\.?|daily|once daily|bid|b\.?i\.?d\.?|"
                  r"twice daily|tid|t\.?i\.?d\.?|three times daily|qid|"
                  r"q\.?i\.?d\.?|four times daily|qhs|nightly|at bedtime|"
                  r"weekly|every other day|q\d+h|every \d+ hours|prn|"
                  r"as needed)\b", re.I)


def _med_attrs(sent, start, end):
    """Dose/route/frequency from the window after the drug name.

    Window-based attribution is a real limitation and is stated rather than
    hidden: in 'metformin 500 mg and lisinopril 10 mg daily' the frequency
    attaches to whichever drug's window reaches it first. Correct attribution
    needs a parse, or at minimum a list-aware segmenter.
    """
    window = sent[end:end + 60]
    stop = re.search(r"\b(and|,|;)\s+(?=[a-z])", window)
    attrs = {}
    d = DOSE.search(window)
    if d:
        attrs["dose"] = f"{d.group(1)} {d.group(2).lower()}"
    r = ROUTE.search(window)
    if r:
        attrs["route"] = r.group(1).lower()
    f = FREQ.search(window)
    if f:
        attrs["frequency"] = f.group(1).lower()
    if stop and d and d.start() > stop.start():
        attrs.pop("dose", None)
    return attrs


# ---------------------------------------------------------------------------
def _scopes(sent_lower):
    """Return [(assertion, start, end)] scopes opened by triggers."""
    masked = sent_lower
    for pseudo in PSEUDO_NEGATION:
        for m in re.finditer(re.escape(pseudo), masked):
            masked = masked[:m.start()] + ("\x01" * len(pseudo)) + masked[m.end():]

    def terminations_after(pos):
        best = len(masked)
        for t in TERMINATION:
            m = re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", masked[pos:])
            if m:
                best = min(best, pos + m.start())
        return best

    def terminations_before(pos):
        best = 0
        for t in TERMINATION:
            for m in re.finditer(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])",
                                 masked[:pos]):
                best = max(best, m.end())
        return best

    scopes = []
    # Negation reads the pseudo-masked text. Longest triggers first, and each
    # match is then hidden from the other axes: "no history of diabetes" must
    # resolve to ABSENT, but it literally contains the historical trigger
    # "history of", which outranks absent in ASSERTION_PRIORITY and would win.
    # Consuming the span is what makes trigger precedence work.
    # Built from the RAW sentence, not the pseudo-masked one: pseudo-negation
    # masking exists to stop a phrase NEGATING, not to hide it from every other
    # axis. "cannot exclude" must be blocked from negating and must still fire
    # uncertainty -- it is an expression of doubt, which is the whole point.
    consumed = list(sent_lower)
    for trig in sorted(NEGATION_FORWARD, key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])", masked):
            scopes.append(("absent", m.end(), terminations_after(m.end()), "fwd"))
            for i in range(m.start(), m.end()):
                consumed[i] = ""
    other = "".join(consumed)

    for table, assertion in [(FAMILY, "family"), (HISTORICAL, "historical"),
                             (HYPOTHETICAL, "hypothetical"), (POSSIBLE, "possible")]:
        for trig in table:
            for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                                 other):
                scopes.append((assertion, m.end(), terminations_after(m.end()), "fwd"))
    for table, assertion in [(HISTORICAL_BACKWARD, "historical"),
                             (HYPOTHETICAL_BACKWARD, "hypothetical")]:
        for trig in table:
            for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                                 other):
                scopes.append((assertion, terminations_before(m.start()), m.start(), "bwd"))
    # A few uncertainty markers scope BACKWARD as well as forward: in
    # "pneumonia versus atelectasis" the doubt attaches to both alternatives.
    for trig in ["versus", "vs", "cannot be excluded", "cannot be ruled out",
                 "is unlikely", "cannot exclude"]:
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                             sent_lower):
            scopes.append(("possible", terminations_before(m.start()), m.start(), "bwd"))
    for trig in NEGATION_BACKWARD:
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])", masked):
            scopes.append(("absent", terminations_before(m.start()), m.start(), "bwd"))

    return _bound_by_other_triggers(scopes, sent_lower)


def _all_trigger_spans(sent_lower):
    """Every (start, end) where any ConText trigger of any axis matches."""
    spans = []
    tables = (list(NEGATION_FORWARD) + list(NEGATION_BACKWARD) + list(FAMILY)
              + list(HISTORICAL) + list(HISTORICAL_BACKWARD)
              + list(HYPOTHETICAL) + list(HYPOTHETICAL_BACKWARD)
              + list(POSSIBLE))
    for trig in tables:
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                             sent_lower):
            spans.append((m.start(), m.end()))
    return sorted(set(spans))


def _bound_by_other_triggers(scopes, sent_lower):
    """A scope ends where the NEXT trigger begins (and starts where the previous one ends).

    THE SECOND BUG, WHICH THE FIRST ONE WAS HIDING.

    Telegraphic notes chain clauses with commas:

        "No fever, denies chest pain, history of MI, mother had breast cancer."

    TERMINATION holds conjunctions and a few clause words, not commas -- and
    correctly so, because negation genuinely does carry across commas in a
    list: "no fever, chills, or cough" negates all three. So the `absent` scope
    opened by "no" ran to the end of the sentence and covered every entity in
    it, including the MI and the breast cancer.

    That produced the right answer anyway, for the wrong reason, as long as
    `historical` and `family` outranked `absent` in the old priority list. The
    moment presence was given precedence -- which is correct -- this sentence
    collapsed to four `absent` entities and the hidden bug surfaced. Two wrongs
    had been cancelling, and no gold set could tell.

    The rule that separates the two cases without special-casing commas:
    A SCOPE ENDS WHERE THE NEXT TRIGGER BEGINS. "no fever, chills, cough" has
    no second trigger, so the negation carries across the whole list. "No
    fever, denies chest pain" has one at "denies", so the first scope stops
    there and the second takes over.

    DIRECTION MATTERS AND IS TRACKED EXPLICITLY. A forward scope is bounded on
    its RIGHT edge only; a backward scope on its LEFT. The first version of
    this function moved both edges of every scope, which pushed entities
    outside their own trigger's scope and turned four correct labels into four
    `present` ones -- caught immediately, because the fix made the test output
    worse rather than better.

    This is not what the ConText paper specifies -- it uses termination terms
    only. It is a strictly stronger condition on the same idea, and it exists
    because a termination list cannot express "a comma, but only when a new
    predicate follows".
    """
    triggers = _all_trigger_spans(sent_lower)
    if len(triggers) < 2:
        return [(a, x, y) for a, x, y, _d in scopes]

    starts = [ts for ts, _te in triggers]
    ends = [te for _ts, te in triggers]

    bounded = []
    for assertion, a, b, direction in scopes:
        if direction == "fwd":
            # `a` is this trigger's own end; stop at the next trigger to start
            for ts in starts:
                if ts >= a and _separates(sent_lower, a, ts):
                    b = min(b, ts)
                    break
        else:
            # `b` is this trigger's own start; start after the previous trigger
            for te in reversed(ends):
                if te <= b and _separates(sent_lower, te, b):
                    a = max(a, te)
                    break
        if a < b:
            bounded.append((assertion, a, b))
    return bounded


def _separates(sent_lower, a, b):
    """Is there a clause boundary between two positions?

    THE CONDITION THE WHOLE RULE TURNS ON, and it took three attempts to state.

    A following trigger only ends the previous scope if a clause separator sits
    between them. Without this test, adjacent triggers truncate each other and
    negation is lost in the most common construction there is:

        "No family history of breast cancer."
         ^^                                     negation trigger
            ^^^^^^^^^^^^^^                      family trigger, ADJACENT

    "no" and "family history of" are consecutive words of ONE predicate, and
    truncating the negation at the family trigger produced `family` with
    presence=present -- a positive family history of breast cancer, from a
    sentence denying one. The same inversion the priority fix had just removed,
    reintroduced by its own fix.

    Contrast the case the rule exists for:

        "No fever, denies chest pain, history of MI"
         ^^        ^^^^^^                              separated by a comma

    Here the triggers belong to different clauses and each should own its own.

    So: a comma or semicolon between them means different clauses; nothing
    between them means one phrase. This is exactly the "a comma, but only when
    a new predicate follows" condition the termination list cannot express.
    """
    return any(ch in sent_lower[a:b] for ch in ",;")


def _apply_assertion(ent, scopes, offset):
    """Resolve all three axes independently, then derive the single label.

    Independently is the point. The previous version took one argmax over a
    priority list, so a sentence opening both an `absent` scope and a `family`
    scope produced one label and silently discarded the other. Each axis now
    resolves against only the scopes belonging to it, so a statement can be
    absent AND family -- which is exactly what "no family history of X" is.

    WITHIN an axis a precedence is still needed, and those are kept: absent
    beats possible (an explicit denial is stronger than a hedge), and
    historical beats hypothetical.
    """
    s, e = ent.start - offset, ent.end - offset
    covering = [a for a, x, y in scopes if x <= s and e <= y]

    presence = "present"
    for label in PRESENCE_LABELS:                 # later wins
        if label in covering:
            presence = label
    experiencer = "family" if "family" in covering else "patient"
    temporality = "current"
    for label in TEMPORALITY_LABELS:              # later wins
        if label in covering:
            temporality = label

    ent.presence, ent.experiencer, ent.temporality = (presence, experiencer,
                                                      temporality)
    ent.assertion = collapse_assertion(presence, experiencer, temporality)
    ent.assertion_source = "sentence" if covering else "default"


def map_terminology(ent):
    if ent.kind == "condition":
        hit = SNOMED.get(ent.concept)
        if hit:
            ent.code_system, ent.code, ent.code_display = "SNOMED-CT", hit[0], hit[1]
    elif ent.kind == "medication":
        hit = RXNORM.get(ent.concept)
        if hit:
            ent.code_system, ent.code, ent.code_display = "RxNorm", hit[0], hit[1]
    return ent


def extract(text):
    """Full pipeline: sentences -> entities -> assertions -> terminology."""
    out = []
    for sent, (s0, _s1) in split_sentences(text):
        low = sent.lower()
        scopes = _scopes(low)
        found = []
        for a, b, concept in _find_terms(low, CONDITION_TERMS):
            found.append(Entity(sent[a:b], s0 + a, s0 + b, "condition", concept))
        for a, b, concept in _find_terms(low, MEDICATION_TERMS):
            e = Entity(sent[a:b], s0 + a, s0 + b, "medication", concept)
            e.attrs = _med_attrs(sent, a, b)
            found.append(e)
        # allergies only inside an allergy context, or "allergic to X"
        allergy_ctx = bool(re.search(r"allerg|adverse reaction|intoleran", low))
        if allergy_ctx:
            for a, b, concept in _find_terms(low, ALLERGEN_TERMS):
                found.append(Entity(sent[a:b], s0 + a, s0 + b, "allergy", concept))
        for e in found:
            _apply_assertion(e, scopes, s0)
            map_terminology(e)
        # An allergy statement is a positive assertion ABOUT an allergy even
        # though "allergic to penicillin" contains no assertion trigger; but
        # "no known drug allergies" must still negate.
        out.extend(found)
    return sorted(out, key=lambda e: e.start)
