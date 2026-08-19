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
ASSERTION_PRIORITY = ["present", "possible", "absent", "hypothetical",
                      "historical", "family"]


class Entity:
    __slots__ = ("text", "start", "end", "kind", "concept", "assertion",
                 "attrs", "code_system", "code", "code_display")

    def __init__(self, text, start, end, kind, concept):
        self.text, self.start, self.end = text, start, end
        self.kind, self.concept = kind, concept
        self.assertion = "present"
        self.attrs = {}
        self.code_system = self.code = self.code_display = None

    def __repr__(self):
        return (f"Entity({self.text!r}, {self.kind}, {self.concept}, "
                f"{self.assertion})")

    def as_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end,
                "kind": self.kind, "concept": self.concept,
                "assertion": self.assertion, "attrs": dict(self.attrs),
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
            scopes.append(("absent", m.end(), terminations_after(m.end())))
            for i in range(m.start(), m.end()):
                consumed[i] = ""
    other = "".join(consumed)

    for table, assertion in [(FAMILY, "family"), (HISTORICAL, "historical"),
                             (HYPOTHETICAL, "hypothetical"), (POSSIBLE, "possible")]:
        for trig in table:
            for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                                 other):
                scopes.append((assertion, m.end(), terminations_after(m.end())))
    for table, assertion in [(HISTORICAL_BACKWARD, "historical"),
                             (HYPOTHETICAL_BACKWARD, "hypothetical")]:
        for trig in table:
            for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                                 other):
                scopes.append((assertion, terminations_before(m.start()), m.start()))
    # A few uncertainty markers scope BACKWARD as well as forward: in
    # "pneumonia versus atelectasis" the doubt attaches to both alternatives.
    for trig in ["versus", "vs", "cannot be excluded", "cannot be ruled out",
                 "is unlikely", "cannot exclude"]:
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])",
                             sent_lower):
            scopes.append(("possible", terminations_before(m.start()), m.start()))
    for trig in NEGATION_BACKWARD:
        for m in re.finditer(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])", masked):
            scopes.append(("absent", terminations_before(m.start()), m.start()))
    return scopes


def _apply_assertion(ent, scopes, offset):
    s, e = ent.start - offset, ent.end - offset
    best = "present"
    for assertion, a, b in scopes:
        if a <= s and e <= b:
            if (ASSERTION_PRIORITY.index(assertion)
                    > ASSERTION_PRIORITY.index(best)):
                best = assertion
    ent.assertion = best


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
