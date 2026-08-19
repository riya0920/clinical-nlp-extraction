"""A supervised assertion classifier, to compare against the rule layer.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The spec asks for a fine-tuned transformer compared against the ConText rules.
No pretrained clinical transformer is available offline -- no HuggingFace, no
BERT weights, and a transformer trained from scratch on a few thousand
templated sentences would be a strawman rather than a baseline.

So this is a **supervised statistical classifier**: character and word n-grams
over the left and right context of the entity, fed to multinomial logistic
regression. That is a real, standard, and genuinely competitive baseline for
assertion classification -- pre-transformer clinical NLP ran on exactly this --
and it is labelled as what it is everywhere it appears. It is NOT ClinicalBERT
and no claim about transformer performance is made or implied.

The comparison it enables is still the one the interview question turns on:
does a learned model beat hand-written rules, and **where does each win?**

THE EXPERIMENT DESIGN, AND WHY IT IS THE INTERESTING PART
---------------------------------------------------------
The classifier trains on sentences from `notes.py`'s templates. The rule layer
was written against a lexicon. Both are then evaluated on two different things:

  IN-DISTRIBUTION   held-out generated sentences -- same templates, unseen draws
  OUT-OF-DISTRIBUTION  the hand-written gold sets, whose phrasings the
                       generator never produces

That second row is the whole point, and it is not a contrived handicap: it is
the actual deployment condition. A model trained on one institution's notes
meets another institution's phrasing on day one, and the question a
risk-adjustment team needs answered is which approach degrades more gracefully.

The metric that decides it is not accuracy. It is **precision on `absent`** --
how often the system says a negated finding is present -- because in risk
adjustment a false `present` becomes a submitted diagnosis and an unsupported
submitted diagnosis is a false-claims exposure. That is a legal problem, not a
metrics problem, and it is not interchangeable with a missed condition.
"""

from __future__ import annotations

import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from lexicon import CONDITION_TERMS

# Trigger phrasings the GENERATOR uses. Deliberately a subset of what the rule
# lexicon knows, and deliberately disjoint from several phrasings used in the
# hand-written held-out set -- so "unseen phrasing" is a real condition rather
# than an assumed one.
TRAIN_TRIGGERS = {
    "absent": ["no", "denies", "no evidence of", "negative for", "without",
               "no signs of", "patient denies"],
    "family": ["family history of", "mother had", "father with",
               "brother with", "FH:", "sister has"],
    "historical": ["history of", "h/o", "PMH:", "status post", "s/p",
                   "past medical history of"],
    "hypothetical": ["return if", "call the clinic if", "come back if",
                     "instructed to return if"],
    "possible": ["possible", "probable", "concerning for", "suspicious for",
                 "likely", "cannot be ruled out"],
    "present": ["", "the patient has", "admitted with", "known",
                "presents with", "diagnosed with"],
}

CARRIERS = [
    "{trigger} {cond}.",
    "{trigger} {cond} on examination.",
    "Patient with {trigger} {cond}.",
    "Assessment: {trigger} {cond}.",
    "{trigger} {cond}; plan discussed.",
    "Noted {trigger} {cond} this admission.",
]

# surface forms to draw entities from
SURFACES = sorted(CONDITION_TERMS)


def make_corpus(n=4000, seed=5, triggers=None):
    """Generate (sentence, start, end, concept, assertion) with exact spans.

    Spans are built by construction rather than by searching for the surface
    form afterwards, because several surface forms occur inside others
    ("diabetes" inside "type 2 diabetes") and a search would mis-locate them.
    """
    rng = random.Random(seed)
    triggers = triggers or TRAIN_TRIGGERS
    rows = []
    for _ in range(n):
        assertion = rng.choice(list(triggers))
        trig = rng.choice(triggers[assertion])
        surface = rng.choice(SURFACES)
        carrier = rng.choice(CARRIERS)

        left, _sep, right = carrier.partition("{cond}")
        left = left.format(trigger=trig).replace("  ", " ")
        text = (left + surface + right).replace("  ", " ").strip()
        start = text.index(surface, max(0, len(left) - 2))
        rows.append({"text": text, "start": start, "end": start + len(surface),
                     "concept": CONDITION_TERMS[surface],
                     "assertion": assertion})
    return rows


# ---------------------------------------------------------------------------
def _left_context(row, width=60):
    return row["text"][max(0, row["start"] - width):row["start"]].lower()


def _right_context(row, width=40):
    return row["text"][row["end"]:row["end"] + width].lower()


class ContextExtractor:
    """Turns a row into the two strings the vectorisers see.

    Left and right context are vectorised SEPARATELY. Direction carries the
    meaning in assertion classification: "no evidence of pneumonia" and
    "pneumonia, no evidence of recurrence" share a bag of words and mean
    different things. A single bag over the whole sentence throws that away,
    which is the most common way a learned assertion classifier underperforms
    rules for no good reason.
    """

    def __init__(self, side):
        self.side = side

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fn = _left_context if self.side == "left" else _right_context
        return [fn(r) for r in X]

    def get_params(self, deep=True):
        return {"side": self.side}

    def set_params(self, **kw):
        self.side = kw.get("side", self.side)
        return self


def build_classifier(C=4.0, seed=0):
    left = Pipeline([("ctx", ContextExtractor("left")),
                     ("tf", TfidfVectorizer(ngram_range=(1, 3), min_df=2,
                                            sublinear_tf=True))])
    right = Pipeline([("ctx", ContextExtractor("right")),
                      ("tf", TfidfVectorizer(ngram_range=(1, 3), min_df=2,
                                             sublinear_tf=True))])
    return Pipeline([
        ("feats", FeatureUnion([("left", left), ("right", right)])),
        ("clf", LogisticRegression(max_iter=3000, C=C, random_state=seed)),
    ])


def train(rows, C=4.0, seed=0):
    clf = build_classifier(C, seed)
    clf.fit(rows, [r["assertion"] for r in rows])
    return clf


def predict(clf, rows):
    return list(clf.predict(rows))


# ---------------------------------------------------------------------------
def rows_from_gold(cases, extractor):
    """Turn a hand-written gold case list into classifier rows.

    The ENTITY SPANS come from the rule extractor, so both systems are scored
    on exactly the same entities and the comparison isolates the assertion
    decision. Otherwise a difference in what was found would contaminate a
    measurement of how it was labelled.
    """
    rows, truth = [], []
    for text, expected in cases:
        found = extractor(text)
        by_concept = {}
        for concept, kind, assertion in expected:
            if kind == "condition":
                by_concept.setdefault(concept, []).append(assertion)
        for e in found:
            if e.kind != "condition":
                continue
            want = by_concept.get(e.concept)
            if not want:
                continue
            rows.append({"text": text, "start": e.start, "end": e.end,
                         "concept": e.concept, "assertion": want[0]})
            truth.append(want[0])
    return rows, truth
