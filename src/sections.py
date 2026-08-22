"""Section context: the gap the README named, and the reason it is subtle.

THE CLAIM THIS FILE TESTS
-------------------------
The README said: "A finding under a `FAMILY HISTORY:` header should arguably
default to `family` regardless of phrasing." Note the *arguably*. That hedge is
the whole design problem, and it is worth stating before any code.

A section header is EVIDENCE about assertion, not a DECISION about it. Consider:

    FAMILY HISTORY: Mother with breast cancer. Patient had a colonoscopy in
    2019 and has no history of colon cancer herself.

Both sentences sit under `FAMILY HISTORY`. The first is family history. The
second is about the patient, is negated, and a header that overrode the
sentence would turn "the patient does NOT have colon cancer" into "the family
HAS colon cancer" -- an inversion, in the section where inversions are most
dangerous, because family history drives genetic-risk referrals.

So the rule implemented here is:

    THE SECTION SUPPLIES A DEFAULT. AN EXPLICIT SENTENCE-LEVEL TRIGGER WINS.

If ConText fired anything at all -- negation, uncertainty, historical,
hypothetical, family -- that result stands. The section default applies only to
entities that came out of the sentence layer as bare `present`, which is
exactly the case the header is informative about: a telegraphic list under a
header, with no trigger words in it at all.

    PMH: asthma, GERD, obesity.          -> historical, from the header
    FAMILY HISTORY: stroke, diabetes.    -> family, from the header
    FAMILY HISTORY: no colon cancer.     -> absent, from the SENTENCE

WHY THIS IS NOT JUST A DICTIONARY OF HEADERS
--------------------------------------------
Three things make it more than that, and each is a place real notes break:

1. HEADERS ARE NOT UNIFORM. `PMH`, `PAST MEDICAL HISTORY`, `Past Medical Hx`,
   and `HISTORY:` all mean the same thing and none of them is canonical. The
   normalisation table here is a curated subset, not a standard, and it is
   nothing like the coverage of a real note corpus.

2. SECTIONS RUN UNTIL THE NEXT HEADER, not until the next blank line. An entity
   three paragraphs below a header is still in that section, which means the
   parser has to carry state across sentence boundaries -- the reason this is a
   document-level pass rather than something `extract()` could do per sentence.

3. NOT EVERY SECTION IMPLIES AN ASSERTION. `ASSESSMENT AND PLAN` and
   `MEDICATIONS` carry no default at all, and giving them one would be worse
   than having no section layer: an assessment is the most `present` thing in a
   note, and inventing a default there could only ever demote it.

WHAT THIS IS NOT
----------------
Not a section classifier. Real notes carry headers this table has never seen,
carry no headers at all, or carry headers inside a template that a downstream
system stamped in without meaning them. Detecting sections in the wild is a
sequence-labelling problem; this is a regex over a curated list, and its recall
on any real corpus is unknown and probably poor.
"""

from __future__ import annotations

import re

# Canonical section -> the assertion it makes the DEFAULT for bare entities.
# None means "this section is recognised but implies nothing", which is a
# deliberate entry rather than an omission -- see point 3 above.
SECTION_DEFAULT = {
    "family_history": "family",
    "past_medical_history": "historical",
    "past_surgical_history": "historical",
    "social_history": None,
    "chief_complaint": None,
    "history_of_present_illness": None,
    "review_of_systems": None,
    "allergies": None,
    "medications": None,
    "physical_exam": None,
    "assessment_and_plan": None,
    "labs": None,
}

# Surface forms -> canonical name. Longest-first matching at use site, because
# "PAST MEDICAL HISTORY" contains "HISTORY" and the short form must not win.
SECTION_ALIASES = {
    "family history": "family_history",
    "fh": "family_history",
    "family hx": "family_history",
    "past medical history": "past_medical_history",
    "pmh": "past_medical_history",
    "past medical hx": "past_medical_history",
    "medical history": "past_medical_history",
    "past surgical history": "past_surgical_history",
    "psh": "past_surgical_history",
    "surgical history": "past_surgical_history",
    "social history": "social_history",
    "sh": "social_history",
    "chief complaint": "chief_complaint",
    "cc": "chief_complaint",
    "history of present illness": "history_of_present_illness",
    "hpi": "history_of_present_illness",
    "review of systems": "review_of_systems",
    "ros": "review_of_systems",
    "allergies": "allergies",
    "allergy": "allergies",
    "medications": "medications",
    "meds": "medications",
    "current medications": "medications",
    "home medications": "medications",
    "physical exam": "physical_exam",
    "physical examination": "physical_exam",
    "pe": "physical_exam",
    "exam": "physical_exam",
    "assessment and plan": "assessment_and_plan",
    "assessment": "assessment_and_plan",
    "impression": "assessment_and_plan",
    "plan": "assessment_and_plan",
    "a/p": "assessment_and_plan",
    "labs": "labs",
    "laboratory": "labs",
    "laboratory data": "labs",
}

_ALIASES_LONGEST_FIRST = sorted(SECTION_ALIASES, key=len, reverse=True)

# A header is a known label followed by a colon, at a line start or after a
# sentence end. The colon is required: "family history of stroke" mid-sentence
# is a ConText trigger, not a section header, and treating it as a header would
# put the REST OF THE NOTE into family context.
_HEADER = re.compile(
    r"(?:(?<=^)|(?<=\n))[ \t]*(" + "|".join(
        re.escape(a) for a in _ALIASES_LONGEST_FIRST) + r")[ \t]*:",
    re.IGNORECASE)


def find_sections(text):
    """[(canonical_name, body_start, body_end)] covering the whole document.

    Sections run until the NEXT header, not until the next blank line. Text
    before the first header is reported as `preamble`, which has no default --
    an unlabelled opening paragraph is not evidence of anything.
    """
    spans = []
    matches = list(_HEADER.finditer(text))
    if not matches:
        return [("preamble", 0, len(text))]
    if matches[0].start() > 0:
        spans.append(("preamble", 0, matches[0].start()))
    for i, m in enumerate(matches):
        name = SECTION_ALIASES[m.group(1).strip().lower()]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((name, m.end(), end))
    return spans


def section_at(sections, offset):
    for name, a, b in sections:
        if a <= offset < b:
            return name
    return "preamble"


def apply_section_context(entities, text, *, kinds=("condition",)):
    """Fill in bare `present` assertions from the enclosing section.

    Returns the entities, mutated, plus a list of what changed -- because a
    layer that silently rewrites another layer's output is unauditable, and
    "which of these came from the header rather than the words" is the first
    question anyone reviewing an extraction will ask.

    CONDITIONS ONLY, by default. A medication under `PAST MEDICAL HISTORY` is
    not a historical medication -- notes list active home medications under
    history headers constantly -- and an allergy is an allergy wherever it is
    written. Widening `kinds` is possible and is not recommended.
    """
    sections = find_sections(text)
    changes = []
    for e in entities:
        if e.kind not in kinds:
            continue
        # THE PRECEDENCE RULE. Anything the sentence layer decided stands.
        if e.assertion != "present":
            continue
        name = section_at(sections, e.start)
        default = SECTION_DEFAULT.get(name)
        if not default:
            continue
        changes.append({"text": e.text, "start": e.start, "concept": e.concept,
                        "from": e.assertion, "to": default, "section": name})
        e.assertion = default
    return entities, changes


def extract_with_sections(text, extract_fn):
    """`extract()` plus the section pass, as one call.

    Kept separate from `extract()` rather than folded into it. The sentence
    layer is testable in isolation and its gold sets are sentence-level; a
    document-level default that silently changed those results would make every
    existing number in this project incomparable to the next one.
    """
    ents = extract_fn(text)
    ents, changes = apply_section_context(ents, text)
    return ents, changes
