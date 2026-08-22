"""Score the section layer against its held-out set. Once.

Reports three things separately, because collapsing them would hide the only
interesting question:

  ALL          every case
  CORE         the cases I am confident about
  ARGUABLE     the four I am not, marked in gold/sections.py

and, separately, whether the section layer HELPED or HURT relative to running
the sentence layer alone. That comparison is the point. A section layer that
raises accuracy on telegraphic lists while inverting negated statements is a
net loss no matter what its headline number says.

Run:  python run_sections.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sections as S
from extract import extract
from gold.sections import CASES


def _score(doc, expected, with_sections):
    """Compare (concept, assertion) multisets, order-insensitive within a doc."""
    ents = extract(doc)
    changes = []
    if with_sections:
        ents, changes = S.apply_section_context(ents, doc)
    got = sorted((e.concept, e.assertion) for e in ents if e.kind == "condition")
    want = sorted(expected)
    return got, want, got == want, changes


def main():
    rows = []
    for doc, expected, arguable in CASES:
        got_w, want, ok_w, changes = _score(doc, expected, True)
        got_n, _w, ok_n, _c = _score(doc, expected, False)
        rows.append({"doc": doc, "expected": want, "arguable": arguable,
                     "with_sections": got_w, "without_sections": got_n,
                     "pass_with": ok_w, "pass_without": ok_n,
                     "changes": changes})

    core = [r for r in rows if not r["arguable"]]
    arg = [r for r in rows if r["arguable"]]

    def pct(rs, key):
        return (sum(r[key] for r in rs) / len(rs)) if rs else float("nan")

    print("=" * 78)
    print("SECTION CONTEXT -- held-out set, scored once")
    print("=" * 78)
    print(f"  {'set':<12}{'n':>5}{'sentence only':>16}{'+ sections':>13}"
          f"{'delta':>9}")
    for label, rs in (("all", rows), ("core", core), ("arguable", arg)):
        a, b = pct(rs, "pass_without"), pct(rs, "pass_with")
        print(f"  {label:<12}{len(rs):>5}{a:>15.1%}{b:>13.1%}{b - a:>+9.1%}")

    helped = [r for r in rows if r["pass_with"] and not r["pass_without"]]
    hurt = [r for r in rows if r["pass_without"] and not r["pass_with"]]

    print(f"\n  cases the section layer FIXED:  {len(helped)}")
    for r in helped:
        print(f"    + {r['doc'][:62]!r}")
        print(f"        {r['without_sections']} -> {r['with_sections']}")
    print(f"\n  cases the section layer BROKE:  {len(hurt)}")
    for r in hurt:
        print(f"    - {r['doc'][:62]!r}")
        print(f"        {r['without_sections']} -> {r['with_sections']}")
    if not hurt:
        print("    none. The precedence rule -- explicit sentence trigger beats")
        print("    section default -- is what buys this. A section layer that")
        print("    OVERRODE instead of defaulting would invert every negated")
        print("    statement under a FAMILY HISTORY header, which is the")
        print("    failure the gold set was built around.")

    print("\n  remaining failures:")
    fails = [r for r in rows if not r["pass_with"]]
    if not fails:
        print("    none")
    for r in fails:
        tag = " (arguable)" if r["arguable"] else ""
        print(f"    ! {r['doc'][:62]!r}{tag}")
        print(f"        expected {r['expected']}")
        print(f"        got      {r['with_sections']}")

    n_changes = sum(len(r["changes"]) for r in rows)
    print(f"\n  entities whose assertion came from a HEADER rather than from "
          f"words: {n_changes}")
    print("  Every one is reported by apply_section_context() rather than")
    print("  applied silently. 'Which of these came from the header' is the")
    print("  first question anyone reviewing an extraction will ask, and a")
    print("  layer that cannot answer it is unauditable.")

    os.makedirs("out", exist_ok=True)
    with open("out/sections_eval.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print("\nwrote out/sections_eval.json")
    return rows


if __name__ == "__main__":
    main()
