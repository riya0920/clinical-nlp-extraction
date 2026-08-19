"""Rules vs a learned classifier, measured rather than argued.

The earlier build could not answer the spec's question -- "your transformer
beats rules overall but loses on negation precision; what ships to a
risk-adjustment team?" -- because only the rules existed. This runs the
bakeoff.

Both systems are scored on the SAME entity spans (the rule extractor finds
them, both systems then label them), so the comparison isolates the assertion
decision and is not contaminated by a difference in what was found.

Run:  python compare_assertion.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold"))

import adversarial
import assertion_ml as AML
import extract as E
import heldout
from lexicon import ASSERTIONS

OUT = "out"


def per_class(truth, pred, labels=ASSERTIONS):
    """Precision / recall / F1 per assertion class, plus the overall accuracy."""
    out = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(truth, pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(truth, pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and prec + rec else float("nan"))
        out[lab] = {"support": tp + fn, "precision": prec, "recall": rec,
                    "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / len(truth) if truth else 0
    return out, acc


def rule_predictions(rows):
    """Ask the rule layer for the assertion of each specific entity span."""
    preds = []
    for r in rows:
        got = "present"
        for e in E.extract(r["text"]):
            if e.start == r["start"] and e.end == r["end"]:
                got = e.assertion
                break
        preds.append(got)
    return preds


def report(name, truth, pred):
    stats, acc = per_class(truth, pred)
    print(f"\n  {name}   n={len(truth)}   accuracy {acc:.1%}")
    print(f"    {'class':<14}{'support':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    for lab in ASSERTIONS:
        s = stats[lab]
        if s["support"] == 0 and s["fp"] == 0:
            continue
        p = f"{s['precision']:.3f}" if s["precision"] == s["precision"] else "  n/a"
        r = f"{s['recall']:.3f}" if s["recall"] == s["recall"] else "  n/a"
        f = f"{s['f1']:.3f}" if s["f1"] == s["f1"] else "  n/a"
        print(f"    {lab:<14}{s['support']:>8}{p:>11}{r:>9}{f:>8}")
    return stats, acc


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- train the learned model ----------------------------------------
    print("=" * 78)
    print("TRAINING THE LEARNED BASELINE")
    print("=" * 78)
    train_rows = AML.make_corpus(n=6000, seed=5)
    test_rows_id = AML.make_corpus(n=1500, seed=99)      # same templates, new draws
    clf = AML.train(train_rows)
    print(f"  supervised n-gram logistic regression")
    print(f"  trained on {len(train_rows):,} generated sentences, "
          f"{len(set(r['assertion'] for r in train_rows))} classes")
    print("  NOT a pretrained clinical transformer -- no HuggingFace or BERT")
    print("  weights are available offline, and a transformer trained from")
    print("  scratch on templated sentences would be a strawman. This is the")
    print("  standard pre-transformer baseline and is labelled as such.")

    # ---- in-distribution -------------------------------------------------
    print("\n" + "=" * 78)
    print("IN-DISTRIBUTION: held-out generated sentences (same templates)")
    print("=" * 78)
    truth_id = [r["assertion"] for r in test_rows_id]
    pred_ml_id = AML.predict(clf, test_rows_id)
    pred_rule_id = rule_predictions(test_rows_id)
    ml_id, ml_id_acc = report("LEARNED", truth_id, pred_ml_id)
    rule_id, rule_id_acc = report("RULES  ", truth_id, pred_rule_id)

    print("\n  The rules score 100% here BY CONSTRUCTION, and that is not a")
    print("  win: the generator draws its trigger phrases from a subset of the")
    print("  rule lexicon, so every in-distribution sentence is one the rules")
    print("  were built to handle. Reporting this row as evidence for rules")
    print("  would be circular. It is shown only to establish that the learned")
    print("  model did in fact learn the task before it meets unseen phrasing.")

    # ---- out-of-distribution, SPLIT by contamination ----------------------
    # The adversarial set was used to DEVELOP the rules -- six of its cases
    # failed on first run and the lexicon was extended until they passed.
    # Scoring the rules on it and calling the result out-of-distribution would
    # repeat exactly the error this project's README criticises elsewhere. The
    # held-out set was written afterwards, scored once, and never used to fix
    # anything, so it is the only clean comparison available here.
    print("\n" + "=" * 78)
    print("OUT-OF-DISTRIBUTION (A): held-out set -- CLEAN for both systems")
    print("=" * 78)
    rows_ho, truth_ho = AML.rows_from_gold(heldout.CASES, E.extract)
    pred_ml_ho = AML.predict(clf, rows_ho)
    pred_rule_ho = rule_predictions(rows_ho)
    ml_ho, ml_ho_acc = report("LEARNED", truth_ho, pred_ml_ho)
    rule_ho, rule_ho_acc = report("RULES  ", truth_ho, pred_rule_ho)
    print("\n  Neither system saw these sentences during development.")
    print("  This is the row that counts.")

    print("\n" + "=" * 78)
    print("OUT-OF-DISTRIBUTION (B): adversarial set -- CONTAMINATED for rules")
    print("=" * 78)
    rows_adv, truth_adv = AML.rows_from_gold(adversarial.CASES, E.extract)
    pred_ml_adv = AML.predict(clf, rows_adv)
    pred_rule_adv = rule_predictions(rows_adv)
    ml_adv, ml_adv_acc = report("LEARNED", truth_adv, pred_ml_adv)
    rule_adv, rule_adv_acc = report("RULES  ", truth_adv, pred_rule_adv)
    print("\n  The rules were TUNED on these sentences; the learned model never")
    print("  saw them. The gap here is an upper bound on the rules' advantage,")
    print("  not a measurement of it, and quoting it as the headline would be")
    print("  the same error as reporting training accuracy.")

    # everything below uses the CLEAN held-out set
    rows_ood, truth_ood = rows_ho, truth_ho
    pred_ml_ood, pred_rule_ood = pred_ml_ho, pred_rule_ho
    ml_ood, ml_ood_acc = ml_ho, ml_ho_acc
    rule_ood, rule_ood_acc = rule_ho, rule_ho_acc

    # ---- the money comparison -------------------------------------------
    print("\n" + "=" * 78)
    print("THE COMPARISON THAT DECIDES WHAT SHIPS")
    print("=" * 78)
    print(f"  {'':<42}{'LEARNED':>10}{'RULES':>10}{'winner':>10}")
    comparisons = []
    def row(label, a, b, higher_is_better=True):
        if a != a or b != b:
            win = "n/a"
        elif abs(a - b) < 1e-9:
            win = "tie"
        else:
            win = "learned" if (a > b) == higher_is_better else "rules"
        comparisons.append({"metric": label, "learned": a, "rules": b,
                            "winner": win})
        fa = f"{a:.3f}" if a == a else "n/a"
        fb = f"{b:.3f}" if b == b else "n/a"
        print(f"  {label:<42}{fa:>10}{fb:>10}{win:>10}")

    print("  (all OOD figures below are the CLEAN held-out set)")
    row("in-distribution accuracy", ml_id_acc, rule_id_acc)
    row("held-out accuracy", ml_ood_acc, rule_ood_acc)
    row("held-out precision on `absent`",
        ml_ood["absent"]["precision"], rule_ood["absent"]["precision"])
    row("held-out recall on `absent`",
        ml_ood["absent"]["recall"], rule_ood["absent"]["recall"])
    row("held-out precision on `family`",
        ml_ood["family"]["precision"], rule_ood["family"]["precision"])
    row("adversarial accuracy (rules contaminated)", ml_adv_acc, rule_adv_acc)

    # the error that matters most in risk adjustment
    def absent_to_present(truth, pred):
        n = sum(1 for t in truth if t == "absent")
        leak = sum(1 for t, p in zip(truth, pred)
                   if t == "absent" and p == "present")
        return leak, n

    ml_leak, n_abs = absent_to_present(truth_ood, pred_ml_ood)
    rule_leak, _ = absent_to_present(truth_ood, pred_rule_ood)
    ml_fam, rule_fam = (
        sum(1 for t, p in zip(truth_ood, pred_ml_ood)
            if t == "family" and p == "present"),
        sum(1 for t, p in zip(truth_ood, pred_rule_ood)
            if t == "family" and p == "present"))

    print("\n" + "-" * 78)
    print("  THE ERRORS THAT ARE NOT INTERCHANGEABLE (out-of-distribution)")
    print("-" * 78)
    print(f"    negated finding reported as PRESENT   "
          f"learned {ml_leak}/{n_abs}   rules {rule_leak}/{n_abs}")
    print(f"    family history reported as PRESENT    "
          f"learned {ml_fam}          rules {rule_fam}")
    print()
    if ml_leak < rule_leak:
        print()
        print("    NOTE: on the clean held-out set the LEARNED model leaks fewer")
        print("    negations to `present` than the rules do. The rules' misses")
        print("    are the known `neither X nor Y` construction, which is absent")
        print("    from their lexicon. So 'rules win on negation precision' is")
        print("    NOT a law -- rules win when the phrasing is in the lexicon and")
        print("    fail hard when it is not, while the learned model degrades")
        print("    more evenly. That is the real shape of the trade-off, and it")
        print("    is the opposite of what the interview question presumes.")
    print()
    print("  In a risk-adjustment pipeline both of these become submitted")
    print("  diagnoses. A submitted diagnosis unsupported by the record is a")
    print("  potential false-claims exposure -- a legal problem, not a metrics")
    print("  problem. A missed condition is worse for the patient and worse for")
    print("  the plan financially, and is not a compliance event. Accuracy")
    print("  averages the two as though they were the same error.")

    # ---- the verdict -----------------------------------------------------
    print("\n" + "=" * 78)
    print("WHAT SHIPS")
    print("=" * 78)
    ml_wins_id = ml_id_acc > rule_id_acc
    rules_win_ood = rule_ood_acc > ml_ood_acc
    rules_win_precision = (rule_ood["absent"]["precision"]
                           > ml_ood["absent"]["precision"])
    print(f"  learned wins in-distribution      : {ml_wins_id}")
    print(f"  rules win out-of-distribution     : {rules_win_ood}")
    print(f"  rules win OOD `absent` precision  : {rules_win_precision}")
    print()
    if rules_win_precision or rules_win_ood:
        print("  To a RISK-ADJUSTMENT team: ship the rules. The decision is not")
        print("  driven by overall accuracy, it is driven by the asymmetry of")
        print("  the errors and by the fact that production text will not match")
        print("  the training distribution. Rules also fail LEGIBLY -- a missed")
        print("  negation is a lexicon entry someone can add and a regression")
        print("  test someone can pin, whereas a learned model's error is a")
        print("  retraining cycle with no guarantee the specific case is fixed.")
    else:
        print("  On this evidence the learned model is not worse OOD, which")
        print("  weakens the usual argument for rules. It would still need an")
        print("  error-asymmetry-aware threshold before a compliance use.")
    print()
    print("  To a RESEARCH cohort-building team the answer flips: recall")
    print("  matters more than precision, a human reviews the cohort anyway,")
    print("  and the learned model's coverage of unseen phrasing is worth more")
    print("  than the rules' precision. Same measurements, different consumer,")
    print("  different answer -- which is why 'which is better' is not a")
    print("  question with a single answer.")
    print()
    print("  HONEST LIMITS OF THIS BAKEOFF:")
    print("   * the learned model trains on TEMPLATED text, so its")
    print("     out-of-distribution drop is larger than a model trained on")
    print("     real notes would suffer. The direction is right; the magnitude")
    print("     is inflated by the generator's regularity.")
    print("   * it is an n-gram classifier, not a pretrained transformer. A")
    print("     clinical transformer would generalise better on unseen")
    print("     phrasing and would likely narrow the OOD gap substantially.")
    print("   * the rules were tuned on the adversarial set, so that row is")
    print("     reported separately and excluded from the headline. Only the")
    print("     held-out set is clean for both systems.")

    payload = {
        "in_distribution": {"learned": {"accuracy": ml_id_acc, "per_class": ml_id},
                            "rules": {"accuracy": rule_id_acc, "per_class": rule_id}},
        "held_out_clean": {"learned": {"accuracy": ml_ho_acc, "per_class": ml_ho},
                           "rules": {"accuracy": rule_ho_acc, "per_class": rule_ho}},
        "adversarial_contaminated_for_rules": {
            "learned": {"accuracy": ml_adv_acc, "per_class": ml_adv},
            "rules": {"accuracy": rule_adv_acc, "per_class": rule_adv}},
        "comparisons": comparisons,
        "absent_to_present": {"learned": ml_leak, "rules": rule_leak,
                              "n_absent": n_abs},
        "family_to_present": {"learned": ml_fam, "rules": rule_fam},
        "n_train": len(train_rows), "n_ood_entities": len(rows_ood),
    }
    with open(f"{OUT}/rules_vs_learned.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/rules_vs_learned.json")
    return payload


if __name__ == "__main__":
    main()
