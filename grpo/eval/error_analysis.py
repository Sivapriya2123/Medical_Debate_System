"""
Error Analysis: Categorize failures in the GRPO-optimized system.

Examines the ~21% of test samples where the best system (decisive_v1)
gets the wrong answer. Categorizes errors by failure mode.

Usage:
    python -m grpo.eval.error_analysis --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


def load_test_traces(path, train_ratio=0.8):
    with open(path) as f:
        all_traces = [json.loads(line) for line in f]
    split_idx = int(len(all_traces) * train_ratio)
    return all_traces[split_idx:]


def categorize_error(trace):
    """
    Categorize why the system got this question wrong.
    Returns a list of error categories (a sample can have multiple).
    """
    categories = []

    gold = trace["gold_label"]
    judge_pred = trace.get("judge_prediction", "maybe")
    majority_pred = trace.get("majority_vote_prediction", "maybe")

    r2 = trace.get("round_2", {})
    doc_a_answer = r2.get("doctor_a_answer", "")
    doc_b_answer = r2.get("doctor_b_answer", "")

    ts = trace.get("trust_signals", {})
    agreement = ts.get("agreement_score", 0.5)
    composite = ts.get("composite_trust_score", 0.5)

    # Category 1: Both doctors wrong
    doc_a_correct = (doc_a_answer.lower().strip() == gold.lower().strip())
    doc_b_correct = (doc_b_answer.lower().strip() == gold.lower().strip())

    if not doc_a_correct and not doc_b_correct:
        categories.append("both_doctors_wrong")
    elif not doc_a_correct and doc_b_correct:
        categories.append("doctor_a_wrong_b_right")
    elif doc_a_correct and not doc_b_correct:
        categories.append("doctor_b_wrong_a_right")

    # Category 2: Maybe over-correction (judge said maybe, gold is yes/no)
    if judge_pred == "maybe" and gold in ("yes", "no"):
        categories.append("maybe_overcorrection")

    # Category 3: Gold is "maybe" (inherently hard)
    if gold == "maybe":
        categories.append("gold_is_maybe")

    # Category 4: High trust but wrong
    if composite > 0.85 and judge_pred != gold:
        categories.append("high_trust_but_wrong")

    # Category 5: Doctors agree but wrong
    if doc_a_answer == doc_b_answer and doc_a_answer != gold:
        categories.append("doctors_agree_but_wrong")

    # Category 6: Doctors disagree and judge picked wrong one
    if doc_a_answer != doc_b_answer:
        if doc_a_correct and judge_pred != gold:
            categories.append("judge_ignored_correct_doctor_a")
        elif doc_b_correct and judge_pred != gold:
            categories.append("judge_ignored_correct_doctor_b")

    # Category 7: Direction flip (judge said yes but gold is no, or vice versa)
    if judge_pred in ("yes", "no") and gold in ("yes", "no") and judge_pred != gold:
        categories.append("direction_flip")

    if not categories:
        categories.append("uncategorized")

    return categories


def run_error_analysis(traces):
    """Full error analysis on test traces."""

    # Separate correct vs incorrect
    correct_traces = [t for t in traces if t.get("is_correct_judge")]
    error_traces = [t for t in traces if not t.get("is_correct_judge")]

    print("=" * 70)
    print("ERROR ANALYSIS -- GRPO-Optimized Judge System")
    print("=" * 70)
    print(f"Total test samples: {len(traces)}")
    print(f"Correct: {len(correct_traces)} ({len(correct_traces)/len(traces):.1%})")
    print(f"Errors:  {len(error_traces)} ({len(error_traces)/len(traces):.1%})")

    # Categorize all errors
    error_categories = Counter()
    category_examples = defaultdict(list)

    for trace in error_traces:
        cats = categorize_error(trace)
        for cat in cats:
            error_categories[cat] += 1
            if len(category_examples[cat]) < 3:  # keep up to 3 examples per category
                category_examples[cat].append({
                    "question": trace.get("question", "")[:120],
                    "gold": trace["gold_label"],
                    "judge_pred": trace.get("judge_prediction", ""),
                    "doc_a": trace.get("round_2", {}).get("doctor_a_answer", ""),
                    "doc_b": trace.get("round_2", {}).get("doctor_b_answer", ""),
                    "trust": trace.get("trust_signals", {}).get("composite_trust_score", 0),
                })

    print(f"\n{'='*70}")
    print("ERROR CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    print(f"(Note: one error can belong to multiple categories)")
    print(f"\n{'Category':<35} {'Count':>6} {'% of errors':>12}")
    print("-" * 55)

    for cat, count in error_categories.most_common():
        pct = count / len(error_traces) if error_traces else 0
        print(f"  {cat:<35} {count:>5} {pct:>11.1%}")

    # Detailed breakdown by gold label
    print(f"\n{'='*70}")
    print("ERROR DISTRIBUTION BY GOLD LABEL")
    print(f"{'='*70}")

    for gold_label in ["yes", "no", "maybe"]:
        gold_traces = [t for t in traces if t["gold_label"] == gold_label]
        gold_errors = [t for t in error_traces if t["gold_label"] == gold_label]
        if gold_traces:
            print(f"\n  Gold = {gold_label}: {len(gold_errors)}/{len(gold_traces)} errors ({len(gold_errors)/len(gold_traces):.1%})")

            # What did the judge predict instead?
            pred_dist = Counter(t.get("judge_prediction", "") for t in gold_errors)
            for pred, count in pred_dist.most_common():
                print(f"    -> predicted '{pred}': {count}")

    # The "both doctors wrong" analysis
    print(f"\n{'='*70}")
    print("UNRECOVERABLE ERRORS (both doctors wrong)")
    print(f"{'='*70}")

    both_wrong = [t for t in error_traces if "both_doctors_wrong" in categorize_error(t)]
    recoverable = [t for t in error_traces if "both_doctors_wrong" not in categorize_error(t)]

    print(f"Both doctors wrong: {len(both_wrong)}/{len(error_traces)} ({len(both_wrong)/len(error_traces):.1%} of errors)")
    print(f"Recoverable errors: {len(recoverable)}/{len(error_traces)} ({len(recoverable)/len(error_traces):.1%} of errors)")
    print(f"\nTheoretical ceiling if judge were perfect on recoverable errors:")
    theoretical_max = (len(correct_traces) + len(recoverable)) / len(traces)
    print(f"  Max possible accuracy: {theoretical_max:.1%}")
    print(f"  Current accuracy:      {len(correct_traces)/len(traces):.1%}")
    print(f"  Remaining gap:         {theoretical_max - len(correct_traces)/len(traces):.1%}")

    # Trust score comparison: correct vs error
    print(f"\n{'='*70}")
    print("TRUST SIGNALS: CORRECT vs ERROR SAMPLES")
    print(f"{'='*70}")

    for signal_name in ["agreement_score", "embedding_similarity", "confidence_stability", "composite_trust_score"]:
        correct_vals = [t.get("trust_signals", {}).get(signal_name, 0) for t in correct_traces]
        error_vals = [t.get("trust_signals", {}).get(signal_name, 0) for t in error_traces]
        if correct_vals and error_vals:
            print(f"  {signal_name:<25} correct={np.mean(correct_vals):.4f}  error={np.mean(error_vals):.4f}  gap={np.mean(correct_vals)-np.mean(error_vals):+.4f}")

    # Example errors for qualitative analysis
    print(f"\n{'='*70}")
    print("EXAMPLE ERRORS (for qualitative discussion in paper)")
    print(f"{'='*70}")

    for cat in ["both_doctors_wrong", "maybe_overcorrection", "doctors_agree_but_wrong", "direction_flip", "gold_is_maybe"]:
        examples = category_examples.get(cat, [])
        if examples:
            print(f"\n  --- {cat} ---")
            for i, ex in enumerate(examples[:2]):
                print(f"  Example {i+1}:")
                print(f"    Q: {ex['question']}...")
                print(f"    Gold: {ex['gold']} | Judge: {ex['judge_pred']} | Doc A: {ex['doc_a']} | Doc B: {ex['doc_b']} | Trust: {ex['trust']:.3f}")

    # Save full error analysis
    output = {
        "total_samples": len(traces),
        "total_errors": len(error_traces),
        "error_rate": len(error_traces) / len(traces),
        "category_counts": dict(error_categories),
        "both_doctors_wrong_count": len(both_wrong),
        "recoverable_count": len(recoverable),
        "theoretical_max_accuracy": theoretical_max,
        "examples": dict(category_examples),
    }

    output_path = Path("experiments/results/error_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull error analysis saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    traces = load_test_traces(args.traces)
    run_error_analysis(traces)


if __name__ == "__main__":
    main()
