"""
Compute baseline metrics from collected debate traces.
Generates the numbers for Table 1 of the paper.

Usage:
    python -m grpo.eval.baseline_metrics --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from collections import Counter
from pathlib import Path


def load_traces(path):
    traces = []
    with open(path) as f:
        for line in f:
            traces.append(json.loads(line))
    return traces


def bootstrap_ci(correct_flags, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    accs = []
    arr = np.array(correct_flags, dtype=float)
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        accs.append(sample.mean())
    lower = np.percentile(accs, (1 - ci) / 2 * 100)
    upper = np.percentile(accs, (1 + ci) / 2 * 100)
    return lower, upper


def compute_metrics(traces):
    """Compute all baseline metrics from debate traces."""
    n = len(traces)

    # === Accuracy metrics ===
    judge_correct = [t["is_correct_judge"] for t in traces]
    majority_correct = [t["is_correct_majority"] for t in traces]
    doctor_a_correct = [t.get("is_correct_doctor_a_r2", False) for t in traces]
    doctor_b_correct = [t.get("is_correct_doctor_b_r2", False) for t in traces]

    judge_acc = sum(judge_correct) / n
    majority_acc = sum(majority_correct) / n
    doc_a_acc = sum(doctor_a_correct) / n
    doc_b_acc = sum(doctor_b_correct) / n

    judge_ci = bootstrap_ci(judge_correct)
    majority_ci = bootstrap_ci(majority_correct)

    print("=" * 65)
    print("BASELINE METRICS (Table 1)")
    print("=" * 65)
    print(f"Total samples: {n}")
    print()
    print(f"{'System':<35} {'Accuracy':>10} {'95% CI':>18}")
    print("-" * 65)
    print(f"{'Doctor A (round 2)':<35} {doc_a_acc:>9.1%}")
    print(f"{'Doctor B (round 2)':<35} {doc_b_acc:>9.1%}")
    print(f"{'Majority vote (no trust)':<35} {majority_acc:>9.1%} [{majority_ci[0]:.1%}, {majority_ci[1]:.1%}]")
    print(f"{'Trust-aware judge':<35} {judge_acc:>9.1%} [{judge_ci[0]:.1%}, {judge_ci[1]:.1%}]")
    print(f"{'Gap (judge - majority)':<35} {judge_acc - majority_acc:>+9.1%}")
    print()

    # === "Maybe" analysis ===
    gold_labels = [t["gold_label"] for t in traces]
    judge_preds = [t["judge_prediction"] for t in traces]

    gold_dist = Counter(gold_labels)
    judge_dist = Counter(judge_preds)

    print("=" * 65)
    print("LABEL DISTRIBUTION")
    print("=" * 65)
    print(f"{'Label':<15} {'Gold':>10} {'Judge pred':>12} {'Majority pred':>14}")
    print("-" * 55)
    majority_preds = [t["majority_vote_prediction"] for t in traces]
    majority_dist = Counter(majority_preds)
    for label in ["yes", "no", "maybe"]:
        print(f"{label:<15} {gold_dist.get(label, 0):>10} {judge_dist.get(label, 0):>12} {majority_dist.get(label, 0):>14}")
    print()

    # === Maybe over-correction analysis ===
    print("=" * 65)
    print("MAYBE OVER-CORRECTION ANALYSIS (Figure 1)")
    print("=" * 65)

    # When gold is yes or no, how often does judge say maybe?
    yes_no_samples = [t for t in traces if t["gold_label"] in ("yes", "no")]
    judge_says_maybe_on_yes_no = sum(1 for t in yes_no_samples if t["judge_prediction"] == "maybe")
    majority_says_maybe_on_yes_no = sum(1 for t in yes_no_samples if t["majority_vote_prediction"] == "maybe")

    print(f"Samples where gold is yes/no: {len(yes_no_samples)}")
    print(f"Judge says 'maybe' on these:  {judge_says_maybe_on_yes_no} ({judge_says_maybe_on_yes_no/len(yes_no_samples):.1%})")
    print(f"Majority says 'maybe' on these: {majority_says_maybe_on_yes_no} ({majority_says_maybe_on_yes_no/len(yes_no_samples):.1%})")
    print()

    # === Confusion matrix ===
    print("=" * 65)
    print("CONFUSION MATRIX: Judge Prediction vs Gold Label")
    print("=" * 65)
    labels = ["yes", "no", "maybe"]
    matrix = {pred: {gold: 0 for gold in labels} for pred in labels}
    for t in traces:
        pred = t["judge_prediction"]
        gold = t["gold_label"]
        if pred in matrix and gold in matrix[pred]:
            matrix[pred][gold] += 1

    print(f"{'Judge \\ Gold':<15}", end="")
    for g in labels:
        print(f"{g:>10}", end="")
    print(f"{'Total':>10}")
    print("-" * 55)
    for pred in labels:
        print(f"{pred:<15}", end="")
        row_total = 0
        for gold in labels:
            count = matrix[pred][gold]
            print(f"{count:>10}", end="")
            row_total += count
        print(f"{row_total:>10}")
    print()

    # === Trust score analysis ===
    print("=" * 65)
    print("TRUST SCORE ANALYSIS")
    print("=" * 65)

    trust_when_correct = [t["trust_signals"]["composite_trust_score"] for t in traces if t["is_correct_judge"]]
    trust_when_wrong = [t["trust_signals"]["composite_trust_score"] for t in traces if not t["is_correct_judge"]]
    trust_when_maybe = [t["trust_signals"]["composite_trust_score"] for t in traces if t["judge_prediction"] == "maybe"]
    trust_when_not_maybe = [t["trust_signals"]["composite_trust_score"] for t in traces if t["judge_prediction"] != "maybe"]

    if trust_when_correct:
        print(f"Avg trust (judge correct):    {np.mean(trust_when_correct):.4f} (n={len(trust_when_correct)})")
    if trust_when_wrong:
        print(f"Avg trust (judge wrong):      {np.mean(trust_when_wrong):.4f} (n={len(trust_when_wrong)})")
    if trust_when_maybe:
        print(f"Avg trust (judge says maybe): {np.mean(trust_when_maybe):.4f} (n={len(trust_when_maybe)})")
    if trust_when_not_maybe:
        print(f"Avg trust (judge says y/n):   {np.mean(trust_when_not_maybe):.4f} (n={len(trust_when_not_maybe)})")

    # Trust sub-signal breakdown
    print()
    print("Trust sub-signal means:")
    for signal in ["agreement_score", "embedding_similarity", "confidence_stability"]:
        correct_vals = [t["trust_signals"][signal] for t in traces if t["is_correct_judge"]]
        wrong_vals = [t["trust_signals"][signal] for t in traces if not t["is_correct_judge"]]
        if correct_vals and wrong_vals:
            print(f"  {signal:<30} correct={np.mean(correct_vals):.4f}  wrong={np.mean(wrong_vals):.4f}  gap={np.mean(correct_vals)-np.mean(wrong_vals):+.4f}")

    # === Correlation ===
    trust_scores = [t["trust_signals"]["composite_trust_score"] for t in traces]
    correct_flags = [1 if t["is_correct_judge"] else 0 for t in traces]
    correlation = np.corrcoef(trust_scores, correct_flags)[0, 1]
    print(f"\nTrust-accuracy correlation: {correlation:.4f}")

    # === Per-round analysis ===
    print()
    print("=" * 65)
    print("PER-ROUND ANALYSIS")
    print("=" * 65)

    # Did round 2 help or hurt?
    r1_agreement = sum(1 for t in traces if t["round_1"]["doctor_a_answer"] == t["round_1"]["doctor_b_answer"])
    r2_agreement = sum(1 for t in traces if t["round_2"]["doctor_a_answer"] == t["round_2"]["doctor_b_answer"])
    print(f"Doctor agreement after round 1: {r1_agreement}/{n} ({r1_agreement/n:.1%})")
    print(f"Doctor agreement after round 2: {r2_agreement}/{n} ({r2_agreement/n:.1%})")

    # Cases where doctors changed their answer between rounds
    a_changed = sum(1 for t in traces if t["round_1"]["doctor_a_answer"] != t["round_2"]["doctor_a_answer"])
    b_changed = sum(1 for t in traces if t["round_1"]["doctor_b_answer"] != t["round_2"]["doctor_b_answer"])
    print(f"Doctor A changed answer r1->r2: {a_changed}/{n} ({a_changed/n:.1%})")
    print(f"Doctor B changed answer r1->r2: {b_changed}/{n} ({b_changed/n:.1%})")

    return {
        "n": n,
        "judge_accuracy": judge_acc,
        "majority_accuracy": majority_acc,
        "judge_ci": judge_ci,
        "majority_ci": majority_ci,
        "maybe_overcorrection_rate": judge_says_maybe_on_yes_no / len(yes_no_samples) if yes_no_samples else 0,
        "trust_accuracy_correlation": correlation,
        "gold_distribution": dict(gold_dist),
        "judge_distribution": dict(judge_dist),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--save_json", type=str, default="experiments/results/baseline_metrics.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)
    metrics = compute_metrics(traces)

    # Save metrics as JSON for later comparison
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {args.save_json}")


if __name__ == "__main__":
    main()
