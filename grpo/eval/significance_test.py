"""
Statistical significance tests between system variants.

McNemar's test for paired binary outcomes.

Usage:
    python -m grpo.eval.significance_test --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_test_traces(path, train_ratio=0.8):
    with open(path) as f:
        all_traces = [json.loads(line) for line in f]
    split_idx = int(len(all_traces) * train_ratio)
    return all_traces[split_idx:]


def mcnemar_test(a_correct, b_correct):
    """McNemar's test for paired binary data."""
    assert len(a_correct) == len(b_correct)

    b = sum(1 for a, bb in zip(a_correct, b_correct) if not a and bb)  # A wrong, B right
    c = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)  # A right, B wrong

    n = b + c
    if n == 0:
        return b, c, 1.0

    # Use scipy binomial test
    try:
        from scipy.stats import binomtest
        result = binomtest(b, n, 0.5)
        p_value = result.pvalue
    except (ImportError, AttributeError):
        try:
            from scipy.stats import binom_test
            p_value = binom_test(b, n, 0.5)
        except ImportError:
            # Fallback: chi-squared approximation
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
            from math import exp, sqrt, pi
            # Simple approximation
            p_value = max(0.001, 1.0 - 0.5 * (1 + (1 - exp(-chi2/2))))

    return b, c, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    traces = load_test_traces(args.traces)
    n = len(traces)

    print("=" * 65)
    print(f"STATISTICAL SIGNIFICANCE TESTS (n={n} test samples)")
    print("=" * 65)

    judge_correct = [t.get("is_correct_judge", False) for t in traces]
    majority_correct = [t.get("is_correct_majority", False) for t in traces]

    print(f"\n--- McNemar's Test: Static Trust Judge vs Majority Vote ---")
    b, c, p = mcnemar_test(judge_correct, majority_correct)
    print(f"Majority right, Judge wrong: {b}")
    print(f"Judge right, Majority wrong: {c}")
    print(f"Total discordant pairs: {b + c}")
    print(f"P-value: {p:.4f}")
    print(f"Significant at 0.05? {'Yes' if p < 0.05 else 'No'}")
    print(f"Significant at 0.10? {'Yes' if p < 0.10 else 'No'}")

    both_correct = sum(1 for j, m in zip(judge_correct, majority_correct) if j and m)
    both_wrong = sum(1 for j, m in zip(judge_correct, majority_correct) if not j and not m)
    print(f"\nBoth correct: {both_correct}/{n}")
    print(f"Both wrong: {both_wrong}/{n}")
    print(f"Only judge correct: {c}/{n}")
    print(f"Only majority correct: {b}/{n}")

    # Bootstrap confidence intervals
    print(f"\n--- Bootstrap 95% Confidence Intervals ---")
    np.random.seed(42)
    n_bootstrap = 10000

    judge_accs = []
    majority_accs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        judge_accs.append(np.mean([judge_correct[i] for i in idx]))
        majority_accs.append(np.mean([majority_correct[i] for i in idx]))

    j_ci = np.percentile(judge_accs, [2.5, 97.5])
    m_ci = np.percentile(majority_accs, [2.5, 97.5])
    print(f"Judge accuracy:    {np.mean(judge_correct):.1%}  95% CI: [{j_ci[0]:.1%}, {j_ci[1]:.1%}]")
    print(f"Majority accuracy: {np.mean(majority_correct):.1%}  95% CI: [{m_ci[0]:.1%}, {m_ci[1]:.1%}]")
    print(f"CIs overlap? {'Yes' if j_ci[1] >= m_ci[0] and m_ci[1] >= j_ci[0] else 'No'}")

    results = {
        "test_size": n,
        "judge_accuracy": sum(judge_correct) / n,
        "majority_accuracy": sum(majority_correct) / n,
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p_value": p,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "judge_ci_95": list(j_ci),
        "majority_ci_95": list(m_ci),
    }

    output_path = Path("experiments/results/significance_tests.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
