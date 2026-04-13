"""
Trust Weight Optimization via Grid Search and Learned Weighting.

Phase 2 of the GRPO integration: instead of static 0.40/0.35/0.25 weights,
find the optimal weight combination for the trust composite score.

This runs entirely on existing debate traces — no new API calls needed.

Usage:
    # Grid search
    python -m grpo.training.trust_weight_optimizer --mode grid_search --traces experiments/traces/debate_traces_full.jsonl

    # Per-regime analysis
    python -m grpo.training.trust_weight_optimizer --mode regime_analysis --traces experiments/traces/debate_traces_full.jsonl

    # Train a learned weight predictor
    python -m grpo.training.trust_weight_optimizer --mode learn_weights --traces experiments/traces/debate_traces_full.jsonl

    # Full pipeline
    python -m grpo.training.trust_weight_optimizer --mode full --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import product


def load_traces(path, split="train", train_ratio=0.8):
    """Load traces and split into train/test."""
    with open(path) as f:
        all_traces = [json.loads(line) for line in f]

    split_idx = int(len(all_traces) * train_ratio)

    if split == "train":
        return all_traces[:split_idx]
    elif split == "test":
        return all_traces[split_idx:]
    else:
        return all_traces


def get_trust_signals(trace):
    """Extract the three trust sub-signals from a trace."""
    ts = trace.get("trust_signals", {})
    return {
        "agreement": ts.get("agreement_score", 0.5),
        "similarity": ts.get("embedding_similarity", 0.5),
        "stability": ts.get("confidence_stability", 0.5),
    }


def compute_composite_trust(signals, weights):
    """Compute weighted composite trust score."""
    return (
        weights[0] * signals["agreement"] +
        weights[1] * signals["similarity"] +
        weights[2] * signals["stability"]
    )


def simulate_judge_with_trust_threshold(traces, weights, high_trust_threshold=0.85, low_trust_threshold=0.60):
    """
    Simulate judge decision-making with given trust weights and thresholds.

    Logic:
    - High composite trust -> follow doctor majority (be decisive)
    - Medium composite trust -> follow doctor majority but allow maybe
    - Low composite trust -> use original judge decision

    This simulates what the judge WOULD have decided with different trust weights.
    """
    correct = 0
    maybe_on_yes_no = 0
    yes_no_total = 0
    decisions = []

    for trace in traces:
        signals = get_trust_signals(trace)
        composite = compute_composite_trust(signals, weights)

        gold = trace["gold_label"]
        majority_pred = trace.get("majority_vote_prediction", "maybe")
        judge_pred = trace.get("judge_prediction", "maybe")

        # Decision logic based on trust level
        if composite >= high_trust_threshold:
            # High trust: follow majority vote (be decisive)
            decision = majority_pred
        elif composite >= low_trust_threshold:
            # Medium trust: follow majority but keep judge's "maybe" if both doctors disagree
            r2 = trace.get("round_2", {})
            doc_a = r2.get("doctor_a_answer", "")
            doc_b = r2.get("doctor_b_answer", "")
            if doc_a == doc_b:
                decision = doc_a  # doctors agree, follow them
            else:
                decision = judge_pred  # doctors disagree, trust the judge
        else:
            # Low trust: use judge's original decision
            decision = judge_pred

        is_correct = (decision == gold)
        correct += int(is_correct)

        if gold in ("yes", "no"):
            yes_no_total += 1
            if decision == "maybe":
                maybe_on_yes_no += 1

        decisions.append({
            "gold": gold,
            "decision": decision,
            "composite_trust": composite,
            "is_correct": is_correct,
        })

    accuracy = correct / len(traces) if traces else 0
    maybe_rate = maybe_on_yes_no / yes_no_total if yes_no_total > 0 else 0

    return {
        "accuracy": accuracy,
        "maybe_rate": maybe_rate,
        "n": len(traces),
        "weights": list(weights),
        "decisions": decisions,
    }


# ============================================================
# METHOD 1: Grid Search
# ============================================================

def grid_search(traces, step_size=0.05):
    """
    Exhaustive grid search over weight combinations.
    Weights must sum to 1.0, each between 0.0 and 1.0.
    """
    print("=" * 65)
    print("GRID SEARCH OVER TRUST WEIGHTS")
    print("=" * 65)

    best_result = None
    best_accuracy = 0
    all_results = []

    # Generate weight combinations that sum to 1.0
    steps = np.arange(0.0, 1.0 + step_size, step_size)
    n_combos = 0

    for w_agree in steps:
        for w_sim in steps:
            w_stab = 1.0 - w_agree - w_sim
            if w_stab < -0.001 or w_stab > 1.001:
                continue
            w_stab = max(0.0, min(1.0, w_stab))

            weights = (w_agree, w_sim, w_stab)
            result = simulate_judge_with_trust_threshold(traces, weights)
            all_results.append(result)
            n_combos += 1

            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_result = result

    print(f"Tested {n_combos} weight combinations")
    print(f"\nCurrent weights: (0.40, 0.35, 0.25)")

    # Score current weights for comparison
    current = simulate_judge_with_trust_threshold(traces, (0.40, 0.35, 0.25))
    print(f"Current accuracy: {current['accuracy']:.1%}, maybe rate: {current['maybe_rate']:.1%}")

    print(f"\nBest weights: ({best_result['weights'][0]:.2f}, {best_result['weights'][1]:.2f}, {best_result['weights'][2]:.2f})")
    print(f"Best accuracy: {best_result['accuracy']:.1%}, maybe rate: {best_result['maybe_rate']:.1%}")
    print(f"Improvement: {best_result['accuracy'] - current['accuracy']:+.1%}")

    # Top 10 weight combinations
    all_results.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"\nTop 10 weight combinations:")
    print(f"{'Rank':<6} {'Agreement':>10} {'Similarity':>11} {'Stability':>10} {'Accuracy':>10} {'Maybe%':>8}")
    print("-" * 58)
    for i, r in enumerate(all_results[:10]):
        w = r["weights"]
        marker = " <- current" if abs(w[0]-0.40)<0.01 and abs(w[1]-0.35)<0.01 else ""
        print(f"{i+1:<6} {w[0]:>10.2f} {w[1]:>11.2f} {w[2]:>10.2f} {r['accuracy']:>9.1%} {r['maybe_rate']:>7.1%}{marker}")

    # Also show: what if we set one weight to 0?
    print(f"\nSingle-signal ablation (what if we remove one signal?):")
    ablations = {
        "No agreement":  (0.00, 0.50, 0.50),
        "No similarity":  (0.50, 0.00, 0.50),
        "No stability":  (0.50, 0.50, 0.00),
        "Agreement only": (1.00, 0.00, 0.00),
        "Similarity only": (0.00, 1.00, 0.00),
        "Stability only": (0.00, 0.00, 1.00),
    }
    for name, w in ablations.items():
        r = simulate_judge_with_trust_threshold(traces, w)
        print(f"  {name:<20} accuracy={r['accuracy']:.1%}  maybe={r['maybe_rate']:.1%}")

    return best_result, all_results


# ============================================================
# METHOD 2: Per-Regime Analysis
# ============================================================

def regime_analysis(traces):
    """
    Analyze which trust weights work best in different trust regimes.
    Splits traces by trust level and finds optimal weights for each.
    """
    print("\n" + "=" * 65)
    print("PER-REGIME TRUST ANALYSIS")
    print("=" * 65)

    # Categorize traces by trust regime
    regimes = {
        "high_agreement": [],     # both doctors agree
        "low_agreement": [],      # doctors disagree
        "high_trust": [],         # composite > 0.85
        "medium_trust": [],       # 0.60 < composite < 0.85
        "low_trust": [],          # composite < 0.60
        "gold_yes": [],
        "gold_no": [],
        "gold_maybe": [],
    }

    for trace in traces:
        ts = trace.get("trust_signals", {})
        composite = ts.get("composite_trust_score", 0.5)
        agreement = ts.get("agreement_score", 0.5)
        gold = trace["gold_label"]

        if agreement > 0.7:
            regimes["high_agreement"].append(trace)
        else:
            regimes["low_agreement"].append(trace)

        if composite > 0.85:
            regimes["high_trust"].append(trace)
        elif composite > 0.60:
            regimes["medium_trust"].append(trace)
        else:
            regimes["low_trust"].append(trace)

        regimes[f"gold_{gold}"].append(trace)

    print(f"\nRegime sizes:")
    for name, group in regimes.items():
        print(f"  {name:<20}: {len(group)} samples")

    # For each regime, find which decision strategy works best
    print(f"\nPer-regime accuracy analysis:")
    print(f"{'Regime':<20} {'N':>5} {'Judge':>8} {'Majority':>10} {'Best':>8} {'Winner':>10}")
    print("-" * 65)

    for name, group in regimes.items():
        if len(group) < 10:
            continue

        judge_correct = sum(1 for t in group if t.get("is_correct_judge"))
        majority_correct = sum(1 for t in group if t.get("is_correct_majority"))
        n = len(group)

        judge_acc = judge_correct / n
        majority_acc = majority_correct / n
        best_acc = max(judge_acc, majority_acc)
        winner = "Judge" if judge_acc > majority_acc else "Majority" if majority_acc > judge_acc else "Tie"

        print(f"  {name:<20} {n:>5} {judge_acc:>7.1%} {majority_acc:>9.1%} {best_acc:>7.1%} {winner:>10}")

    # Key insight: when should we trust the judge vs majority?
    print(f"\nKey insight -- optimal strategy per regime:")

    for name, group in [("high_agreement", regimes["high_agreement"]),
                         ("low_agreement", regimes["low_agreement"])]:
        if not group:
            continue
        judge_acc = sum(1 for t in group if t.get("is_correct_judge")) / len(group)
        majority_acc = sum(1 for t in group if t.get("is_correct_majority")) / len(group)

        if majority_acc > judge_acc:
            print(f"  {name}: Follow MAJORITY vote ({majority_acc:.1%} vs judge {judge_acc:.1%})")
        else:
            print(f"  {name}: Follow JUDGE ({judge_acc:.1%} vs majority {majority_acc:.1%})")

    return regimes


# ============================================================
# METHOD 3: Learned Weight Predictor
# ============================================================

def learn_adaptive_weights(traces):
    """
    Learn a simple rule-based adaptive weight system.

    Instead of a neural network, we use the regime analysis to build
    an if/else weight selector that picks different weights based on
    the trust signal values.

    This is more interpretable and doesn't need torch/GPU.
    """
    print("\n" + "=" * 65)
    print("ADAPTIVE TRUST WEIGHT LEARNING")
    print("=" * 65)

    # Strategy: grid search within each regime to find regime-specific optimal weights

    def categorize_trace(trace):
        """Assign trace to a trust regime."""
        ts = trace.get("trust_signals", {})
        agreement = ts.get("agreement_score", 0.5)

        if agreement > 0.7:
            return "doctors_agree"
        elif agreement < 0.3:
            return "doctors_disagree"
        else:
            return "doctors_mixed"

    # Group traces by regime
    regime_traces = defaultdict(list)
    for trace in traces:
        regime = categorize_trace(trace)
        regime_traces[regime].append(trace)

    print(f"Regime distribution:")
    for regime, group in regime_traces.items():
        print(f"  {regime}: {len(group)} samples")

    # Grid search within each regime
    regime_weights = {}
    step = 0.1
    steps = np.arange(0.0, 1.0 + step, step)

    for regime, group in regime_traces.items():
        if len(group) < 10:
            regime_weights[regime] = (0.40, 0.35, 0.25)  # fallback to default
            continue

        best_acc = 0
        best_w = (0.40, 0.35, 0.25)

        for w_agree in steps:
            for w_sim in steps:
                w_stab = 1.0 - w_agree - w_sim
                if w_stab < -0.001 or w_stab > 1.001:
                    continue
                w_stab = max(0.0, min(1.0, w_stab))

                result = simulate_judge_with_trust_threshold(group, (w_agree, w_sim, w_stab))
                if result["accuracy"] > best_acc:
                    best_acc = result["accuracy"]
                    best_w = (w_agree, w_sim, w_stab)

        regime_weights[regime] = best_w
        print(f"\n  {regime}:")
        print(f"    Optimal weights: ({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f})")
        print(f"    Accuracy with optimal: {best_acc:.1%}")

        # Compare to static weights
        static_result = simulate_judge_with_trust_threshold(group, (0.40, 0.35, 0.25))
        print(f"    Accuracy with static:  {static_result['accuracy']:.1%}")
        print(f"    Improvement: {best_acc - static_result['accuracy']:+.1%}")

    # Now simulate the full adaptive system
    print(f"\n{'='*65}")
    print("ADAPTIVE SYSTEM SIMULATION (full training set)")
    print(f"{'='*65}")

    correct_adaptive = 0
    correct_static = 0
    correct_majority = 0
    maybe_adaptive = 0
    yes_no_total = 0

    for trace in traces:
        regime = categorize_trace(trace)
        adaptive_w = regime_weights.get(regime, (0.40, 0.35, 0.25))

        signals = get_trust_signals(trace)
        adaptive_composite = compute_composite_trust(signals, adaptive_w)
        static_composite = compute_composite_trust(signals, (0.40, 0.35, 0.25))

        gold = trace["gold_label"]
        majority_pred = trace.get("majority_vote_prediction", "maybe")
        judge_pred = trace.get("judge_prediction", "maybe")

        # Adaptive decision
        r2 = trace.get("round_2", {})
        doc_a = r2.get("doctor_a_answer", "")
        doc_b = r2.get("doctor_b_answer", "")

        if adaptive_composite >= 0.85:
            adaptive_decision = majority_pred
        elif doc_a == doc_b:
            adaptive_decision = doc_a
        else:
            adaptive_decision = judge_pred

        correct_adaptive += int(adaptive_decision == gold)
        correct_static += int(judge_pred == gold)
        correct_majority += int(majority_pred == gold)

        if gold in ("yes", "no"):
            yes_no_total += 1
            if adaptive_decision == "maybe":
                maybe_adaptive += 1

    n = len(traces)
    print(f"\n{'System':<35} {'Accuracy':>10} {'Maybe rate':>12}")
    print("-" * 60)
    print(f"{'Majority vote':<35} {correct_majority/n:>9.1%} {'--':>12}")
    print(f"{'Static trust (0.40/0.35/0.25)':<35} {correct_static/n:>9.1%} {'--':>12}")
    print(f"{'Adaptive trust (learned weights)':<35} {correct_adaptive/n:>9.1%} {maybe_adaptive/yes_no_total:>11.1%}")

    return regime_weights


# ============================================================
# METHOD 4: Threshold Optimization
# ============================================================

def optimize_thresholds(traces):
    """
    Optimize the high/low trust thresholds used in the decision logic.
    """
    print("\n" + "=" * 65)
    print("TRUST THRESHOLD OPTIMIZATION")
    print("=" * 65)

    best_acc = 0
    best_params = (0.85, 0.60)
    results = []

    for high_thresh in np.arange(0.70, 0.96, 0.05):
        for low_thresh in np.arange(0.40, high_thresh, 0.05):
            result = simulate_judge_with_trust_threshold(
                traces, (0.40, 0.35, 0.25),
                high_trust_threshold=high_thresh,
                low_trust_threshold=low_thresh
            )
            results.append({
                "high_threshold": round(high_thresh, 2),
                "low_threshold": round(low_thresh, 2),
                "accuracy": result["accuracy"],
                "maybe_rate": result["maybe_rate"],
            })

            if result["accuracy"] > best_acc:
                best_acc = result["accuracy"]
                best_params = (round(high_thresh, 2), round(low_thresh, 2))

    print(f"Current thresholds: high=0.85, low=0.60")
    current = simulate_judge_with_trust_threshold(traces, (0.40, 0.35, 0.25), 0.85, 0.60)
    print(f"Current accuracy: {current['accuracy']:.1%}")

    print(f"\nOptimal thresholds: high={best_params[0]}, low={best_params[1]}")
    print(f"Optimal accuracy: {best_acc:.1%}")
    print(f"Improvement: {best_acc - current['accuracy']:+.1%}")

    # Top 5 threshold combinations
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"\nTop 5 threshold combinations:")
    print(f"{'Rank':<6} {'High':>8} {'Low':>8} {'Accuracy':>10} {'Maybe%':>8}")
    print("-" * 42)
    for i, r in enumerate(results[:5]):
        print(f"{i+1:<6} {r['high_threshold']:>8.2f} {r['low_threshold']:>8.2f} {r['accuracy']:>9.1%} {r['maybe_rate']:>7.1%}")

    return best_params, results


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline(traces_path):
    """Run all trust weight optimization methods and compile results."""

    train_traces = load_traces(traces_path, "train")
    test_traces = load_traces(traces_path, "test")

    print(f"Train: {len(train_traces)} samples | Test: {len(test_traces)} samples")

    # 1. Grid search
    print("\n\n>>> METHOD 1: GRID SEARCH")
    best_grid, all_grid = grid_search(train_traces)

    # 2. Regime analysis
    print("\n\n>>> METHOD 2: REGIME ANALYSIS")
    regimes = regime_analysis(train_traces)

    # 3. Adaptive weight learning
    print("\n\n>>> METHOD 3: ADAPTIVE WEIGHT LEARNING")
    regime_weights = learn_adaptive_weights(train_traces)

    # 4. Threshold optimization
    print("\n\n>>> METHOD 4: THRESHOLD OPTIMIZATION")
    best_thresholds, _ = optimize_thresholds(train_traces)

    # ==========================================
    # FINAL: Evaluate all methods on TEST set
    # ==========================================
    print("\n\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    # Static weights
    static = simulate_judge_with_trust_threshold(test_traces, (0.40, 0.35, 0.25))

    # Best grid weights
    best_w = tuple(best_grid["weights"])
    grid_result = simulate_judge_with_trust_threshold(test_traces, best_w)

    # Best grid weights + optimized thresholds
    grid_thresh = simulate_judge_with_trust_threshold(
        test_traces, best_w, best_thresholds[0], best_thresholds[1]
    )

    # Adaptive per-regime weights
    # Re-simulate adaptive on test set
    correct_adaptive = 0
    maybe_adaptive = 0
    yes_no_total = 0

    def categorize_trace(trace):
        ts = trace.get("trust_signals", {})
        agreement = ts.get("agreement_score", 0.5)
        if agreement > 0.7:
            return "doctors_agree"
        elif agreement < 0.3:
            return "doctors_disagree"
        else:
            return "doctors_mixed"

    for trace in test_traces:
        regime = categorize_trace(trace)
        w = regime_weights.get(regime, (0.40, 0.35, 0.25))
        signals = get_trust_signals(trace)
        composite = compute_composite_trust(signals, w)

        gold = trace["gold_label"]
        majority_pred = trace.get("majority_vote_prediction", "maybe")
        judge_pred = trace.get("judge_prediction", "maybe")

        r2 = trace.get("round_2", {})
        doc_a = r2.get("doctor_a_answer", "")
        doc_b = r2.get("doctor_b_answer", "")

        if composite >= 0.85:
            decision = majority_pred
        elif doc_a == doc_b:
            decision = doc_a
        else:
            decision = judge_pred

        correct_adaptive += int(decision == gold)
        if gold in ("yes", "no"):
            yes_no_total += 1
            if decision == "maybe":
                maybe_adaptive += 1

    n = len(test_traces)
    adaptive_acc = correct_adaptive / n
    adaptive_maybe = maybe_adaptive / yes_no_total if yes_no_total > 0 else 0

    # Majority vote baseline
    majority_correct = sum(1 for t in test_traces if t.get("is_correct_majority"))
    majority_acc = majority_correct / n

    print(f"\n{'System':<45} {'Accuracy':>10} {'Maybe%':>8}")
    print("-" * 65)
    print(f"{'Majority vote (no trust)':<45} {majority_acc:>9.1%} {'--':>8}")
    print(f"{'Static trust (0.40/0.35/0.25)':<45} {static['accuracy']:>9.1%} {static['maybe_rate']:>7.1%}")
    print(f"{'Grid-optimized weights':<45} {grid_result['accuracy']:>9.1%} {grid_result['maybe_rate']:>7.1%}")
    print(f"{'Grid weights + optimized thresholds':<45} {grid_thresh['accuracy']:>9.1%} {grid_thresh['maybe_rate']:>7.1%}")
    print(f"{'Adaptive per-regime weights':<45} {adaptive_acc:>9.1%} {adaptive_maybe:>7.1%}")

    print(f"\nBest grid weights: ({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f})")
    print(f"Best thresholds: high={best_thresholds[0]}, low={best_thresholds[1]}")
    print(f"Regime weights: {json.dumps({k: [round(x,2) for x in v] for k,v in regime_weights.items()})}")

    # Save all results
    results = {
        "static_weights": {"weights": [0.40, 0.35, 0.25], "accuracy": static["accuracy"], "maybe_rate": static["maybe_rate"]},
        "grid_optimized": {"weights": list(best_w), "accuracy": grid_result["accuracy"], "maybe_rate": grid_result["maybe_rate"]},
        "grid_plus_thresholds": {"weights": list(best_w), "thresholds": list(best_thresholds), "accuracy": grid_thresh["accuracy"], "maybe_rate": grid_thresh["maybe_rate"]},
        "adaptive_regime": {"regime_weights": {k: list(v) for k,v in regime_weights.items()}, "accuracy": adaptive_acc, "maybe_rate": adaptive_maybe},
        "majority_vote": {"accuracy": majority_acc},
        "test_set_size": n,
    }

    output_path = Path("experiments/results/trust_weight_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid_search", "regime_analysis", "learn_weights", "thresholds", "full"], required=True)
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "grid_search":
        traces = load_traces(args.traces, "train")
        grid_search(traces)

    elif args.mode == "regime_analysis":
        traces = load_traces(args.traces, "train")
        regime_analysis(traces)

    elif args.mode == "learn_weights":
        traces = load_traces(args.traces, "train")
        learn_adaptive_weights(traces)

    elif args.mode == "thresholds":
        traces = load_traces(args.traces, "train")
        optimize_thresholds(traces)

    elif args.mode == "full":
        run_full_pipeline(args.traces)


if __name__ == "__main__":
    main()
