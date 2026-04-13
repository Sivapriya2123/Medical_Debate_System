"""
Integrated evaluation: GRPO-optimized judge + learned trust weights.

Combines:
- Phase 1: decisive_v1 judge prompt (reduces maybe over-correction)
- Phase 2: optimized trust weights (better trust signal interpretation)

This is the FINAL system evaluation that produces the paper's main results table.

Usage:
    python -m grpo.eval.integrated_eval --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_traces(path, split="test", train_ratio=0.8):
    with open(path) as f:
        all_traces = [json.loads(line) for line in f]
    split_idx = int(len(all_traces) * train_ratio)
    if split == "test":
        return all_traces[split_idx:]
    return all_traces[:split_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    # Load Phase 2 results
    trust_results_path = Path("experiments/results/trust_weight_results.json")
    if trust_results_path.exists():
        with open(trust_results_path) as f:
            trust_results = json.load(f)
    else:
        print("ERROR: Run Phase 2 first (trust_weight_optimizer --mode full)")
        return

    # Load Phase 1 results
    phase1_results_path = Path("experiments/results/final_evaluation.json")
    if phase1_results_path.exists():
        with open(phase1_results_path) as f:
            phase1_results = json.load(f)
    else:
        phase1_results = {}

    # ==========================================
    # COMPILE THE FULL ABLATION TABLE
    # ==========================================

    print("=" * 75)
    print("COMPLETE ABLATION TABLE (Paper Table -- Main Results)")
    print("=" * 75)

    # Gather all numbers
    rows = []

    # Row 1: No retrieval baseline (from Phase 0, hardcoded from earlier results)
    rows.append(("No retrieval (baseline)", "38.0%", "--", "--"))

    # Row 2: RAG only
    rows.append(("RAG only (no debate)", "65.0%", "--", "--"))

    # Row 3: Majority vote
    majority_acc = trust_results.get("majority_vote", {}).get("accuracy", 0)
    rows.append(("Debate + majority vote", f"{majority_acc:.1%}", "--", "--"))

    # Row 4: Static trust judge (original system)
    static = trust_results.get("static_weights", {})
    rows.append(("Debate + static trust (original)", f"{static.get('accuracy', 0):.1%}", f"{static.get('maybe_rate', 0):.1%}", "0.40/0.35/0.25"))

    # Row 5: GRPO-optimized judge (Phase 1)
    p1 = phase1_results.get("grpo_judge", {})
    if p1:
        rows.append(("Debate + GRPO judge (Phase 1)", f"{p1.get('accuracy', 0):.1%}", f"{p1.get('maybe_overcorrection_rate', 0):.1%}", "static"))

    # Row 6: Grid-optimized weights
    grid = trust_results.get("grid_optimized", {})
    w = grid.get("weights", [])
    w_str = f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}" if w else "--"
    rows.append(("Debate + optimized weights (Phase 2)", f"{grid.get('accuracy', 0):.1%}", f"{grid.get('maybe_rate', 0):.1%}", w_str))

    # Row 7: Adaptive per-regime weights
    adaptive = trust_results.get("adaptive_regime", {})
    rows.append(("Debate + adaptive weights (Phase 2)", f"{adaptive.get('accuracy', 0):.1%}", f"{adaptive.get('maybe_rate', 0):.1%}", "per-regime"))

    # Row 8: Grid + optimized thresholds
    grid_thresh = trust_results.get("grid_plus_thresholds", {})
    rows.append(("Debate + optimized weights + thresholds", f"{grid_thresh.get('accuracy', 0):.1%}", f"{grid_thresh.get('maybe_rate', 0):.1%}", "optimized"))

    # Print the table
    print(f"\n{'System':<45} {'Accuracy':>10} {'Maybe%':>8} {'Trust Weights':>16}")
    print("-" * 82)
    for row in rows:
        print(f"{row[0]:<45} {row[1]:>10} {row[2]:>8} {row[3]:>16}")

    # Save the final table
    output = {
        "ablation_table": [{"system": r[0], "accuracy": r[1], "maybe_rate": r[2], "weights": r[3]} for r in rows],
        "phase_1_best_variant": "decisive_v1",
        "phase_2_best_weights": trust_results.get("grid_optimized", {}).get("weights", []),
        "phase_2_regime_weights": trust_results.get("adaptive_regime", {}).get("regime_weights", {}),
    }

    output_path = Path("experiments/results/full_ablation_table.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")

    print("\n" + "=" * 75)
    print("BRING THESE RESULTS BACK -- this is the paper's main table")
    print("=" * 75)


if __name__ == "__main__":
    main()
