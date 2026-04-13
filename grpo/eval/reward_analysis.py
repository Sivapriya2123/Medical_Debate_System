"""
Retrospective reward analysis on existing debate traces.
Shows what the reward distribution looks like BEFORE GRPO training.
This tells us the "reward landscape" and confirms training is feasible.

Usage:
    python -m grpo.eval.reward_analysis --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from grpo.rewards.reward_functions import compute_total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    with open(args.traces) as f:
        traces = [json.loads(line) for line in f]

    rewards = []
    reward_breakdowns = defaultdict(list)

    for t in traces:
        r = compute_total_reward(
            prediction=t["judge_prediction"],
            gold_label=t["gold_label"],
            response=t.get("judge_full_response", t.get("judge_reasoning", "")),
            retrieved_evidence=t.get("retrieved_evidence", []),
        )
        rewards.append(r["total"])
        for key, val in r.items():
            reward_breakdowns[key].append(val)

    print("=" * 55)
    print("REWARD DISTRIBUTION (before GRPO training)")
    print("=" * 55)
    print(f"Samples: {len(traces)}")
    print()

    for component in ["total", "correctness", "format", "anti_maybe", "evidence"]:
        vals = reward_breakdowns[component]
        print(f"{component:<20} mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  min={np.min(vals):+.4f}  max={np.max(vals):+.4f}")

    print()
    print("Reward histogram (total):")
    hist, bin_edges = np.histogram(rewards, bins=10)
    for i in range(len(hist)):
        bar = "#" * (hist[i] * 40 // max(hist)) if max(hist) > 0 else ""
        print(f"  [{bin_edges[i]:+6.2f}, {bin_edges[i+1]:+6.2f}): {hist[i]:>4} {bar}")

    # Key insight: what percentage of samples get the anti-maybe penalty?
    anti_maybe_hits = sum(1 for v in reward_breakdowns["anti_maybe"] if v < 0)
    print(f"\nSamples receiving anti-maybe penalty: {anti_maybe_hits}/{len(traces)} ({anti_maybe_hits/len(traces):.1%})")
    print("^ This is the fraction GRPO will learn to reduce")

    # Average reward by gold label
    print()
    print("Average total reward by gold label:")
    by_gold = defaultdict(list)
    for t, r in zip(traces, rewards):
        by_gold[t["gold_label"]].append(r)
    for label in ["yes", "no", "maybe"]:
        if by_gold[label]:
            print(f"  gold={label:<6}  mean_reward={np.mean(by_gold[label]):+.4f}  n={len(by_gold[label])}")


if __name__ == "__main__":
    main()
