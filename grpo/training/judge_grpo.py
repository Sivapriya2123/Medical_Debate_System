"""
Phase 1: Reward-Guided Judge Prompt Optimization

This is the core GRPO-inspired training loop for Option B (prompt optimization).
For each prompt variant, we:
1. Re-run the judge on all training traces using that variant's prompt
2. Score each response with the reward function
3. Rank variants by total reward
4. In round 2, generate evolved variants from the winners

Usage:
    # Score all initial variants
    python -m grpo.training.judge_grpo --mode score_variants --traces experiments/traces/debate_traces_full.jsonl

    # Run the full optimization loop
    python -m grpo.training.judge_grpo --mode full --traces experiments/traces/debate_traces_full.jsonl

    # Evaluate the best variant on the held-out test set
    python -m grpo.training.judge_grpo --mode evaluate --traces experiments/traces/debate_traces_full.jsonl --variant decisive_v1
"""

import json
import os
import argparse
import time
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()

# Fix SSL cert path for conda environments
if os.environ.get("SSL_CERT_FILE") and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ.pop("SSL_CERT_FILE", None)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# LLM call — uses same API client as existing judge agent
# ============================================================

_llm_instance = None


def _get_llm():
    """Lazy-load the LLM instance (same config as src/debate/agents.py create_llm)."""
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        _llm_instance = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.7,
            api_key=api_key,
            base_url=base_url,
        )
    return _llm_instance


def call_judge_llm(system_prompt: str, user_message: str, max_retries: int = 3) -> str:
    """
    Call GPT-4o-mini with a given system prompt and user message.
    Returns the full response text. Retries on connection errors.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            if attempt < max_retries - 1 and "onnect" in str(e):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            raise


# ============================================================
# Answer extraction — handles both current and variant formats
# ============================================================

def extract_prediction(response: str) -> str:
    """
    Extract yes/no/maybe prediction from judge response.
    Handles both the current format (FINAL_ANSWER: yes) and
    variant formats (Final Answer: yes).
    """
    response_lower = response.lower().strip()

    # Pattern 1: "FINAL_ANSWER: yes/no/maybe" (current judge format)
    match = re.search(r"final_answer\s*[:=]\s*(yes|no|maybe)", response_lower)
    if match:
        return match.group(1)

    # Pattern 2: "Final Answer: yes/no/maybe" (variant format)
    match = re.search(r"final\s*answer\s*[:=]\s*(yes|no|maybe)", response_lower)
    if match:
        return match.group(1)

    # Pattern 3: "Answer: yes/no/maybe"
    match = re.search(r"answer\s*[:=]\s*(yes|no|maybe)", response_lower)
    if match:
        return match.group(1)

    # Pattern 4: Last occurrence of yes/no/maybe in the response
    matches = re.findall(r"\b(yes|no|maybe)\b", response_lower)
    if matches:
        return matches[-1]

    return "maybe"  # default fallback


# ============================================================
# Judge input builder — matches existing format_transcript_for_judge
# ============================================================

def build_judge_input(trace: dict) -> str:
    """
    Build the user message for the judge from a debate trace.
    Matches the format from src/Judge/judge_agent.py format_transcript_for_judge.
    """
    # If the trace already has the exact judge input, use it
    if trace.get("judge_input"):
        return trace["judge_input"]

    # Otherwise reconstruct from trace fields
    parts = []

    parts.append(f"MEDICAL QUESTION: {trace['question']}")
    parts.append("")

    # Retrieved evidence
    evidence = trace.get("retrieved_evidence", [])
    if evidence:
        parts.append("RETRIEVED EVIDENCE:")
        for i, chunk in enumerate(evidence, 1):
            parts.append(f"[Evidence {i}] (relevance score: 0.500)")
            parts.append(chunk)
            parts.append("")

    # Debate transcript
    parts.append("DEBATE TRANSCRIPT:")
    parts.append("-" * 40)

    for round_key in ["round_1", "round_2"]:
        rd = trace.get(round_key, {})
        round_num = 1 if round_key == "round_1" else 2

        for doctor, label in [("doctor_a", "Doctor A"), ("doctor_b", "Doctor B")]:
            position = rd.get(f"{doctor}_answer", "maybe")
            confidence = rd.get(f"{doctor}_confidence", 0.5)
            reasoning = rd.get(f"{doctor}_reasoning", "")
            parts.append(
                f"[Round {round_num} | {label} | "
                f"Position: {position} | Confidence: {confidence:.2f}]"
            )
            parts.append(f"Reasoning: {reasoning}")
            parts.append("-" * 40)

    # Final positions
    r2 = trace.get("round_2", {})
    parts.append("")
    parts.append("FINAL POSITIONS:")
    parts.append(
        f"  Doctor A: {r2.get('doctor_a_answer', 'maybe')} "
        f"(confidence: {r2.get('doctor_a_confidence', 0.5):.2f})"
    )
    parts.append(
        f"  Doctor B: {r2.get('doctor_b_answer', 'maybe')} "
        f"(confidence: {r2.get('doctor_b_confidence', 0.5):.2f})"
    )

    # Trust signals
    ts = trace.get("trust_signals", {})
    parts.append(
        f"\n\nTRUST SCORE: {ts.get('composite_trust_score', 0.0):.3f}\n"
        f"  Agent Agreement: {ts.get('agreement_score', 0.0):.2f}\n"
        f"  Reasoning Consistency: {ts.get('embedding_similarity', 0.0):.2f}\n"
        f"  Confidence Stability: {ts.get('confidence_stability', 0.0):.2f}\n"
    )

    return "\n".join(parts)


# ============================================================
# Reward computation
# ============================================================

def compute_reward(prediction: str, gold_label: str) -> dict:
    """
    Simplified reward for prompt optimization.
    Only correctness + anti-maybe, since format/evidence are always maxed.
    """
    pred = prediction.strip().lower()
    gold = gold_label.strip().lower()

    r_correct = 1.0 if pred == gold else 0.0
    r_anti_maybe = -0.3 if (pred == "maybe" and gold in ("yes", "no")) else 0.0

    return {
        "total": r_correct + r_anti_maybe,
        "correctness": r_correct,
        "anti_maybe": r_anti_maybe,
    }


# ============================================================
# Scoring engine
# ============================================================

def score_variant(variant_name: str, variant_prompt: str, traces: list,
                  max_samples: int = None, verbose: bool = False) -> tuple:
    """
    Score a single prompt variant on the given traces.
    Returns detailed metrics.
    """
    if max_samples:
        traces = traces[:max_samples]

    results = []
    maybe_on_yes_no = 0
    yes_no_total = 0
    predictions = []

    for i, trace in enumerate(traces):
        if verbose and i % 25 == 0:
            print(f"    Scoring {variant_name}: {i}/{len(traces)}...")

        # Build judge input from trace
        user_message = build_judge_input(trace)

        # Call LLM with this variant's prompt
        try:
            response = call_judge_llm(variant_prompt, user_message)
            prediction = extract_prediction(response)
        except Exception as e:
            print(f"    ERROR on sample {i}: {e}")
            prediction = "maybe"  # worst case fallback
            response = ""

        # Compute reward
        reward = compute_reward(prediction, trace["gold_label"])

        # Track maybe over-correction
        if trace["gold_label"] in ("yes", "no"):
            yes_no_total += 1
            if prediction == "maybe":
                maybe_on_yes_no += 1

        predictions.append(prediction)
        results.append({
            "trace_idx": i,
            "gold": trace["gold_label"],
            "prediction": prediction,
            "reward": reward,
        })

    # Aggregate metrics
    total_rewards = [r["reward"]["total"] for r in results]
    accuracies = [r["reward"]["correctness"] for r in results]

    pred_dist = Counter(predictions)

    metrics = {
        "variant_name": variant_name,
        "n_samples": len(traces),
        "accuracy": float(np.mean(accuracies)),
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "maybe_overcorrection_rate": maybe_on_yes_no / yes_no_total if yes_no_total > 0 else 0,
        "maybe_overcorrection_count": maybe_on_yes_no,
        "prediction_distribution": dict(pred_dist),
        "per_gold_accuracy": {},
    }

    # Per-gold-label accuracy
    for gold_label in ["yes", "no", "maybe"]:
        label_results = [r for r in results if r["gold"] == gold_label]
        if label_results:
            label_acc = float(np.mean([r["reward"]["correctness"] for r in label_results]))
            metrics["per_gold_accuracy"][gold_label] = {
                "accuracy": label_acc,
                "count": len(label_results),
            }

    return metrics, results


def print_variant_results(metrics: dict):
    """Pretty-print results for a single variant."""
    print(f"\n  {metrics['variant_name']}:")
    print(f"    Accuracy:        {metrics['accuracy']:.1%}")
    print(f"    Mean reward:     {metrics['mean_reward']:+.4f}")
    print(f"    Maybe on y/n:    {metrics['maybe_overcorrection_count']}/{metrics['n_samples']} ({metrics['maybe_overcorrection_rate']:.1%})")
    print(f"    Pred dist:       {metrics['prediction_distribution']}")
    if metrics['per_gold_accuracy']:
        for label, info in metrics['per_gold_accuracy'].items():
            print(f"    Acc on {label:<6}: {info['accuracy']:.1%} (n={info['count']})")


# ============================================================
# Main workflows
# ============================================================

def run_scoring(traces_path: str, train_split: float = 0.8, max_samples: int = None, only_variant: str = None):
    """Score all prompt variants on the training split."""
    from grpo.training.prompt_variants import get_all_variants

    with open(traces_path) as f:
        all_traces = [json.loads(line) for line in f]

    split_idx = int(len(all_traces) * train_split)
    train_traces = all_traces[:split_idx]
    test_traces = all_traces[split_idx:]

    print(f"Total traces: {len(all_traces)}")
    print(f"Train split:  {len(train_traces)}")
    print(f"Test split:   {len(test_traces)} (held out for final eval)")

    if max_samples:
        train_traces = train_traces[:max_samples]
        print(f"Using first {max_samples} train samples for this run")

    variants = get_all_variants()

    # If re-running a single variant, load existing results and replace that entry
    existing_metrics = {}
    output_path = Path("experiments/results/variant_scoring.json")
    if only_variant and output_path.exists():
        with open(output_path) as f:
            for m in json.load(f):
                existing_metrics[m["variant_name"]] = m

    if only_variant:
        variants = {only_variant: variants[only_variant]}

    all_metrics = []

    # For incremental saving, load any previously saved results
    output_path = Path("experiments/results/variant_scoring.json")
    if not only_variant and output_path.exists():
        with open(output_path) as f:
            for m in json.load(f):
                existing_metrics[m["variant_name"]] = m

    print(f"\nScoring {len(variants)} prompt variant(s)...")
    print("=" * 65)

    n_expected = len(train_traces)
    for name, prompt in variants.items():
        # Skip variants already scored with the same sample count (unless re-running specific one)
        if not only_variant and name in existing_metrics and existing_metrics[name].get("n_samples") == n_expected:
            print(f"\nVariant: {name} — already scored ({n_expected} samples), skipping")
            all_metrics.append(existing_metrics[name])
            continue
        print(f"\nVariant: {name}")
        metrics, _ = score_variant(name, prompt, train_traces, verbose=True)
        print_variant_results(metrics)
        all_metrics.append(metrics)
        existing_metrics[name] = metrics

        # Save incrementally after each variant
        incremental = list(existing_metrics.values())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(incremental, f, indent=2, default=str)
        print(f"  (incremental save: {len(incremental)} variants saved)")

    # Rank by mean reward
    all_metrics.sort(key=lambda m: m["mean_reward"], reverse=True)

    print("\n" + "=" * 65)
    print("RANKING (by mean reward)")
    print("=" * 65)
    for i, m in enumerate(all_metrics):
        marker = " <- CURRENT" if m["variant_name"] == "current" else ""
        marker = " * BEST" if i == 0 else marker
        print(f"  {i+1}. {m['variant_name']:<25} reward={m['mean_reward']:+.4f}  acc={m['accuracy']:.1%}  maybe_rate={m['maybe_overcorrection_rate']:.1%}{marker}")

    # Save results
    output_path = Path("experiments/results/variant_scoring.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Identify top 2 for evolution
    top_2 = all_metrics[:2]
    print(f"\nTop 2 variants for evolution round:")
    for m in top_2:
        print(f"  - {m['variant_name']}: reward={m['mean_reward']:+.4f}, acc={m['accuracy']:.1%}")

    return all_metrics


def run_evolution(traces_path: str, top_variant_names: list = None, train_split: float = 0.8):
    """
    Round 2: Generate evolved variants from top performers and re-score.
    """
    from grpo.training.prompt_variants import get_all_variants

    variants = get_all_variants()

    # Load scoring results to find top variants
    if top_variant_names is None:
        results_path = Path("experiments/results/variant_scoring.json")
        if results_path.exists():
            with open(results_path) as f:
                prev_results = json.load(f)
            top_variant_names = [r["variant_name"] for r in prev_results[:2]]
        else:
            print("ERROR: No previous scoring results found. Run --mode score_variants first.")
            return

    top_prompts = {name: variants[name] for name in top_variant_names if name in variants}

    print(f"Evolving from top variants: {list(top_prompts.keys())}")

    # Generate evolved variants using GPT-4o-mini
    evolution_meta_prompt = f"""You are an expert at writing system prompts for medical AI judges.

I have two judge prompts that performed well in a medical debate evaluation task (PubMedQA).
The judge's job is to evaluate a debate between Doctor A and Doctor B and decide: yes, no, or maybe.

The main problems we're solving:
1. The judge over-predicts "maybe" on questions where the answer is clearly yes or no (11.8% error rate)
2. The judge doesn't use trust signals correctly — high trust should mean BE DECISIVE, not be cautious
3. Doctor B is much stronger than Doctor A (70.4% vs 51.6%), so the judge should weight reasoning quality

TOP PROMPT 1 ({top_variant_names[0]}):
{list(top_prompts.values())[0]}

TOP PROMPT 2 ({top_variant_names[1]}):
{list(top_prompts.values())[1]}

Generate 4 NEW judge prompts that combine the best aspects of both top prompts.
Each new prompt should try a DIFFERENT strategy for reducing "maybe" over-prediction.

IMPORTANT: Each prompt MUST instruct the judge to respond with the format:
FINAL_ANSWER: [yes/no/maybe]
EXPLANATION: [reasoning]
DEBATE_SUMMARY: [summary]

Format your response as:
===VARIANT_1===
[prompt text]
===VARIANT_2===
[prompt text]
===VARIANT_3===
[prompt text]
===VARIANT_4===
[prompt text]"""

    print("Generating evolved variants via LLM...")
    evolution_response = call_judge_llm(
        "You are a prompt engineering expert.",
        evolution_meta_prompt
    )

    # Parse evolved variants
    evolved_variants = {}
    parts = re.split(r"===VARIANT_(\d+)===", evolution_response)
    variant_num = 0
    for i in range(1, len(parts), 2):
        variant_num += 1
        name = f"evolved_v{variant_num}"
        prompt_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if prompt_text:
            evolved_variants[name] = prompt_text

    print(f"Generated {len(evolved_variants)} evolved variants")

    # Score evolved variants
    with open(traces_path) as f:
        all_traces = [json.loads(line) for line in f]

    split_idx = int(len(all_traces) * train_split)
    train_traces = all_traces[:split_idx]

    all_metrics = []

    # Re-score top originals + evolved
    all_to_score = {**top_prompts, **evolved_variants}

    for name, prompt in all_to_score.items():
        print(f"\nScoring: {name}")
        metrics, _ = score_variant(name, prompt, train_traces, verbose=True)
        print_variant_results(metrics)
        all_metrics.append(metrics)

    # Rank
    all_metrics.sort(key=lambda m: m["mean_reward"], reverse=True)

    print("\n" + "=" * 65)
    print("EVOLUTION RANKING")
    print("=" * 65)
    for i, m in enumerate(all_metrics):
        print(f"  {i+1}. {m['variant_name']:<25} reward={m['mean_reward']:+.4f}  acc={m['accuracy']:.1%}  maybe_rate={m['maybe_overcorrection_rate']:.1%}")

    # Save evolved variants and results
    output_path = Path("experiments/results/evolution_scoring.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save the winning prompt
    best = all_metrics[0]
    best_prompt = all_to_score.get(best["variant_name"], "")

    prompt_path = Path("experiments/results/best_judge_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"# Best variant: {best['variant_name']}\n")
        f.write(f"# Accuracy: {best['accuracy']:.1%}\n")
        f.write(f"# Mean reward: {best['mean_reward']:+.4f}\n")
        f.write(f"# Maybe over-correction: {best['maybe_overcorrection_rate']:.1%}\n\n")
        f.write(best_prompt)

    print(f"\nBest prompt saved to {prompt_path}")

    # Save all evolved prompts for reference
    evolved_path = Path("experiments/results/evolved_prompts.json")
    with open(evolved_path, "w", encoding="utf-8") as f:
        json.dump(evolved_variants, f, indent=2)

    return all_metrics, evolved_variants


def run_final_evaluation(traces_path: str, variant_name: str, train_split: float = 0.8):
    """
    Evaluate the best variant on the HELD-OUT test split.
    This is the number that goes in the paper.
    """
    from grpo.training.prompt_variants import get_all_variants

    # Load the variant prompt
    all_variants = get_all_variants()

    # Check evolved variants too
    evolved_path = Path("experiments/results/evolved_prompts.json")
    if evolved_path.exists():
        with open(evolved_path) as f:
            evolved = json.load(f)
        all_variants.update(evolved)

    if variant_name not in all_variants:
        # Try loading from best_judge_prompt.txt
        best_path = Path("experiments/results/best_judge_prompt.txt")
        if best_path.exists():
            with open(best_path) as f:
                lines = f.readlines()
            prompt_text = "".join(line for line in lines if not line.startswith("#")).strip()
            all_variants[variant_name] = prompt_text
        else:
            print(f"ERROR: Variant '{variant_name}' not found")
            return

    variant_prompt = all_variants[variant_name]

    # Load test split
    with open(traces_path) as f:
        all_traces = [json.loads(line) for line in f]

    split_idx = int(len(all_traces) * train_split)
    test_traces = all_traces[split_idx:]

    print(f"Final evaluation on held-out test set: {len(test_traces)} samples")
    print(f"Variant: {variant_name}")
    print("=" * 65)

    # Also score the current prompt on the test set for comparison
    current_prompt = all_variants.get("current", "")

    print("\nScoring CURRENT judge prompt on test set...")
    current_metrics, _ = score_variant("current", current_prompt, test_traces, verbose=True)
    print_variant_results(current_metrics)

    print(f"\nScoring BEST variant ({variant_name}) on test set...")
    best_metrics, best_results = score_variant(variant_name, variant_prompt, test_traces, verbose=True)
    print_variant_results(best_metrics)

    # Comparison
    print("\n" + "=" * 65)
    print("FINAL COMPARISON (Table 2 for paper)")
    print("=" * 65)

    # Load original majority vote accuracy from traces
    majority_correct = sum(1 for t in test_traces if t.get("is_correct_majority"))
    majority_acc = majority_correct / len(test_traces)

    print(f"\n{'System':<35} {'Accuracy':>10} {'Maybe rate':>12} {'Reward':>10}")
    print("-" * 70)
    print(f"{'Majority vote (no trust)':<35} {majority_acc:>9.1%} {'N/A':>12} {'N/A':>10}")
    print(f"{'Current judge (static trust)':<35} {current_metrics['accuracy']:>9.1%} {current_metrics['maybe_overcorrection_rate']:>11.1%} {current_metrics['mean_reward']:>+9.4f}")
    print(f"{'GRPO-optimized judge':<35} {best_metrics['accuracy']:>9.1%} {best_metrics['maybe_overcorrection_rate']:>11.1%} {best_metrics['mean_reward']:>+9.4f}")
    print(f"{'Improvement over current':<35} {best_metrics['accuracy'] - current_metrics['accuracy']:>+9.1%} {best_metrics['maybe_overcorrection_rate'] - current_metrics['maybe_overcorrection_rate']:>+11.1%} {best_metrics['mean_reward'] - current_metrics['mean_reward']:>+9.4f}")
    print(f"{'Improvement over majority':<35} {best_metrics['accuracy'] - majority_acc:>+9.1%}")

    # Save final results
    final_results = {
        "majority_vote": {"accuracy": majority_acc},
        "current_judge": current_metrics,
        "grpo_judge": best_metrics,
        "variant_name": variant_name,
        "test_set_size": len(test_traces),
    }

    output_path = Path("experiments/results/final_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nFinal results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["score_variants", "evolve", "evaluate", "full"], required=True)
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--variant", type=str, default=None, help="Variant name for evaluate mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick testing")
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    if args.mode == "score_variants":
        run_scoring(args.traces, args.train_split, args.max_samples, only_variant=args.variant)

    elif args.mode == "evolve":
        run_evolution(args.traces, train_split=args.train_split)

    elif args.mode == "evaluate":
        if not args.variant:
            print("ERROR: --variant required for evaluate mode")
            return
        run_final_evaluation(args.traces, args.variant, args.train_split)

    elif args.mode == "full":
        # Run the complete optimization pipeline
        print("=" * 65)
        print("PHASE 1: FULL OPTIMIZATION PIPELINE")
        print("=" * 65)

        print("\n>>> Step 1: Scoring initial variants...")
        metrics = run_scoring(args.traces, args.train_split, args.max_samples)

        print("\n>>> Step 2: Evolving from top performers...")
        evo_metrics, _ = run_evolution(args.traces, train_split=args.train_split)

        # Find overall best
        best = evo_metrics[0]
        print(f"\n>>> Step 3: Final evaluation with best variant: {best['variant_name']}")
        run_final_evaluation(args.traces, best["variant_name"], args.train_split)


if __name__ == "__main__":
    main()
