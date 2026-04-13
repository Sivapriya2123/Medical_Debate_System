"""
Generation Quality Evaluation

Measures the quality of doctor and judge reasoning beyond just accuracy.
Three dimensions:
1. Faithfulness: Is the reasoning grounded in retrieved evidence?
2. Relevancy: Does the reasoning address the actual question?
3. Citation accuracy: Do evidence citations point to real, relevant passages?

Usage:
    python -m grpo.eval.generation_metrics --traces experiments/traces/debate_traces_full.jsonl
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_traces(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return text.lower().split()


def compute_faithfulness(response_text, evidence_list, min_phrase_len=4):
    """Faithfulness: fraction of response n-grams found in evidence."""
    if not response_text or not evidence_list:
        return 0.0

    response_tokens = tokenize(response_text)
    evidence_text = " ".join(evidence_list).lower()
    evidence_tokens = tokenize(evidence_text)

    if len(response_tokens) < min_phrase_len:
        return 0.0

    evidence_ngrams = set()
    for n in [4, 5, 6]:
        for i in range(len(evidence_tokens) - n + 1):
            ngram = " ".join(evidence_tokens[i:i+n])
            evidence_ngrams.add(ngram)

    grounded = 0
    total = 0
    for n in [4, 5, 6]:
        for i in range(len(response_tokens) - n + 1):
            ngram = " ".join(response_tokens[i:i+n])
            total += 1
            if ngram in evidence_ngrams:
                grounded += 1

    if total == 0:
        return 0.0
    return grounded / total


def compute_relevancy(response_text, question_text):
    """Relevancy: token overlap between question key terms and response."""
    if not response_text or not question_text:
        return 0.0

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "this", "that",
                 "it", "be", "have", "has", "had", "do", "does", "did", "will", "would",
                 "can", "could", "may", "might", "shall", "should", "by", "from", "as",
                 "we", "our", "these", "those", "been", "being", "which", "their", "than",
                 "what", "how", "where", "when", "who", "why", "there"}

    question_terms = set(tokenize(question_text)) - stopwords
    response_terms = set(tokenize(response_text)) - stopwords

    if len(question_terms) == 0:
        return 0.0

    overlap = question_terms & response_terms
    return len(overlap) / len(question_terms)


def compute_citation_accuracy(response_text, evidence_list):
    """Citation accuracy: do cited evidence indices exist?"""
    if not response_text or not evidence_list:
        return 0, 0, 0.0

    citation_patterns = re.findall(r'\[(\d+)\]|\((\d+)\)|evidence\s+(\d+)', response_text.lower())

    cited_indices = set()
    for groups in citation_patterns:
        for g in groups:
            if g:
                try:
                    idx = int(g) - 1
                    cited_indices.add(idx)
                except ValueError:
                    pass

    if not cited_indices:
        return 0, 0, 0.0

    n_citations = len(cited_indices)
    n_valid = sum(1 for idx in cited_indices if 0 <= idx < len(evidence_list))
    accuracy = n_valid / n_citations if n_citations > 0 else 0.0
    return n_citations, n_valid, accuracy


def evaluate_generation(traces):
    """Full generation quality evaluation."""

    print("=" * 65)
    print("GENERATION QUALITY EVALUATION")
    print("=" * 65)

    faithfulness_scores = {"doctor_a": [], "doctor_b": [], "judge": []}
    relevancy_scores = {"doctor_a": [], "doctor_b": [], "judge": []}
    citation_stats = {"doctor_a": [], "doctor_b": [], "judge": []}
    faithful_correct = []
    faithful_wrong = []

    for trace in traces:
        evidence = trace.get("retrieved_evidence", [])
        question = trace.get("question", "")
        is_correct = trace.get("is_correct_judge", False)

        r2 = trace.get("round_2", {})
        doc_a_resp = r2.get("doctor_a_full_response", r2.get("doctor_a_reasoning", ""))
        doc_b_resp = r2.get("doctor_b_full_response", r2.get("doctor_b_reasoning", ""))
        judge_resp = trace.get("judge_full_response", trace.get("judge_reasoning", ""))

        for name, resp in [("doctor_a", doc_a_resp), ("doctor_b", doc_b_resp), ("judge", judge_resp)]:
            faithfulness_scores[name].append(compute_faithfulness(resp, evidence))

        judge_faith = compute_faithfulness(judge_resp, evidence)
        if is_correct:
            faithful_correct.append(judge_faith)
        else:
            faithful_wrong.append(judge_faith)

        for name, resp in [("doctor_a", doc_a_resp), ("doctor_b", doc_b_resp), ("judge", judge_resp)]:
            relevancy_scores[name].append(compute_relevancy(resp, question))

        for name, resp in [("doctor_a", doc_a_resp), ("doctor_b", doc_b_resp), ("judge", judge_resp)]:
            n_cite, n_valid, acc = compute_citation_accuracy(resp, evidence)
            citation_stats[name].append({"n_citations": n_cite, "n_valid": n_valid, "accuracy": acc})

    n = len(traces)

    print(f"\nSamples evaluated: {n}")
    print(f"\n--- Faithfulness (evidence grounding) ---")
    print(f"{'Agent':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    results = {}
    for name in ["doctor_a", "doctor_b", "judge"]:
        vals = faithfulness_scores[name]
        if vals:
            print(f"{name:<15} {np.mean(vals):>7.3f} {np.std(vals):>7.3f} {np.min(vals):>7.3f} {np.max(vals):>7.3f}")
            results[f"faithfulness_{name}"] = float(np.mean(vals))

    print(f"\n--- Answer Relevancy (question addressing) ---")
    print(f"{'Agent':<15} {'Mean':>8} {'Std':>8}")
    print("-" * 35)
    for name in ["doctor_a", "doctor_b", "judge"]:
        vals = relevancy_scores[name]
        if vals:
            print(f"{name:<15} {np.mean(vals):>7.3f} {np.std(vals):>7.3f}")
            results[f"relevancy_{name}"] = float(np.mean(vals))

    print(f"\n--- Citation Accuracy ---")
    print(f"{'Agent':<15} {'Avg Citations':>15} {'Avg Valid':>12} {'Accuracy':>10}")
    print("-" * 55)
    for name in ["doctor_a", "doctor_b", "judge"]:
        stats = citation_stats[name]
        if stats:
            avg_cite = np.mean([s["n_citations"] for s in stats])
            avg_valid = np.mean([s["n_valid"] for s in stats])
            avg_acc = np.mean([s["accuracy"] for s in stats if s["n_citations"] > 0]) if any(s["n_citations"] > 0 for s in stats) else 0
            samples_with_citations = sum(1 for s in stats if s["n_citations"] > 0)
            print(f"{name:<15} {avg_cite:>14.2f} {avg_valid:>11.2f} {avg_acc:>9.3f}")
            print(f"{'':>15} ({samples_with_citations}/{n} samples cite evidence)")
            results[f"citation_accuracy_{name}"] = float(avg_acc)
            results[f"citation_rate_{name}"] = samples_with_citations / n

    print(f"\n--- Faithfulness vs Correctness (Judge) ---")
    if faithful_correct and faithful_wrong:
        print(f"{'Outcome':<20} {'Mean Faithfulness':>20} {'N':>6}")
        print("-" * 48)
        print(f"{'Correct answers':<20} {np.mean(faithful_correct):>19.3f} {len(faithful_correct):>6}")
        print(f"{'Wrong answers':<20} {np.mean(faithful_wrong):>19.3f} {len(faithful_wrong):>6}")
        print(f"{'Gap':<20} {np.mean(faithful_correct)-np.mean(faithful_wrong):>+19.3f}")
        results["faithfulness_correct"] = float(np.mean(faithful_correct))
        results["faithfulness_wrong"] = float(np.mean(faithful_wrong))

    print(f"\n{'='*65}")
    print("GENERATION QUALITY SUMMARY TABLE (for paper)")
    print(f"{'='*65}")
    print(f"\n{'Metric':<30} {'Doctor A':>10} {'Doctor B':>10} {'Judge':>10}")
    print("-" * 62)
    print(f"{'Faithfulness':<30} {np.mean(faithfulness_scores['doctor_a']):>9.3f} {np.mean(faithfulness_scores['doctor_b']):>9.3f} {np.mean(faithfulness_scores['judge']):>9.3f}")
    print(f"{'Relevancy':<30} {np.mean(relevancy_scores['doctor_a']):>9.3f} {np.mean(relevancy_scores['doctor_b']):>9.3f} {np.mean(relevancy_scores['judge']):>9.3f}")

    cite_accs = {}
    for name in ["doctor_a", "doctor_b", "judge"]:
        stats_with_cite = [s for s in citation_stats[name] if s["n_citations"] > 0]
        cite_accs[name] = np.mean([s["accuracy"] for s in stats_with_cite]) if stats_with_cite else 0
    print(f"{'Citation Accuracy':<30} {cite_accs['doctor_a']:>9.3f} {cite_accs['doctor_b']:>9.3f} {cite_accs['judge']:>9.3f}")

    cite_rates = {}
    for name in ["doctor_a", "doctor_b", "judge"]:
        cite_rates[name] = sum(1 for s in citation_stats[name] if s["n_citations"] > 0) / n
    print(f"{'Citation Rate':<30} {cite_rates['doctor_a']:>9.1%} {cite_rates['doctor_b']:>9.1%} {cite_rates['judge']:>9.1%}")

    output_path = Path("experiments/results/generation_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    args = parser.parse_args()

    traces = load_traces(args.traces)
    evaluate_generation(traces)


if __name__ == "__main__":
    main()
