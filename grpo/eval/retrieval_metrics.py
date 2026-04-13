"""
Retrieval Quality Evaluation

Computes Recall@K, Precision@K, MRR, and Hit Rate by comparing
retrieved passages against PubMedQA ground-truth contexts.

Usage:
    python -m grpo.eval.retrieval_metrics --traces experiments/traces/debate_traces_full.jsonl --pubmedqa_path data/ori_pqal.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_traces(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_pubmedqa_contexts(path):
    """Load PubMedQA ground-truth contexts (the source abstracts)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    contexts = {}
    if isinstance(data, dict):
        for qid, item in data.items():
            if isinstance(item, dict):
                ctx = item.get("CONTEXTS", item.get("context", item.get("contexts", [])))
                if isinstance(ctx, list):
                    contexts[str(qid)] = " ".join(ctx)
                elif isinstance(ctx, str):
                    contexts[str(qid)] = ctx

                if not contexts.get(str(qid)):
                    la = item.get("LONG_ANSWER", item.get("long_answer", ""))
                    if la:
                        contexts[str(qid)] = la
    elif isinstance(data, list):
        for item in data:
            qid = str(item.get("question_id", item.get("pubid", item.get("pmid", ""))))
            ctx = item.get("context", item.get("CONTEXTS", ""))
            if isinstance(ctx, list):
                contexts[qid] = " ".join(ctx)
            elif isinstance(ctx, str):
                contexts[qid] = ctx

    return contexts


def tokenize(text):
    """Simple whitespace + lowercase tokenization."""
    return set(text.lower().split())


def compute_token_overlap(retrieved_text, ground_truth_text, min_overlap=5):
    """Compute token-level overlap between retrieved passage and ground truth."""
    if not retrieved_text or not ground_truth_text:
        return 0.0

    ret_tokens = tokenize(retrieved_text)
    gt_tokens = tokenize(ground_truth_text)

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "this", "that",
                 "it", "be", "have", "has", "had", "do", "does", "did", "will", "would",
                 "can", "could", "may", "might", "shall", "should", "by", "from", "as",
                 "we", "our", "these", "those", "been", "being", "which", "their", "than"}

    ret_meaningful = ret_tokens - stopwords
    gt_meaningful = gt_tokens - stopwords

    if len(gt_meaningful) == 0:
        return 0.0

    overlap = ret_meaningful & gt_meaningful

    if len(overlap) < min_overlap:
        return 0.0

    return len(overlap) / len(gt_meaningful)


def compute_rouge_l(retrieved_text, ground_truth_text):
    """Compute ROUGE-L (longest common subsequence) F1 score."""
    if not retrieved_text or not ground_truth_text:
        return 0.0

    ret_words = retrieved_text.lower().split()[:500]
    gt_words = ground_truth_text.lower().split()[:500]

    if len(ret_words) == 0 or len(gt_words) == 0:
        return 0.0

    m, n = len(ret_words), len(gt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ret_words[i-1] == gt_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def evaluate_retrieval(traces, ground_truth_contexts, relevance_threshold=0.15):
    """Compute retrieval metrics for all traces."""

    k_values = [1, 3, 5]
    metrics = {"n_evaluated": 0, "n_skipped": 0}

    for k in k_values:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
        metrics[f"hit@{k}"] = []

    metrics["mrr"] = []
    metrics["rouge_l_best"] = []
    metrics["overlap_scores"] = []
    per_sample = []

    for trace in traces:
        qid = str(trace.get("question_id", ""))
        gt_context = ground_truth_contexts.get(qid, "")

        if not gt_context:
            question_text = trace.get("question", "")
            for stored_qid, stored_ctx in ground_truth_contexts.items():
                if question_text and question_text[:50].lower() in stored_ctx.lower():
                    gt_context = stored_ctx
                    break

        if not gt_context:
            gt_context = trace.get("context", "")

        if not gt_context:
            metrics["n_skipped"] += 1
            continue

        metrics["n_evaluated"] += 1

        retrieved = trace.get("retrieved_evidence", [])
        if not retrieved:
            for k in k_values:
                metrics[f"recall@{k}"].append(0.0)
                metrics[f"precision@{k}"].append(0.0)
                metrics[f"hit@{k}"].append(0)
            metrics["mrr"].append(0.0)
            metrics["rouge_l_best"].append(0.0)
            continue

        passage_scores = []
        for passage in retrieved:
            overlap = compute_token_overlap(passage, gt_context)
            rouge = compute_rouge_l(passage, gt_context)
            combined = max(overlap, rouge)
            passage_scores.append({
                "overlap": overlap, "rouge_l": rouge,
                "combined": combined, "relevant": combined >= relevance_threshold,
            })

        metrics["overlap_scores"].extend([p["combined"] for p in passage_scores])
        best_rouge = max(p["rouge_l"] for p in passage_scores) if passage_scores else 0
        metrics["rouge_l_best"].append(best_rouge)

        relevant_passages = [p["relevant"] for p in passage_scores]
        total_relevant = sum(relevant_passages)

        for k in k_values:
            top_k_relevant = sum(relevant_passages[:k])
            recall = top_k_relevant / max(total_relevant, 1)
            metrics[f"recall@{k}"].append(recall)
            metrics[f"precision@{k}"].append(top_k_relevant / k)
            metrics[f"hit@{k}"].append(1 if top_k_relevant > 0 else 0)

        first_relevant_rank = None
        for rank, is_rel in enumerate(relevant_passages, 1):
            if is_rel:
                first_relevant_rank = rank
                break
        metrics["mrr"].append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        per_sample.append({
            "question_id": qid, "n_retrieved": len(retrieved),
            "n_relevant": total_relevant, "best_rouge_l": best_rouge,
            "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
            "is_correct": trace.get("is_correct_judge", False),
        })

    # Print results
    print("=" * 65)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 65)
    print(f"Samples evaluated: {metrics['n_evaluated']}")
    print(f"Samples skipped (no ground truth): {metrics['n_skipped']}")
    print(f"Relevance threshold: {relevance_threshold}")

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 38)

    results = {}
    for k in k_values:
        r = np.mean(metrics[f"recall@{k}"]) if metrics[f"recall@{k}"] else 0
        p = np.mean(metrics[f"precision@{k}"]) if metrics[f"precision@{k}"] else 0
        h = np.mean(metrics[f"hit@{k}"]) if metrics[f"hit@{k}"] else 0
        print(f"Recall@{k:<20} {r:>9.3f}")
        print(f"Precision@{k:<18} {p:>9.3f}")
        print(f"Hit Rate@{k:<19} {h:>9.3f}")
        results[f"recall@{k}"] = r
        results[f"precision@{k}"] = p
        results[f"hit@{k}"] = h

    mrr_mean = np.mean(metrics["mrr"]) if metrics["mrr"] else 0
    rouge_mean = np.mean(metrics["rouge_l_best"]) if metrics["rouge_l_best"] else 0
    print(f"{'MRR':<25} {mrr_mean:>9.3f}")
    print(f"{'Best ROUGE-L (mean)':<25} {rouge_mean:>9.3f}")
    results["mrr"] = mrr_mean
    results["rouge_l_best"] = rouge_mean

    # Retrieval quality vs correctness
    print(f"\n{'='*65}")
    print("RETRIEVAL QUALITY vs ANSWER CORRECTNESS")
    print(f"{'='*65}")

    correct_samples = [s for s in per_sample if s["is_correct"]]
    wrong_samples = [s for s in per_sample if not s["is_correct"]]

    if correct_samples and wrong_samples:
        correct_rouge = np.mean([s["best_rouge_l"] for s in correct_samples])
        wrong_rouge = np.mean([s["best_rouge_l"] for s in wrong_samples])
        correct_relevant = np.mean([s["n_relevant"] for s in correct_samples])
        wrong_relevant = np.mean([s["n_relevant"] for s in wrong_samples])
        correct_mrr = np.mean([s["mrr"] for s in correct_samples])
        wrong_mrr = np.mean([s["mrr"] for s in wrong_samples])

        print(f"{'Metric':<30} {'Correct':>10} {'Wrong':>10} {'Gap':>10}")
        print("-" * 62)
        print(f"{'Avg best ROUGE-L':<30} {correct_rouge:>9.3f} {wrong_rouge:>9.3f} {correct_rouge-wrong_rouge:>+9.3f}")
        print(f"{'Avg relevant passages':<30} {correct_relevant:>9.2f} {wrong_relevant:>9.2f} {correct_relevant-wrong_relevant:>+9.2f}")
        print(f"{'Avg MRR':<30} {correct_mrr:>9.3f} {wrong_mrr:>9.3f} {correct_mrr-wrong_mrr:>+9.3f}")

        results["correct_rouge_l"] = correct_rouge
        results["wrong_rouge_l"] = wrong_rouge

    return results, per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--pubmedqa_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    traces = load_traces(args.traces)

    if args.pubmedqa_path:
        gt_contexts = load_pubmedqa_contexts(args.pubmedqa_path)
        print(f"Loaded {len(gt_contexts)} ground-truth contexts")
    else:
        gt_contexts = {}
        for t in traces:
            qid = str(t.get("question_id", ""))
            ctx = t.get("context", "")
            if qid and ctx:
                gt_contexts[qid] = ctx
        print(f"Using {len(gt_contexts)} contexts from traces")

    results, per_sample = evaluate_retrieval(traces, gt_contexts, args.threshold)

    output_path = Path("experiments/results/retrieval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
