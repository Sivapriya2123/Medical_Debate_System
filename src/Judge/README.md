# Judge Module — Implementation Plan

This document describes everything that needs to be built inside `src/Judge/` to complete the
**Trust-Aware Multi-Agent Medical Debate** system. Person 2's debate system is fully implemented
and produces a `DebateTranscript` object. This module consumes that transcript and must produce
a **final answer**, a **trust score**, **experiment results**, and **evaluation metrics**.

---

## What Has Already Been Built (Context)

The upstream components you will consume:

| Component | Location | Status |
|-----------|----------|--------|
| Retrieval pipeline (BM25 + dense + reranker) | `src/retrieval/` | Complete |
| Debate graph (LangGraph, 2 doctors, N rounds) | `src/debate/graph.py` | Complete |
| Data models (`DebateTranscript`, `AgentMessage`, `EvidenceItem`) | `src/debate/models.py` | Complete |
| End-to-end entry point | `main.py` | Partial (needs Judge wired in) |

### Key Input: `DebateTranscript`

Every function you write receives a `DebateTranscript` (defined in `src/debate/models.py`):

```python
class DebateTranscript(BaseModel):
    question: str
    evidence: List[EvidenceItem]          # retrieved evidence with relevance scores
    messages: List[AgentMessage]          # full debate history
    num_rounds: int
    doctor_a_final_position: Literal["yes", "no", "maybe"]
    doctor_b_final_position: Literal["yes", "no", "maybe"]
    doctor_a_final_confidence: float      # 0.0 – 1.0
    doctor_b_final_confidence: float      # 0.0 – 1.0
    metadata: Dict[str, Any]              # includes ground_truth label if set upstream
```

Each `AgentMessage` carries:

```python
class AgentMessage(BaseModel):
    agent: Literal["doctor_a", "doctor_b"]
    round: int
    position: Literal["yes", "no", "maybe"]
    reasoning: str
    evidence_cited: List[str]   # e.g. ["evidence_1", "evidence_3"]
    confidence: float
```

---

## Planned File Structure

```
src/Judge/
├── README.md               ← this file
├── __init__.py             ← public exports
├── models.py               ← output data models (JudgeVerdict, TrustScore, ExperimentResult)
├── judge_agent.py          ← LLM-based judge that produces the final answer
├── trust.py                ← trust score calculation (three sub-signals)
├── experiments.py          ← baseline vs. proposed system runner
├── evaluation.py           ← accuracy, trust-correlation, error-rate metrics
└── pipeline.py             ← single function that wires everything together
```

---

## Component 1 — Output Data Models (`models.py`)

Define Pydantic models so every downstream step has a typed contract.

```python
class TrustScore(BaseModel):
    overall: float                  # 0.0 – 1.0, weighted combination
    agent_agreement: float          # 1.0 when agents agree, 0.0 when fully opposed
    reasoning_consistency: float    # semantic similarity of reasoning chains across rounds
    confidence_stability: float     # how stable each agent's confidence was across rounds
    breakdown: Dict[str, Any]       # raw signals for debugging / ablation

class JudgeVerdict(BaseModel):
    question: str
    final_answer: Literal["yes", "no", "maybe"]
    explanation: str                # judge's synthesised justification
    trust: TrustScore
    debate_summary: str             # 2-3 sentence human-readable summary of the debate

class ExperimentResult(BaseModel):
    system_name: str                # e.g. "baseline_rag", "debate_no_trust", "full_system"
    question: str
    ground_truth: str
    predicted_answer: str
    correct: bool
    trust_score: Optional[float]    # None for baselines that don't compute trust
    metadata: Dict[str, Any]

class EvaluationReport(BaseModel):
    system_name: str
    num_questions: int
    accuracy: float
    error_rate: float               # fraction of "maybe" or wrong answers
    trust_correlation: Optional[float]  # Pearson r between trust_score and correct
    per_question_results: List[ExperimentResult]
```

---

## Component 2 — Judge Agent (`judge_agent.py`)

### What it does

An LLM (same model used by the debate agents) reads the full `DebateTranscript` and produces a
synthesised final answer. The judge is **not** just majority-vote — it reads and weighs the
reasoning, evidence citations, and confidence changes to reach a justified conclusion.

### System prompt (sketch)

```
You are a senior medical expert serving as an impartial judge in a structured debate between two
specialist doctors. You will be given:
  1. The original clinical question.
  2. The retrieved evidence passages.
  3. The full debate transcript, including each doctor's position, confidence, and reasoning.

Your task is to produce:
  FINAL_ANSWER: <yes | no | maybe>
  EXPLANATION: <2-4 sentences citing specific evidence and doctor arguments that drove your decision>

Guidelines:
- If both doctors agree with high confidence, adopt their shared position.
- If they disagree, weigh the quality of cited evidence and the internal consistency of each
  doctor's reasoning across rounds.
- Return "maybe" only when evidence is genuinely insufficient or contradictory after debate.
```

### Implementation steps

1. **`format_transcript_for_judge(transcript: DebateTranscript) -> str`**
   Build a prompt string that includes:
   - The original question
   - Numbered evidence passages (re-use `format_evidence_for_prompt` from `src/debate/prompts.py`)
   - Each round's messages formatted as `[Round N | Doctor A | Position: yes | Confidence: 0.82]\n<reasoning>`

2. **`run_judge(transcript: DebateTranscript, llm) -> JudgeVerdict`**
   - Call `format_transcript_for_judge`
   - Invoke LLM with judge system prompt + formatted transcript
   - Parse `FINAL_ANSWER` and `EXPLANATION` from the response (same regex pattern used in `src/debate/prompts.py`)
   - Compute `TrustScore` (see Component 3)
   - Return a populated `JudgeVerdict`

---

## Component 3 — Trust Score Calculation (`trust.py`)

Trust is a single composite score built from three independently computable sub-signals.

### Sub-signal 1: Agent Agreement (`agent_agreement`)

Measures whether the two doctors converged to the same conclusion.

```
positions = {doctor_a_final_position, doctor_b_final_position}

if len(positions) == 1:           # full agreement
    agreement = 1.0
elif "maybe" in positions:        # partial agreement (one is uncertain)
    agreement = 0.5
else:                             # direct contradiction (yes vs. no)
    agreement = 0.0
```

Weight in overall score: **0.4**

### Sub-signal 2: Reasoning Consistency (`reasoning_consistency`)

Measures whether each doctor's reasoning was self-consistent across rounds (they didn't
contradict themselves mid-debate).

**Implementation approach — embedding cosine similarity:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # small, fast, already in requirements

def reasoning_consistency(messages: List[AgentMessage]) -> float:
    # Separate messages by agent
    for agent in ["doctor_a", "doctor_b"]:
        turns = [m.reasoning for m in messages if m.agent == agent]
        if len(turns) < 2:
            continue
        embeddings = model.encode(turns)
        # Average pairwise cosine similarity across all consecutive pairs
        ...
    # Return mean consistency across both agents, normalised to [0, 1]
```

Weight in overall score: **0.35**

### Sub-signal 3: Confidence Stability (`confidence_stability`)

Measures whether each doctor's confidence moved in a meaningful direction rather than
fluctuating randomly (erratic changes signal uncertainty about the evidence).

```
For each agent, compute the standard deviation of confidence values across rounds.
stability = 1.0 - mean(std_dev_a, std_dev_b)    # low std dev → high stability
```

Weight in overall score: **0.25**

### Composite formula

```python
def compute_trust_score(transcript: DebateTranscript) -> TrustScore:
    a = agent_agreement(transcript)
    r = reasoning_consistency(transcript.messages)
    s = confidence_stability(transcript.messages)

    overall = 0.4 * a + 0.35 * r + 0.25 * s

    return TrustScore(
        overall=round(overall, 4),
        agent_agreement=a,
        reasoning_consistency=r,
        confidence_stability=s,
        breakdown={"weights": {"agreement": 0.4, "consistency": 0.35, "stability": 0.25}}
    )
```

---

## Component 4 — Experiments (`experiments.py`)

Three systems are compared on the same set of questions drawn from PubMedQA.

### System definitions

| System Name | Description |
|-------------|-------------|
| `baseline_rag` | Standard RAG: retrieve top-3 evidence, single LLM call, direct answer. No debate, no trust. |
| `debate_no_trust` | Full debate pipeline, but judge uses simple majority-vote (ignore trust). |
| `full_system` | Full debate pipeline + trust-aware judge (this project's proposed approach). |

### `run_single_experiment(question, evidence, ground_truth, system_name, llm) -> ExperimentResult`

- For `baseline_rag`: format evidence + question as a single prompt, call LLM, parse answer.
- For `debate_no_trust`: run `run_debate()`, apply majority-vote to final positions, skip trust.
- For `full_system`: run `run_debate()`, call `run_judge()`, use `JudgeVerdict.final_answer`.

### `run_all_experiments(dataset, llm, num_questions=50) -> Dict[str, List[ExperimentResult]]`

Iterates over `num_questions` samples from PubMedQA, runs all three systems on each question,
collects results keyed by system name.

```python
def run_all_experiments(dataset, llm, num_questions=50):
    results = {"baseline_rag": [], "debate_no_trust": [], "full_system": []}
    for item in dataset[:num_questions]:
        question = item["question"]
        ground_truth = item["final_decision"]   # "yes" / "no" / "maybe"
        evidence = retrieve_evidence(question)  # calls src/retrieval pipeline
        for system in results:
            result = run_single_experiment(question, evidence, ground_truth, system, llm)
            results[system].append(result)
    return results
```

---

## Component 5 — Evaluation Metrics (`evaluation.py`)

### Metrics to compute

| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | `correct / total` | Per system |
| **Error rate** | `(maybe_answers + wrong) / total` | Measures abstention + mistakes |
| **Trust–accuracy correlation** | Pearson r(`trust_score`, `correct`) | Only for systems that produce trust scores; positive r means trust predicts correctness |

### `compute_evaluation_report(results: List[ExperimentResult], system_name: str) -> EvaluationReport`

```python
from scipy.stats import pearsonr

def compute_evaluation_report(results, system_name):
    correct = [r.correct for r in results]
    accuracy = sum(correct) / len(correct)
    error_rate = sum(1 for r in results if not r.correct or r.predicted_answer == "maybe") / len(results)

    trust_scores = [r.trust_score for r in results if r.trust_score is not None]
    if trust_scores:
        corr, _ = pearsonr(trust_scores, [r.correct for r in results if r.trust_score is not None])
    else:
        corr = None

    return EvaluationReport(
        system_name=system_name,
        num_questions=len(results),
        accuracy=round(accuracy, 4),
        error_rate=round(error_rate, 4),
        trust_correlation=round(corr, 4) if corr else None,
        per_question_results=results,
    )
```

### `print_comparison_table(reports: List[EvaluationReport])`

Pretty-print a markdown table comparing all three systems side-by-side so results can be
copy-pasted into the paper.

---

## Component 6 — Pipeline Entrypoint (`pipeline.py`)

A single function that wires all components together and returns all four required outputs.

```python
def run_judge_pipeline(
    question: str,
    evidence: List[dict],
    ground_truth: Optional[str] = None,
    llm=None,
) -> Tuple[JudgeVerdict, TrustScore, ExperimentResult]:
    """
    Full pipeline for a single question:
      1. Run multi-agent debate
      2. Compute trust score
      3. Run judge agent to produce final answer
      4. Package as ExperimentResult (with ground_truth if provided)

    Returns: (verdict, trust_score, experiment_result)
    """
```

`main.py` will call `run_judge_pipeline` after retrieval and pass results to the evaluation layer.

---

## Integration with `main.py`

After implementing the above, update `main.py` to add:

```python
# 3. Judge + Trust (Person 3)
from src.Judge.pipeline import run_judge_pipeline
from src.Judge.evaluation import compute_evaluation_report

verdict, trust, result = run_judge_pipeline(
    question=question,
    evidence=evidence,
    ground_truth=sample.get("final_decision"),
    llm=llm,
)
print(f"Final Answer : {verdict.final_answer}")
print(f"Explanation  : {verdict.explanation}")
print(f"Trust Score  : {trust.overall:.3f}  (agreement={trust.agent_agreement:.2f}, "
      f"consistency={trust.reasoning_consistency:.2f}, stability={trust.confidence_stability:.2f})")
```

For the full experiment run (50 questions), call `run_all_experiments` then
`compute_evaluation_report` for each system and print the comparison table.

---

## Expected Outputs

### 1. Final Answer (`JudgeVerdict`)
```
Final Answer : yes
Explanation  : Both doctors agreed that metformin reduces HbA1c (Evidence 1, 3).
               Doctor A noted RCT quality; Doctor B highlighted real-world generalisability.
Trust Score  : 0.847
```

### 2. Trust Score breakdown
```
overall              : 0.847
  agent_agreement    : 1.000   (both said "yes")
  reasoning_consist  : 0.821   (high embedding similarity across rounds)
  confidence_stable  : 0.712   (minor dip in round 2 for Doctor B)
```

### 3. Experiment Results table
```
| System            | Accuracy | Error Rate | Trust–Acc Corr |
|-------------------|----------|------------|----------------|
| baseline_rag      |  0.612   |   0.388    |      N/A       |
| debate_no_trust   |  0.674   |   0.326    |      N/A       |
| full_system       |  0.731   |   0.269    |     +0.43      |
```

### 4. Evaluation artefacts
- Per-question CSV: `outputs/experiment_results.csv`
- Full JSON dump: `outputs/experiment_results.json`
- Summary printed to stdout for quick inspection

---

## Dependencies to Add

The following packages are needed and should be added to `requirements.txt`:

```
sentence-transformers>=2.2.0   # reasoning_consistency embeddings
scipy>=1.11.0                  # pearsonr for trust-correlation metric
pandas>=2.0.0                  # CSV export of experiment results
```

`sentence-transformers` will download `all-MiniLM-L6-v2` (~80 MB) on first use.

---

## Implementation Order

1. `models.py` — define all output types first (no dependencies)
2. `trust.py` — pure computation, easy to unit-test without an LLM
3. `judge_agent.py` — depends on models + trust
4. `pipeline.py` — wires debate → trust → judge for a single question
5. `experiments.py` — runs pipeline across the dataset
6. `evaluation.py` — post-processes experiment results into metrics
7. Update `main.py` — wire everything end-to-end
