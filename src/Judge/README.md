# Judge Module (Person 3) — Trust-Aware Medical Debate Adjudication

## Overview

This module is the final stage of the **Trust-Aware Multi-Agent Medical Debate** pipeline. It
receives a `DebateTranscript` from Person 2's debate system, computes a composite trust score,
runs an LLM-based judge agent to produce a final diagnosis, and evaluates the result against
ground truth. It also includes an experiment runner that compares three systems (baseline RAG,
debate-only, and the full proposed system) on PubMedQA questions.

```
[Person 1: Retrieval]  →  [Person 2: Debate]  →  [Person 3: Judge + Trust + Evaluation]
      evidence                transcript              final answer, trust score, metrics
```

### Four Outputs

| Output | Description |
|--------|-------------|
| **Final Answer** | `yes` / `no` / `maybe` — the judge's synthesised verdict |
| **Trust Score** | 0.0–1.0 composite score (agreement + consistency + stability) |
| **Experiment Results** | Per-question results across all three systems |
| **Evaluation Metrics** | Accuracy, error rate, trust-accuracy correlation per system |

---

## File Structure

```
src/Judge/
├── __init__.py          ← public exports (all key functions + models)
├── models.py            ← Pydantic data models: TrustScore, JudgeVerdict, ExperimentResult, EvaluationReport
├── trust.py             ← trust score computation (3 sub-signals → 1 composite score)
├── judge_agent.py       ← LLM judge: system prompt, transcript formatting, response parsing
├── pipeline.py          ← single-question entrypoint: debate → trust → judge → result
├── experiments.py       ← 3-system experiment runner (baseline_rag, debate_no_trust, full_system)
├── evaluation.py        ← metrics computation, comparison table, CSV/JSON export
└── README.md            ← this file
```

---

## Step-by-Step Implementation Details

### Step 1 — Data Models (`models.py`)

Defines four Pydantic models that serve as typed contracts between all components.

**`TrustScore`** — composite trust score with sub-signal breakdown:
```python
class TrustScore(BaseModel):
    overall: float                  # 0.0–1.0, weighted combination of sub-signals
    agent_agreement: float          # 1.0 = both doctors agree, 0.0 = yes vs no
    reasoning_consistency: float    # cosine similarity of reasoning across rounds
    confidence_stability: float     # low std-dev of confidence = high stability
    breakdown: Dict[str, Any]       # weights + raw values for debugging/ablation
```

**`JudgeVerdict`** — the judge agent's final output:
```python
class JudgeVerdict(BaseModel):
    question: str
    final_answer: Literal["yes", "no", "maybe"]
    explanation: str                # 2-4 sentences citing evidence + doctor arguments
    trust: TrustScore               # the computed trust score for this debate
    debate_summary: str             # 2-3 sentence human-readable summary
```

**`ExperimentResult`** — one system's result on one question:
```python
class ExperimentResult(BaseModel):
    system_name: str                # "baseline_rag" | "debate_no_trust" | "full_system"
    question: str
    ground_truth: str
    predicted_answer: str
    correct: bool
    trust_score: Optional[float]    # None for systems that don't compute trust
    metadata: Dict[str, Any]
```

**`EvaluationReport`** — aggregated metrics for one system:
```python
class EvaluationReport(BaseModel):
    system_name: str
    num_questions: int
    accuracy: float                 # correct / total
    error_rate: float               # (wrong + maybe) / total
    trust_correlation: Optional[float]  # Pearson r(trust, correct)
    per_question_results: List[ExperimentResult]
```

---

### Step 2 — Trust Score Calculation (`trust.py`)

Pure computation — no LLM calls. Three independent sub-signals are combined with configurable
weights into a single composite score.

#### Sub-signal 1: Agent Agreement (weight: 0.40)

Measures whether the two doctors converged on the same conclusion.

| Scenario | Score |
|----------|-------|
| Both say the same position (e.g. both "yes") | 1.0 |
| One says "maybe", the other is definitive | 0.5 |
| Direct contradiction ("yes" vs "no") | 0.0 |

```python
def compute_agent_agreement(transcript: DebateTranscript) -> float:
    pos_a = transcript.doctor_a_final_position
    pos_b = transcript.doctor_b_final_position
    if pos_a == pos_b:
        return 1.0
    if "maybe" in (pos_a, pos_b):
        return 0.5
    return 0.0
```

#### Sub-signal 2: Reasoning Consistency (weight: 0.35)

Checks whether each doctor's reasoning stayed coherent across debate rounds. Uses
`sentence-transformers/all-MiniLM-L6-v2` to embed each reasoning text, then computes
average consecutive cosine similarity per agent.

- The embedding model is **lazy-loaded and cached** — loaded once on first call, reused after.
- For each agent, we take all their reasoning strings across rounds, embed them, and compute
  cosine similarity between consecutive pairs.
- Final score = mean of both agents' consistency scores.
- If an agent only has one round, their consistency defaults to 1.0 (trivially consistent).

```python
def compute_reasoning_consistency(messages: List[AgentMessage]) -> float:
    model = _get_embedding_model()  # lazy singleton
    agent_scores = []
    for agent_name in ("doctor_a", "doctor_b"):
        texts = [m.reasoning for m in messages if m.agent == agent_name]
        if texts:
            embeddings = model.encode(texts, convert_to_numpy=True)
            sims = [cosine_sim(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
            agent_scores.append(mean(sims))
    return mean(agent_scores)
```

#### Sub-signal 3: Confidence Stability (weight: 0.25)

Measures whether each doctor's confidence stayed stable vs fluctuating wildly.

- For each agent, compute standard deviation of their confidence values across rounds.
- Stability = `1.0 - 2.0 * mean(std_a, std_b)`, clamped to [0, 1].
- The `2.0` multiplier normalises because confidence (0-1) has a max possible std of 0.5.

#### Composite Formula

```python
overall = 0.40 * agreement + 0.35 * consistency + 0.25 * stability
```

Weights are stored in `DEFAULT_WEIGHTS` and can be overridden by passing a custom `weights`
dict to `compute_trust_score()` — useful for ablation experiments.

---

### Step 3 — Judge Agent (`judge_agent.py`)

An LLM-based judge that reads the full debate transcript and produces a justified final answer.
This is NOT a simple majority-vote — the judge weighs reasoning quality, evidence citations,
and confidence trajectories.

#### System Prompt

The judge is instructed to:
- Adopt the shared position if both doctors agree with high confidence.
- If they disagree, weigh evidence quality and internal consistency of each doctor's reasoning.
- Consider position changes across rounds as a positive signal if backed by evidence.
- Return "maybe" only when evidence is genuinely insufficient after the full debate.

#### Transcript Formatting (`format_transcript_for_judge`)

Builds a structured prompt string from a `DebateTranscript`:

```
MEDICAL QUESTION: <question>

RETRIEVED EVIDENCE:
[Evidence 1] (relevance score: 0.923)
<evidence text>
...

DEBATE TRANSCRIPT:
----------------------------------------
[Round 1 | Doctor A | Position: yes | Confidence: 0.82]
Evidence cited: evidence_1, evidence_3
Reasoning: Based on [Evidence 1], the study demonstrates...
----------------------------------------
[Round 1 | Doctor B | Position: no | Confidence: 0.75]
...

FINAL POSITIONS:
  Doctor A: yes (confidence: 0.85)
  Doctor B: no (confidence: 0.70)
```

Reuses `format_evidence_for_prompt()` from `src/debate/prompts.py` for evidence formatting.

#### Response Parsing (`parse_judge_response`)

Extracts three fields from the LLM response using regex:
- `FINAL_ANSWER:` → `yes` / `no` / `maybe`
- `EXPLANATION:` → 2-4 sentence justification
- `DEBATE_SUMMARY:` → 2-3 sentence debate summary

Falls back to `"maybe"` if parsing fails (safe default).

#### `run_judge(transcript, trust, llm=None) -> JudgeVerdict`

1. Creates LLM if not provided (reuses `create_llm()` from debate module).
2. Formats transcript into prompt string.
3. Sends `[SystemMessage(JUDGE_PROMPT), HumanMessage(formatted_transcript)]` to LLM.
4. Parses response and returns a `JudgeVerdict` with the pre-computed trust score attached.

---

### Step 4 — Pipeline Entrypoint (`pipeline.py`)

Provides two entrypoints that wire the full flow together.

#### `run_judge_pipeline(question, evidence, ground_truth?, max_rounds=2, llm?)`

Full end-to-end pipeline for a single question:

```
question + evidence
       │
       ▼
  run_debate()          ← Person 2's debate system
       │
       ▼ DebateTranscript
  compute_trust_score() ← trust.py
       │
       ▼ TrustScore
  run_judge()           ← judge_agent.py
       │
       ▼ JudgeVerdict
  package ExperimentResult (if ground_truth provided)
       │
       ▼
  return (verdict, trust, experiment_result)
```

#### `run_judge_on_transcript(transcript, ground_truth?, llm?)`

Same flow but **skips the debate step** — accepts a pre-existing `DebateTranscript`. Used in
`main.py` to avoid running the debate twice, and in batch experiments where transcripts can be
cached.

---

### Step 5 — Experiments (`experiments.py`)

Runs three systems on the same set of PubMedQA questions for a fair comparison.

#### System Definitions

| System | How it Works | Produces Trust? |
|--------|-------------|-----------------|
| **`baseline_rag`** | Single LLM call with retrieved evidence. Prompt asks for `ANSWER: yes/no/maybe`. No debate. | No |
| **`debate_no_trust`** | Runs full multi-agent debate, but uses **majority vote** to pick the answer. If both agree → that position. If they disagree → more confident doctor wins. No trust scoring. | No |
| **`full_system`** | Runs full debate + trust-aware judge. The judge reads all rounds, weighs reasoning quality, and produces a justified answer. Trust score computed. | Yes |

#### `run_single_experiment(question, evidence, ground_truth, system_name, ...)`

Dispatches to the appropriate system runner based on `system_name`. Returns an
`ExperimentResult` with predicted answer, correctness flag, and optional trust score.

#### `run_all_experiments(dataset, systems?, num_questions=50, ...)`

Iterates over `num_questions` from the dataset, runs all requested systems on each question,
and collects results. Handles errors gracefully — a failed question gets recorded as `"maybe"`
with the error in metadata instead of crashing the entire batch.

```python
results = run_all_experiments(dataset, num_questions=50)
# results = {
#     "baseline_rag": [ExperimentResult, ...],
#     "debate_no_trust": [ExperimentResult, ...],
#     "full_system": [ExperimentResult, ...],
# }
```

---

### Step 6 — Evaluation Metrics (`evaluation.py`)

Consumes experiment results and produces evaluation reports with three metrics.

#### Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | `correct / total` | Per system |
| **Error Rate** | `(wrong + maybe) / total` | "maybe" counts as an error (abstention penalty) |
| **Trust-Accuracy Correlation** | Pearson r(`trust_score`, `correct`) | Only for `full_system` (only system with trust scores). Requires >= 3 data points. Positive r = trust predicts correctness. |

#### `compute_evaluation_report(results, system_name) -> EvaluationReport`

Computes all three metrics from a list of `ExperimentResult` for one system.

#### `print_comparison_table(reports) -> str`

Pretty-prints a markdown table comparing all systems:

```
| System            | Questions | Accuracy | Error Rate | Trust-Acc Corr |
|-------------------|-----------|----------|------------|----------------|
| baseline_rag      |        50 |   0.6120 |     0.3880 |            N/A |
| debate_no_trust   |        50 |   0.6740 |     0.3260 |            N/A |
| full_system       |        50 |   0.7310 |     0.2690 |        +0.4300 |
```

#### Export Functions

- **`export_results_to_csv(reports, path?)`** — writes per-question results to
  `outputs/experiment_results.csv` with columns: system, question, ground_truth, predicted,
  correct, trust_score.
- **`export_results_to_json(reports, path?)`** — writes full reports + summary to
  `outputs/experiment_results.json`. Includes all metadata, trust breakdowns, and
  judge explanations.

Both functions create the `outputs/` directory automatically if it doesn't exist.

---

## Integration with `main.py`

The pipeline is wired into `main.py` as steps 5 and 6:

```python
# Step 5: Judge pipeline (reuses transcript from step 3)
from src.Judge.pipeline import run_judge_on_transcript

verdict, trust, result = run_judge_on_transcript(
    transcript=transcript,
    ground_truth=ground_truth,
)

# Step 6: Save artifacts to outputs/
from src.Judge.evaluation import (
    compute_evaluation_report,
    export_results_to_csv,
    export_results_to_json,
    print_comparison_table,
)

report = compute_evaluation_report([result], system_name="full_system")
print_comparison_table([report])
export_results_to_csv([report])     # → outputs/experiment_results.csv
export_results_to_json([report])    # → outputs/experiment_results.json
```

### Full Experiment Run (50 questions)

To run all three systems on 50 PubMedQA questions:

```python
from src.Judge.experiments import run_all_experiments
from src.Judge.evaluation import (
    compute_evaluation_report,
    export_results_to_csv,
    export_results_to_json,
    print_comparison_table,
)

# dataset = list of dicts with 'question', 'final_decision', 'evidence' keys
results = run_all_experiments(dataset, num_questions=50)

reports = [
    compute_evaluation_report(results[sys], sys)
    for sys in results
]
print_comparison_table(reports)
export_results_to_csv(reports)
export_results_to_json(reports)
```

---

## Output Artifacts

After a run, the `outputs/` directory contains:

| File | Contents |
|------|----------|
| `experiment_results.csv` | Flat table: system, question, ground_truth, predicted, correct, trust_score |
| `experiment_results.json` | Full nested report: per-question metadata, trust breakdowns, judge explanations, summary metrics |

### Sample Console Output

```
============================================================
JUDGE VERDICT
============================================================
Final Answer : yes
Explanation  : Both doctors agreed that metformin reduces HbA1c based on
               Evidence 1 and 3. Doctor A noted RCT quality; Doctor B
               highlighted real-world generalisability.
Debate Summary: The debate converged quickly with both doctors supporting
                a "yes" position by round 2.

Trust Score  : 0.847
  Agreement       : 1.00
  Consistency     : 0.82
  Stability       : 0.71

Correct: True
============================================================
EVALUATION REPORT
============================================================
| System            | Questions | Accuracy | Error Rate | Trust-Acc Corr |
|-------------------|-----------|----------|------------|----------------|
| full_system       |         1 |   1.0000 |     0.0000 |            N/A |

Artifacts saved:
  CSV  : outputs/experiment_results.csv
  JSON : outputs/experiment_results.json
============================================================
```

---

## Dependencies

All dependencies are listed in `requirements.txt`:

```
sentence-transformers       # all-MiniLM-L6-v2 for reasoning consistency embeddings
scipy>=1.11.0               # pearsonr for trust-accuracy correlation
pandas>=2.0.0               # CSV export of experiment results
pydantic>=2.0.0             # typed data models
langchain                   # LLM invocation (shared with debate module)
langchain-openai            # OpenAI/OpenRouter LLM backend
```

The sentence-transformer model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on first
use and is cached locally.

---

## Environment Variables

Same as the debate module — requires an API key in `.env` at the project root:

```
OPENROUTER_API_KEY=sk-or-...
# or
OPENAI_API_KEY=sk-...
```

---

## Module API Reference

### Models (`src.Judge.models`)
| Class | Description |
|-------|-------------|
| `TrustScore` | Composite trust score with 3 sub-signals + breakdown |
| `JudgeVerdict` | Final answer + explanation + trust + debate summary |
| `ExperimentResult` | One system's result on one question |
| `EvaluationReport` | Aggregated metrics for one system across all questions |

### Trust (`src.Judge.trust`)
| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_trust_score` | `(transcript, weights?) -> TrustScore` | Full composite trust score |
| `compute_agent_agreement` | `(transcript) -> float` | Agreement sub-signal |
| `compute_reasoning_consistency` | `(messages) -> float` | Consistency sub-signal |
| `compute_confidence_stability` | `(messages) -> float` | Stability sub-signal |

### Judge (`src.Judge.judge_agent`)
| Function | Signature | Description |
|----------|-----------|-------------|
| `run_judge` | `(transcript, trust, llm?) -> JudgeVerdict` | Run the LLM judge agent |
| `format_transcript_for_judge` | `(transcript) -> str` | Format transcript as prompt |
| `parse_judge_response` | `(text) -> dict` | Parse judge LLM response |

### Pipeline (`src.Judge.pipeline`)
| Function | Signature | Description |
|----------|-----------|-------------|
| `run_judge_pipeline` | `(question, evidence, ground_truth?, ...) -> (JudgeVerdict, TrustScore, ExperimentResult?)` | Full pipeline: debate + trust + judge |
| `run_judge_on_transcript` | `(transcript, ground_truth?, llm?) -> (JudgeVerdict, TrustScore, ExperimentResult?)` | Judge + trust on existing transcript |

### Experiments (`src.Judge.experiments`)
| Function | Signature | Description |
|----------|-----------|-------------|
| `run_single_experiment` | `(question, evidence, ground_truth, system_name, ...) -> ExperimentResult` | Run one system on one question |
| `run_all_experiments` | `(dataset, systems?, num_questions?, ...) -> Dict[str, List[ExperimentResult]]` | Run all systems on N questions |

### Evaluation (`src.Judge.evaluation`)
| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_evaluation_report` | `(results, system_name) -> EvaluationReport` | Compute metrics for one system |
| `print_comparison_table` | `(reports) -> str` | Print markdown comparison table |
| `export_results_to_csv` | `(reports, path?) -> str` | Export to CSV |
| `export_results_to_json` | `(reports, path?) -> str` | Export to JSON |
