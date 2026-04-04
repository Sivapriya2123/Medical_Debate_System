# Multi-Agent Debate System (Person 2)

## Overview

This module implements a **dual-agent medical debate system** using LangGraph and LangChain. Two LLM-powered doctor agents analyze a biomedical research question using retrieved scientific evidence, debate their positions across multiple rounds, and produce a structured transcript for downstream evaluation.

The system sits between **Person 1's retrieval pipeline** (`src/retrieval/`) and **Person 3's judge + evaluation** (`src/Judge/`, `src/evaluation/`).

```
[Person 1: Retrieval Pipeline] → [Person 2: Debate System] → [Person 3: Judge + Trust Score]
         evidence                    transcript                     final answer
```

---

## Architecture

### Debate Flow (default: 2 rounds)

```
initialize
    │
    ▼
doctor_a (Round 1 opening)
    │
    ▼
doctor_b (Round 1 opening)
    │
    ▼
check_completion ──→ (Round 2 needed?)
    │ yes                   │ no
    ▼                       ▼
doctor_a (rebuttal)     finalize → END
    │
    ▼
doctor_b (rebuttal)
    │
    ▼
check_completion → finalize → END
```

With `max_rounds=2`, this produces **4 messages**: A-opening, B-opening, A-rebuttal, B-rebuttal.

### Agent Personas

| Agent | Role | Approach |
|-------|------|----------|
| **Doctor A** | Conservative clinical specialist | Prioritizes established guidelines, scrutinizes methodology (sample size, controls, confounders). Leans toward "maybe" when evidence is weak or conflicting. |
| **Doctor B** | Diagnostic generalist | Synthesizes across evidence sources, looks for broader patterns. Willing to commit to yes/no when collective evidence supports it. Explores alternative interpretations. |

These distinct personas create productive disagreement, which Person 3's trust score can measure.

---

## Files

| File | Purpose |
|------|---------|
| `models.py` | Pydantic data models: `EvidenceItem`, `AgentMessage`, `DebateTranscript`, `DebateState` |
| `prompts.py` | System prompts for both doctors, user templates for openings and rebuttals, response parser, evidence formatter |
| `agents.py` | LangChain wrapper functions for LLM calls (opening arguments, rebuttals) |
| `graph.py` | LangGraph state machine (5 nodes) and the public `run_debate()` API |
| `__init__.py` | Clean public exports |

---

## Usage

### Quick Start

```python
from src.retrieval.evidence_pipeline import build_retrieval_pipeline
from src.retrieval.retrieve_evidence import retrieve_evidence_hybrid_reranked
from src.retrieval.create_embeddings import load_embedding_model
from src.retrieval.reranker import load_reranker
from src.debate import run_debate

# Setup retrieval (one-time)
collection, bm25, documents = build_retrieval_pipeline(limit=500)
model = load_embedding_model()
reranker = load_reranker()

# Retrieve evidence for a question
question = "Does post-mastectomy radiotherapy reduce breast cancer recurrence?"
evidence = retrieve_evidence_hybrid_reranked(
    collection, model, bm25, documents, reranker,
    query=question, top_k=3, candidate_k=10
)

# Run the debate
transcript = run_debate(question=question, evidence=evidence, max_rounds=2)

# Inspect results
print(transcript.doctor_a_final_position)   # "yes", "no", or "maybe"
print(transcript.doctor_b_final_position)
print(transcript.doctor_a_final_confidence) # 0.0 to 1.0
print(transcript.doctor_b_final_confidence)

for msg in transcript.messages:
    print(f"[Round {msg.round}] {msg.agent}: {msg.position} ({msg.confidence})")
    print(f"  Evidence cited: {msg.evidence_cited}")
    print(f"  Reasoning: {msg.reasoning[:200]}...")
```

### Environment Setup

Requires `OPENAI_API_KEY` in a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

### Dependencies (added to requirements.txt)

```
langchain
langchain-openai
langgraph
python-dotenv
```

---

## Output Contract for Person 3

The `DebateTranscript` is the structured output that Person 3's judge agent consumes:

```python
class DebateTranscript:
    question: str                              # The biomedical question
    evidence: List[EvidenceItem]               # Evidence both agents received
    messages: List[AgentMessage]               # Full debate history
    num_rounds: int                            # Number of debate rounds
    doctor_a_final_position: "yes"|"no"|"maybe"
    doctor_b_final_position: "yes"|"no"|"maybe"
    doctor_a_final_confidence: float           # 0.0 to 1.0
    doctor_b_final_confidence: float           # 0.0 to 1.0
    metadata: dict                             # For ground truth labels, timing, etc.
```

Each `AgentMessage` contains:

```python
class AgentMessage:
    agent: "doctor_a" | "doctor_b"
    round: int
    position: "yes" | "no" | "maybe"
    reasoning: str                    # Detailed reasoning with evidence citations
    evidence_cited: List[str]         # e.g. ["evidence_1", "evidence_2"]
    confidence: float                 # 0.0 to 1.0
```

**Fields useful for trust scoring:**
- Agreement/disagreement between `doctor_a_final_position` and `doctor_b_final_position`
- Confidence trajectories across rounds (did agents become more or less confident?)
- Whether agents changed positions between rounds
- Number and overlap of evidence citations
- Quality and specificity of reasoning

---

## What Has Been Implemented

- [x] **Data models** -- Pydantic models with full serialization (`model_dump()` / JSON)
- [x] **Doctor A system prompt** -- Conservative clinician persona with structured output format
- [x] **Doctor B system prompt** -- Diagnostic generalist persona with structured output format
- [x] **Opening argument template** -- Presents question + numbered evidence to agents
- [x] **Rebuttal template** -- Shows opponent's position, confidence, evidence cited, and reasoning
- [x] **Response parser** -- Regex-based parser with safe fallbacks for malformed LLM output
- [x] **Evidence formatter** -- Handles both dict and EvidenceItem objects, includes relevance scores
- [x] **Agent invocation layer** -- LangChain message building, LLM calls, chat history management
- [x] **LangGraph debate loop** -- 5-node state machine with conditional round control
- [x] **Public API** -- `run_debate(question, evidence, max_rounds)` returns `DebateTranscript`
- [x] **Integration notebook** -- `notebooks/debate_test.ipynb` for end-to-end testing
- [x] **Dependency updates** -- `requirements.txt` updated, `.env` added to `.gitignore`

## What Remains to Be Implemented

### Person 3: Judge Agent, Trust Score & Evaluation (`src/Judge/`, `src/evaluation/`)

- [ ] **Judge agent** -- Takes `DebateTranscript`, synthesizes both agents' arguments, and produces a final answer (yes/no/maybe)
- [ ] **Trust score calculation** -- Compute a reliability score from:
  - Agent agreement/disagreement
  - Confidence levels and changes across rounds
  - Reasoning consistency (do agents cite the same evidence?)
  - Whether agents changed positions during debate
- [ ] **Baseline experiments** -- Run and compare:
  - Baseline 1: Single LLM without retrieval
  - Baseline 2: Single-agent RAG
  - Baseline 3: Multi-agent debate without evidence filtering
  - Proposed system: Debate + temporal filtering + conflict detection + trust scoring
- [ ] **Evaluation metrics** -- Accuracy, evidence error rate, trust-correctness correlation, debate efficiency (token usage)
- [ ] **Results analysis** -- Statistical comparison across baselines

### Person 1: Remaining Retrieval Enhancements

- [ ] **Temporal filtering** -- The `year` field in documents is currently `None`; needs extraction from PubMed metadata to filter outdated evidence
- [ ] **Conflict detection** -- Identify contradictory claims across retrieved passages before passing to debate agents

### Integration

- [ ] **End-to-end pipeline** -- Connect retrieval → filtering → debate → judge → evaluation in a single runnable script
- [ ] **Batch processing** -- Run the full system on the PubMedQA test set for evaluation
- [ ] **Output storage** -- Save debate transcripts, judge decisions, trust scores, and metrics to `outputs/`
