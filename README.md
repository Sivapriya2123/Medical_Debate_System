# Trust-Aware Multi-Agent Medical Debate with Reward-Guided Optimization

A multi-agent debate system for biomedical question answering on [PubMedQA](https://pubmedqa.github.io/). Two LLM-based doctor agents debate over retrieved PubMed evidence, and a judge agent uses trust signals to make final decisions. We apply reward-guided prompt optimization (inspired by GRPO) to reduce the judge's tendency to over-predict "maybe" and improve overall accuracy.

**CS 6180 Generative AI -- Final Project**
**Authors:** Mohammed Ahnaf, Sri Ram, Sivapriya

---

## Key Results

| System | Accuracy | Maybe Over-correction |
|--------|----------|-----------------------|
| No retrieval (baseline) | 38.0% | -- |
| RAG only (no debate) | 65.0% | -- |
| Debate + majority vote | 78.0% | -- |
| Debate + static trust judge | 77.0% | 11.5% |
| **Debate + GRPO-optimized judge** | **79.0%** | **7.3%** |

- Retrieval adds +27%, debate adds +13%, GRPO optimization adds +2%
- Static trust weights actually hurt performance (-1% vs majority vote)
- GRPO prompt optimization fixes this and surpasses all baselines
- Confidence stability is the dominant trust signal (optimal weight 0.65); agreement has near-zero optimal weight

---

## System Architecture

```
Question --> [Hybrid Retrieval] --> [Doctor A] <--> [Doctor B] --> [Trust Scoring] --> [Judge] --> Answer
              ChromaDB + BM25        (conservative)   (generalist)   3 sub-signals      (GRPO-optimized)
              + cross-encoder         2 rounds of structured debate    - agreement
                                                                       - similarity
                                                                       - stability
```

**Agents** (all GPT-4o-mini via OpenRouter):
- **Doctor A**: Conservative specialist -- cautious, evidence-focused
- **Doctor B**: Generalist -- broader reasoning, weighs clinical context
- **Judge**: Impartial evaluator -- synthesizes debate using trust signals

**Retrieval Pipeline**:
- ChromaDB dense retrieval + BM25 sparse retrieval (hybrid)
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- Temporal filtering and conflict detection

**Trust Scoring** (3 sub-signals):
- Agreement score: do doctors reach the same conclusion?
- Embedding similarity: cosine similarity of reasoning embeddings
- Confidence stability: consistency across debate rounds

---

## Demo

https://github.com/user-attachments/assets/REPLACE_WITH_YOUR_VIDEO_ID

A React + FastAPI web interface lets you interact with the full pipeline visually:

- Evidence cards with relevance scores, temporal flags, and conflict badges
- Character-based debate between Dr. Chen (conservative) and Dr. Patel (generalist), with clickable citation pills that link back to evidence
- Trust score breakdown with animated bars
- Judge verdict panel with the GRPO-optimized final answer

**To run the UI:**

```bash
# Terminal 1 -- Backend
conda activate medical_debate
uvicorn ui.backend.server:app --reload --port 8000

# Terminal 2 -- Frontend
cd ui/frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Repository Structure

```
Medical_Debate_System/
|
|-- src/                          # Core system components
|   |-- retrieval/                # Evidence retrieval pipeline
|   |   |-- load_dataset.py      # PubMedQA data loading
|   |   |-- vector_store.py      # ChromaDB vector store
|   |   |-- bm25_index.py        # BM25 sparse retrieval
|   |   |-- reranker.py          # Cross-encoder re-ranking
|   |   |-- retrieve_evidence.py # Hybrid retrieval + reranking
|   |   |-- temporal_filter.py   # Temporal relevance filtering
|   |   +-- conflict_detector.py # Evidence conflict detection
|   |-- debate/                   # Multi-agent debate system
|   |   |-- graph.py             # LangGraph debate orchestration
|   |   |-- agents.py            # Doctor agent definitions
|   |   |-- models.py            # State models for debate
|   |   +-- prompts.py           # Agent system prompts
|   +-- Judge/                    # Judge agent & evaluation
|       |-- judge_agent.py       # Judge with trust-aware decision making
|       |-- trust.py             # Trust signal computation
|       |-- pipeline.py          # Full judge pipeline
|       +-- evaluation.py        # Accuracy evaluation
|
|-- grpo/                         # GRPO optimization framework
|   |-- data/
|   |   +-- collect_traces.py    # Debate trace collection (500 samples)
|   |-- training/
|   |   |-- prompt_variants.py   # 9 judge prompt variants (4 categories)
|   |   |-- judge_grpo.py        # Core scoring & evolution engine
|   |   +-- trust_weight_optimizer.py  # Trust weight grid search
|   |-- eval/
|   |   |-- baseline_metrics.py  # Phase 0 baseline computation
|   |   |-- error_analysis.py    # Error categorization
|   |   |-- generate_figures.py  # Paper figure generation
|   |   |-- retrieval_metrics.py # Recall@K, Precision@K, MRR
|   |   |-- generation_metrics.py# Faithfulness, relevancy, citations
|   |   |-- significance_test.py # McNemar's test
|   |   +-- integrated_eval.py   # Full ablation table
|   |-- rewards/
|   |   +-- reward_functions.py  # Correctness, anti-maybe, format, citation
|   +-- configs/
|       +-- grpo_config.yaml     # Configuration for all phases
|
|-- ui/                           # Web interface
|   |-- backend/
|   |   +-- server.py            # FastAPI backend (POST /api/run)
|   +-- frontend/                # React + Vite + Tailwind
|       |-- src/
|       |   |-- App.jsx          # Main app shell
|       |   |-- components/      # QuestionInput, DebatePanel, JudgePanel, etc.
|       |   |-- hooks/           # usePipeline API hook
|       |   +-- utils/           # Citation parser
|       +-- package.json
|
|-- experiments/                  # Experimental outputs
|   |-- traces/                  # Collected debate traces (500 samples)
|   +-- results/                 # JSON result files from all phases
|
|-- main.py                      # End-to-end pipeline entry point
|-- requirements.txt             # Python dependencies
+-- .env.example                 # API key template
```

---

## Setup & Reproduction

### Prerequisites
- Python 3.10+ (tested on 3.13)
- Node.js 18+ (for the web UI)
- [OpenRouter](https://openrouter.ai/) API key (for GPT-4o-mini access)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/Medical_Debate_System.git
cd Medical_Debate_System

# Create virtual environment (conda or venv)
conda create -n medical_debate python=3.13 -y
conda activate medical_debate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Run the system

**Full experiment (50 questions, all system variants):**
```bash
python main.py --experiment --num_questions 50
```

**Web UI:**
```bash
# Terminal 1
conda activate medical_debate
uvicorn ui.backend.server:app --reload --port 8000

# Terminal 2
cd ui/frontend && npm install && npm run dev
```

---

## Reproducing GRPO Experiments

The GRPO optimization experiments run in 4 phases. Phases 0-1 require API calls (~6000 total); Phases 2-4 are local computation only.

### Phase 0: Collect debate traces and compute baselines

```bash
# Collect 500 debate traces (takes ~2-3 hours, ~500 API calls)
python -m grpo.data.collect_traces --num_samples 500 --output experiments/traces/debate_traces_full.jsonl

# Compute baseline metrics
python -m grpo.eval.baseline_metrics --traces experiments/traces/debate_traces_full.jsonl
```

### Phase 1: Judge prompt optimization

```bash
# Score all 10 prompt variants on 400 training samples (~4000 API calls, ~3-4 hours)
python -m grpo.training.judge_grpo --mode score_variants --traces experiments/traces/debate_traces_full.jsonl

# Evolve top variants and score hybrids (~2400 API calls, ~2-3 hours)
python -m grpo.training.judge_grpo --mode evolve --traces experiments/traces/debate_traces_full.jsonl

# Final evaluation on held-out 100 test samples (~200 API calls)
python -m grpo.training.judge_grpo --mode evaluate --traces experiments/traces/debate_traces_full.jsonl --variant decisive_v1
```

**Note:** Phase 1 scoring supports incremental saving. If interrupted, re-run the same command -- it skips already-completed variants.

### Phase 2: Trust weight optimization (no API calls)

```bash
python -m grpo.training.trust_weight_optimizer --mode full --traces experiments/traces/debate_traces_full.jsonl
```

### Phase 3: Error analysis (no API calls)

```bash
# Error analysis
python -m grpo.eval.error_analysis --traces experiments/traces/debate_traces_full.jsonl

# Integrated ablation table
python -m grpo.eval.integrated_eval --traces experiments/traces/debate_traces_full.jsonl
```

### Phase 4: Retrieval/generation quality and significance tests (no API calls)

```bash
python -m grpo.eval.retrieval_metrics --traces experiments/traces/debate_traces_full.jsonl
python -m grpo.eval.generation_metrics --traces experiments/traces/debate_traces_full.jsonl
python -m grpo.eval.significance_test --traces experiments/traces/debate_traces_full.jsonl
```

---

## Pre-computed Results

All experimental results are included in `experiments/` so the full pipeline does not need to be re-run:

- `experiments/traces/debate_traces_full.jsonl` -- 500 debate traces with all agent responses
- `experiments/results/` -- JSON files with metrics from all phases

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | 1.2.15 | LLM abstraction layer |
| langchain-openai | 1.1.12 | OpenRouter/OpenAI integration |
| langgraph | 1.1.6 | Multi-agent debate orchestration |
| chromadb | 1.5.5 | Dense vector retrieval |
| rank-bm25 | 0.2.2 | Sparse BM25 retrieval |
| sentence-transformers | 5.3.0 | Embedding models & cross-encoder |
| datasets | 3.5.0 | PubMedQA dataset loading |
| numpy | >=2.0.0 | Numerical computation |
| scipy | >=1.11.0 | Statistical tests (McNemar's) |
| matplotlib | >=3.8.0 | Figure generation |
| python-dotenv | 1.2.2 | Environment variable management |
| pydantic | >=2.0.0 | Data validation |
| torch | >=2.0.0 | ML backend for transformers |
| transformers | >=5.0.0 | Model loading for embeddings |
| fastapi | latest | Web UI backend |
| uvicorn | latest | ASGI server |

Full list with versions: see `requirements.txt`

---

## Methodology Overview

### GRPO-Inspired Prompt Optimization (Phase 1)

Instead of gradient-based GRPO (which requires model weight access), we apply the GRPO loop at the prompt level:

1. **Sample**: Generate 9 judge prompt variants targeting 4 failure modes (anti-maybe decisiveness, doctor weighting, trust interpretation, structured reasoning)
2. **Score**: Evaluate each variant on 400 training traces using a reward function: R = R_correctness(+1.0) + R_anti_maybe(-0.3) + R_format(+0.25) + R_evidence(+0.25)
3. **Select**: Pick top 2 performers
4. **Iterate**: Evolve 4 hybrid variants from winners, re-score, select final best

Winner: `decisive_v1` -- 70.0% training accuracy (vs 67.8% baseline), 79.0% on held-out test.

### Trust Weight Analysis (Phase 2)

Grid search over 231 weight combinations revealed:
- Confidence stability should dominate (optimal weight: 0.65)
- Agreement score has near-zero optimal weight
- Removing agreement actually improves accuracy (+0.4%)

### Error Analysis (Phase 3)

91.3% of remaining errors are unrecoverable (both doctors wrong). The theoretical accuracy ceiling is 79.0%, which the GRPO system has reached.
