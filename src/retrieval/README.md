# Retrieval Pipeline – Medical Debate System

## Overview

This module implements the **retrieval pipeline** for the Medical Debate System project.

The goal of this component is to retrieve **relevant biomedical evidence** for a given medical question using a **hybrid retrieval architecture** that combines:

- BM25 keyword search
- Dense embedding similarity search
- Cross-encoder reranking

The retrieved evidence is later used by the **debate agents** to analyze medical questions and generate more reliable answers.

---

## Role in the System Pipeline

This module is the **first stage of the Medical Debate System pipeline**.

It retrieves relevant biomedical evidence for a medical question and passes the ranked evidence to the debate system.

Pipeline flow:

[Person 1: Retrieval Pipeline] → [Person 2: Debate System] → [Person 3: Judge + Trust Score]

The retrieval pipeline provides the **evidence used by the debate agents to reason about medical questions**.

---

## Retrieval Pipeline Architecture

The retrieval system follows this pipeline:

Dataset
↓
Document Chunking
↓
Embedding Generation
↓
Vector Store
↓
BM25 Retrieval
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking


---

## Implemented Components

### 1. Dataset Loading

Loads medical QA datasets used for retrieval experiments.

**File**
src/retrieval/load_dataset.py


**Functionality**

- Loads the PubMedQA dataset
- Extracts questions and associated contexts
- Prepares data for retrieval experiments

---

### 2. Document Chunking

Splits documents into smaller chunks to improve retrieval performance.

**File**
src/retrieval/chunk_documents.py


**Functionality**

- Cleans document text
- Splits documents into retrieval chunks
- Maintains metadata for evaluation

---

### 3. Embedding Generation

Converts document chunks into vector embeddings.

**File**
src/retrieval/create_embeddings.py


**Functionality**

- Loads embedding model
- Generates dense embeddings for documents
- Prepares vectors for similarity search

**Embedding Model**

Sentence Transformers

---

### 4. Vector Store

Stores embeddings in a vector database.

**File**
src/retrieval/vector_store.py


**Functionality**

- Creates ChromaDB collection
- Stores document embeddings
- Retrieves top-k similar documents

**Vector Database**

ChromaDB

---

### 5. BM25 Retrieval

Implements keyword-based lexical retrieval.

**File**
src/retrieval/bm25_index.py


**Functionality**

- Builds BM25 index
- Retrieves documents using lexical similarity

---

### 6. Hybrid Retrieval

Combines lexical and semantic search.

**File**
src/retrieval/retrieve_evidence.py


**Functionality**

- Retrieves candidate documents from BM25
- Retrieves candidate documents from vector search
- Combines both retrieval results

---

### 7. Reranking

Ranks retrieved documents using a cross-encoder model.

**File**
src/retrieval/reranker.py


**Functionality**

- Scores query–document pairs
- Reorders retrieved documents based on relevance

---

### 8. Retrieval Pipeline

The complete evidence retrieval system.

**File**
src/retrieval/evidence_pipeline.py


**Pipeline Flow**
Dataset
→ Document Chunking
→ Embedding Generation
→ Vector Store
→ BM25 Retrieval
→ Hybrid Retrieval
→ Reranking


This pipeline integrates all retrieval components to generate **ranked biomedical evidence for a query**.

---

## Evaluation

Retrieval performance is evaluated using queries from the dataset.

**Metrics Used**

- Recall@K
- HitRate@K
- Mean Reciprocal Rank (MRR)

Evaluation is performed using **held-out queries from the dataset**.

---

## Technologies Used

- Python
- Sentence Transformers
- HuggingFace Transformers
- ChromaDB
- BM25 Retrieval
- PyTorch
- HuggingFace Datasets

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
