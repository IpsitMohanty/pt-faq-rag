# PT FAQ RAG

Local-first FAQ assistant built with **FastAPI**, **Chroma**, **HuggingFace embeddings**, **Ollama**, and a lightweight **Node gateway**.

The system is designed to answer document-grounded FAQ queries with retrieval guardrails, fallback behavior, and a simple API-first architecture.

## What This Project Does

This project implements a local-first Retrieval-Augmented Generation (RAG) workflow for FAQ-style querying.

It retrieves relevant document chunks from a persistent vector store and then either:

- returns direct snippets for short or straightforward questions, or
- uses grounded LLM synthesis for longer or more complex questions

The design emphasizes practical safety and reliability rather than unconstrained generation.

## Why It Exists

FAQ and document-assistant systems often fail in predictable ways:

- retrieving weak or irrelevant context
- over-answering when evidence is thin
- timing out on generation
- responding fluently without enough grounding

This project addresses those risks with retrieval guardrails and fallback behavior so the assistant remains useful even when the LLM path is weak or unavailable.

## Core Design Goals

- local-first deployment
- document-grounded answers
- retrieval-aware guardrails
- graceful fallback when generation is slow or weak
- simple API composition using a lightweight gateway

## Stack

- FastAPI
- Chroma persistent vector store
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Ollama for grounded generation (`phi3:mini` by default, configurable via `OLLAMA_MODEL`)
- Node / Express gateway for proxying and CORS handling

## Architecture

Client -> Node gateway -> FastAPI -> Chroma -> Ollama (when available)

### Request pipeline

Each query passes through these stages in order, short-circuiting at the first hit:

1. **Acronym rewrite** — bare terms like `frs` expand to `What is FRS in Poshan Tracker?` before any lookup
2. **Query canonicalization** — colon/dash follow-up patterns (`registration: ekyc steps`) map to a canonical question
3. **Fuzzy correction** — token-level rapidfuzz matching against known domain terms
4. **Exact FAQ lookup** — normalized match against a pre-built `faq_index.json`; returns `mode: "faq_exact"`
5. **Clarification prompt** — broad single-topic queries (e.g. bare `registration`) return an options list instead of guessing
6. **Vector retrieval** — Chroma cosine similarity with distance gating; threshold relaxed slightly for short queries with a hard cap
7. **LLM generation** — retrieved chunk is passed to Ollama with a grounded prompt; returns `mode: "rag"`
8. **Retrieval fallback** — if Ollama is not running or times out, the top vector chunk is returned directly as `mode: "vector"`

Ollama is probed at startup. If it is not reachable, the service logs a warning and operates in retrieval-only mode automatically — no configuration change needed.

## Guardrails

The workflow includes several practical guardrails:

- distance gating
  retrieval confidence checks before generation is attempted

- acronym expansion
  helps queries match domain language more reliably

- snippet fallback
  ensures the system still returns useful evidence when synthesis is not appropriate

- timeout-safe grounded generation
  prevents the user experience from collapsing when the LLM path is slow or unavailable

## Response Modes

Every `/chat` response includes a `mode` field indicating which pipeline stage answered:

| `mode` | Meaning |
|---|---|
| `faq_exact` | Matched a pre-indexed FAQ entry exactly |
| `clarify` | Query was too broad; response includes options to narrow it |
| `rag` | Ollama generated an answer grounded in the retrieved chunk |
| `vector` | Ollama unavailable; top retrieved chunk returned directly |
| `not_found` | No chunk within the distance threshold |

## Repository Role

This repo is best understood as an applied RAG systems prototype focused on retrieval quality and operational safety, not just prompt orchestration.

It is useful as:

- a local FAQ assistant
- a reference architecture for guarded RAG pipelines
- a prototype for domain-specific document assistants

## Running the Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app:app --reload --port 8000
```

If your backend entrypoint differs, replace `app:app` with the correct module path.

## Running the Gateway

If the repository includes a separate Node gateway:

```powershell
cd gateway
npm install
npm run dev
```

If your scripts differ, use the actual package scripts defined in that folder.

## Local URLs

Typical development setup:

- FastAPI backend: `http://127.0.0.1:8000`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Node gateway: depends on the configured port in the gateway package

## Data / Index Expectations

The project assumes:

- a prepared Chroma collection or local indexing workflow
- compatible embedding model configuration
- local document content already chunked or ingestible into the vector store
- optional Ollama availability for grounded synthesis

## Example Use Cases

- internal FAQ assistant over operational documents
- policy or workflow lookup tool
- local document-grounded support assistant
- prototype RAG service with explicit guardrails

## Project Scope

This repository is focused on local-first, document-grounded FAQ retrieval and guarded answer generation.
