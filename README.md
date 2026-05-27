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
- HuggingFace sentence-transformers embeddings
- Ollama for optional grounded generation
- Node / Express gateway for proxying and CORS handling

## Architecture

Client -> Node gateway -> FastAPI -> Chroma -> Ollama

### Request flow

1. A client sends a query to the Node gateway.
2. The gateway forwards the request to the FastAPI backend.
3. The backend embeds the query and retrieves candidate chunks from Chroma.
4. Guardrails inspect retrieval quality.
5. Depending on the query and retrieval strength:
   - direct snippets are returned, or
   - grounded synthesis is requested from Ollama
6. If generation times out or fails quality checks, the system falls back to snippets.

## Guardrails

The README mentions several important guardrails, and these are a strong part of the project story:

- distance gating
narrow retrieval confidence checks before generation is attempted

- acronym expansion
helps queries match domain language more reliably

- snippet fallback
ensures the system still returns useful evidence when synthesis is not appropriate

- timeout-safe grounded generation
prevents the user experience from collapsing when the LLM path is slow or unavailable

## Typical Query Behavior

- **Short, precise questions**
  usually return direct document snippets

- **Longer or more interpretive questions**
  may use grounded LLM synthesis when retrieval is strong enough

- **Weak retrieval or generation timeout**
  falls back to supporting snippets instead of guessing

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

If the vector store is empty, retrieval quality will be poor regardless of prompt behavior.

## Example Use Cases

- internal FAQ assistant over operational documents
- policy or workflow lookup tool
- local document-grounded support assistant
- prototype RAG service with explicit guardrails

## Current Limitations

- local-first setup may require manual environment preparation
- retrieval and generation quality depend on the indexed corpus
- no claim of production-grade multi-user deployment
- fallback behavior is safer than free-form generation, but not a substitute for full evaluation

## Suggested Next Improvements

- add ingestion/indexing documentation
- add request and response examples
- add architecture diagram or screenshots
- document the retrieval scoring logic more explicitly
- add benchmark or evaluation notes for FAQ quality
