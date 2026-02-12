# PT FAQ RAG (FastAPI + Chroma + Ollama)

Local-first, document-grounded FAQ assistant.

**Backend (`backend/app.py`)**
- FastAPI API
- Chroma persistent vector store (persisted locally; ignored in git)
- HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Ollama local LLM (default `phi3:mini`) for longer answers
- Guardrails: gibberish detection, unknown-definition guard, distance thresholding, snippet fallback

## Architecture
Gateway/UI → `POST /chat` → retrieval (Chroma) → either:
- **Snippet mode** (definitions / procedures / short queries)
- **LLM synthesis** (grounded by retrieved context; timeout falls back to snippets)

## Repo structure
```text
backend/
  app.py
  requirements.txt
  .env.example
  chroma_store/   # ignored (generated)
  data/           # ignored (docx lives here locally)
gateway/
  server.js
  package.json
  package-lock.json
