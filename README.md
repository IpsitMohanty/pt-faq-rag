# PT FAQ RAG (FastAPI + Chroma + Ollama)

Local-first RAG FAQ assistant using FastAPI + Chroma + HuggingFace embeddings + Ollama, fronted by a lightweight Node gateway. Includes retrieval guardrails (distance gating, acronym expansion, snippet fallback, and timeout-safe grounded generation).

Local-first, document-grounded FAQ assistant.

---

## Stack

- FastAPI
- Chroma (persistent vector store)
- HuggingFace sentence-transformers embeddings
- Ollama (optional, grounded generation)
- Node/Express gateway (CORS + proxy)

---

## Architecture

Client → Gateway (Node) → FastAPI → Chroma → Ollama

- Short queries return document snippets  
- Long queries use grounded LLM synthesis  
- LLM timeout falls back to snippets

---

## Run (Backend)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app:app --reload --port 8000
