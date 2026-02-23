import os
import re
from typing import List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------
# Config (env)
# ----------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "12"))

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
COLLECTION = os.getenv("CHROMA_COLLECTION", "pt_faq")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chroma score is "distance" (lower is better).
# For short/acronym queries we allow a slightly looser threshold.
MAX_DISTANCE_OK = float(os.getenv("MAX_DISTANCE_OK", "0.95"))
SHORT_QUERY_DISTANCE_BONUS = float(os.getenv("SHORT_QUERY_DISTANCE_BONUS", "0.35"))

# Retrieval hygiene toggles
EXPAND_ACRONYMS = os.getenv("EXPAND_ACRONYMS", "1") == "1"


# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI(title="PT FAQ RAG (backend/app.py)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Vector DB (Chroma)
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)


# ----------------------------
# Fast definitions / aliases (builtin)
# ----------------------------
THR_DEF = (
    "THR stands for Take Home Ration. It refers to supplementary nutrition distributed "
    "to eligible beneficiaries for consumption at home."
)
FRS_DEF = (
    "FRS stands for Face Recognition System. It is used to verify the identity of beneficiaries "
    "mainly during Beneficiary Registration and Take Home Ration (THR) distribution."
)
SNP_DEF = (
    "SNP stands for Supplementary Nutrition Programme. It refers to supplementary nutrition services "
    "provided to eligible beneficiaries (e.g., THR / Hot Cooked Meal as applicable)."
)
POSHAN_DEF = (
    "Poshan Abhiyaan is India's flagship programme to improve nutritional outcomes for children, "
    "adolescents, pregnant women and lactating mothers by leveraging technology, a targeted approach "
    "and convergence."
)

FAST_DEFS = {
    "thr": THR_DEF,
    "take home ration": THR_DEF,
    "take-home ration": THR_DEF,
    "frs": FRS_DEF,
    "face recognition system": FRS_DEF,
    "ekyc": "eKYC is electronic Know Your Customer using Aadhaar OTP authentication.",
    "e-kyc": "eKYC is electronic Know Your Customer using Aadhaar OTP authentication.",
    "awc": "AWC stands for Anganwadi Centre.",
    "anganwadi centre": "AWC stands for Anganwadi Centre.",
    "snp": SNP_DEF,
    "supplementary nutrition": SNP_DEF,
    "supplementary nutrition programme": SNP_DEF,
    "supplementary nutrition program": SNP_DEF,
    "poshan": POSHAN_DEF,
    "poshan abhiyaan": POSHAN_DEF,

    # --- events / acronyms users type ---
    "cbe": "CBE stands for Community Based Event. In Poshan Tracker, CBE is one of the events that AWW can mark and enter details.",
    "community based event": "CBE stands for Community Based Event. In Poshan Tracker, CBE is one of the events that AWW can mark and enter details.",
    "vhsnd": "VHSND stands for Village Health Sanitation Nutrition Day. In Poshan Tracker, VHSND is one of the events that AWW can mark and enter details.",
    "village health sanitation nutrition day": "VHSND stands for Village Health Sanitation Nutrition Day. In Poshan Tracker, VHSND is one of the events that AWW can mark and enter details.",
    "ecce": "ECCE stands for Early Childhood Care and Education. In Daily Tracking, AWW can mark ECCE learning modules and preschool activities.",
    "early childhood care and education": "ECCE stands for Early Childhood Care and Education. In Daily Tracking, AWW can mark ECCE learning modules and preschool activities.",
    "events": "In Poshan Tracker, there are three events AWW can mark: VHSND, CBE, and AWW Training.",
}

# Domain expansions for retrieval (used only to improve embeddings search)
POSHAN_TERMS = {
    "thr": "take home ration",
    "frs": "face recognition system",
    "awc": "anganwadi centre",
    "snp": "supplementary nutrition programme",
    "cbe": "community based event",
    "vhsnd": "village health sanitation nutrition day",
    "ecce": "early childhood care and education",
    "poshan": "poshan abhiyaan poshan tracker purpose",
    "registration": "beneficiary registration e-kyc face capture",
}

# Improved prompt: still strict but more natural, stepwise output
SYSTEM_PROMPT = (
    "You are a helpful assistant for Poshan Tracker FAQs.\n"
    "Answer using ONLY the provided Context.\n"
    "Be clear and direct in 2–8 lines. Use bullet points for steps.\n"
    "Do not mention 'context', 'retrieval', or internal logic.\n"
    "If the answer is not present, say exactly:\n"
    "\"This information is not available in the PT FAQ document.\""
)


# ----------------------------
# Request schema
# ----------------------------
class ChatRequest(BaseModel):
    query: str


# ----------------------------
# Helpers
# ----------------------------
def normalize(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def normalize_for_retrieval(q: str) -> str:
    """
    Normalize for better recall:
    - lowercase
    - strip edge punctuation
    - naive singularization for trailing 's' in single-token queries (vaccine/vaccines)
    - acronym expansion for short queries (cbe -> cbe (community based event))
    - a small synonym expansion for a few domain terms (poshan/registration)
    """
    t = (q or "").strip()
    t = re.sub(r"\s+", " ", t)
    tl = t.lower().strip().strip("?.!,;:")

    # crude singularization for single-token plurals
    if " " not in tl and len(tl) > 3 and tl.endswith("s"):
        tl = tl[:-1]

    if EXPAND_ACRONYMS:
        toks = tl.split()
        if len(toks) <= 3:
            out = []
            for w in toks:
                ww = w.strip("?.!,;:")
                if ww in POSHAN_TERMS:
                    out.append(f"{ww} ({POSHAN_TERMS[ww]})")
                else:
                    out.append(ww)
            tl = " ".join(out)

    return tl


def is_explainish(q: str) -> bool:
    ql = (q or "").lower()
    return any(w in ql for w in ["explain", "procedure", "steps", "how", "process", "workflow", "detail", "detailed"])


def is_definition_query(q: str) -> bool:
    ql = (q or "").lower().strip()
    return (
        ql.startswith("what is ")
        or ql.startswith("whats ")
        or ql.startswith("define ")
        or ql.startswith("meaning of ")
        or ql.startswith("full form of ")
    )


def is_numbers_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(w in ql for w in ["how many", "maximum", "max", "minimum", "min", "%", "percent", "days", "times"])


def looks_like_acronymish(term: str) -> bool:
    """
    Short single tokens like frs/rch/abha/otp are treated as acronym-ish.
    Used only for *definition* guardrails, and only after retrieval.
    """
    t = re.sub(r"[^a-zA-Z0-9]", "", (term or "").strip())
    if not (2 <= len(t) <= 8):
        return False
    return t.isalnum()


def looks_like_id_or_code(q: str) -> bool:
    """
    Ticket IDs / hash-like strings / device refs should not go to RAG.
    """
    t = (q or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{12,}", t):  # long single token
        return True
    if re.fullmatch(r"(err|error|id|ref)[\-_]?[A-Za-z0-9]{6,}", t.lower()):
        return True
    return False


def maybe_fast_def(q: str) -> Optional[str]:
    """
    Fast answer only for:
    - exact term matches (thr/frs/cbe/vhsnd/ecce/etc)
    - explicit "what is/define/full form of X" patterns
    """
    key = (q or "").lower().strip().strip("?.!,;:")
    if key in FAST_DEFS:
        return FAST_DEFS[key]

    m = re.fullmatch(r"(what is|whats|define|meaning of|full form of)\s+(.{2,60})", key)
    if m:
        term = m.group(2).strip().strip("?.!,;:")
        if term in FAST_DEFS:
            return FAST_DEFS[term]

    return None


def trim_nicely(text: str, limit: int = 900) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t

    cut = t[:limit]
    for sep in [".", "?", "!", "\n"]:
        idx = cut.rfind(sep)
        if idx > int(limit * 0.6):
            return cut[: idx + 1].strip()
    return cut.strip()


def snippet_answer(docs: List, max_docs: int = 2, max_each: int = 900) -> Optional[str]:
    parts = []
    for d in docs[:max_docs]:
        txt = (getattr(d, "page_content", "") or "").strip()
        if txt:
            parts.append(trim_nicely(txt, max_each))
    if not parts:
        return None
    return "\n\n---\n\n".join(parts)


def build_context(docs: List, max_chars: int = 2800) -> str:
    ctx = ""
    for d in docs:
        chunk = (getattr(d, "page_content", "") or "").strip()
        if not chunk:
            continue
        if len(ctx) + len(chunk) + 2 > max_chars:
            break
        ctx += chunk + "\n\n"
    return ctx.strip()


def weak_retrieval(docs: List, min_total_chars: int) -> bool:
    if not docs:
        return True
    total = sum(len((getattr(d, "page_content", "") or "").strip()) for d in docs)
    return total < min_total_chars


def top_sources(docs: List, n: int = 3) -> List[str]:
    out = []
    for d in docs[:n]:
        meta = getattr(d, "metadata", {}) or {}
        s = meta.get("source") or meta.get("file") or meta.get("doc") or ""
        if s:
            out.append(s)
    return out


def is_gibberish(q: str) -> bool:
    t = (q or "").strip()
    if not t:
        return True

    tokens = t.split()
    if len(tokens) == 1:
        w = tokens[0].strip().strip("?.!,;:")
        if len(w) <= 2:
            return False

        letters = re.sub(r"[^a-zA-Z]", "", w)
        if len(letters) >= 6:
            vowels = sum(1 for c in letters.lower() if c in "aeiou")
            if vowels / len(letters) < 0.20:
                return True

        if re.fullmatch(r"[A-Za-z0-9]{10,}", w) and not re.search(r"[aeiouAEIOU]", w):
            return True

    return False


def extract_unknown_term_if_definition(q: str) -> Optional[str]:
    ql = (q or "").lower().strip().strip("?.!,;:")
    m = re.fullmatch(r"(what is|whats|define|meaning of|full form of)\s+(.{2,60})", ql)
    if not m:
        return None
    term = m.group(2).strip().strip("?.!,;:")
    term = re.sub(r"[^a-zA-Z0-9\- ]", "", term).strip()
    return term or None


def retrieve_with_scores(query: str, k: int) -> Tuple[List, List[float]]:
    retrieved = vectordb.similarity_search_with_score(query, k=k)
    docs = [d for d, _ in retrieved]
    scores = [float(s) for _, s in retrieved]
    return docs, scores


def pick_best_retrieval(raw_query: str, k: int) -> Tuple[str, List, List[float]]:
    """
    Try both raw and normalized-for-retrieval queries.
    Choose the one with lowest best-distance score.
    """
    q_raw = (raw_query or "").strip()
    q_norm = normalize_for_retrieval(raw_query)

    candidates = []
    for q in [q_raw, q_norm]:
        if not q:
            continue
        docs, scores = retrieve_with_scores(q, k=k)
        best = min(scores) if scores else 999.0
        candidates.append((best, q, docs, scores))

    if not candidates:
        return q_raw, [], []

    candidates.sort(key=lambda x: x[0])
    _, q, docs, scores = candidates[0]
    return q, docs, scores


def call_ollama(messages) -> str:
    import httpx

    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 140,  # lower = faster
        },
    }

    with httpx.Client(timeout=OLLAMA_TIMEOUT_S) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "") or ""


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "backend/app.py",
        "persist_dir": PERSIST_DIR,
        "collection": COLLECTION,
        "embed_model": EMBED_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_timeout_s": OLLAMA_TIMEOUT_S,
        "max_distance_ok": MAX_DISTANCE_OK,
        "short_query_distance_bonus": SHORT_QUERY_DISTANCE_BONUS,
        "expand_acronyms": EXPAND_ACRONYMS,
    }


@app.get("/debug/search")
def debug_search(q: str = "cbe"):
    used_query, docs, scores = pick_best_retrieval(q, k=5)
    return {
        "query": q,
        "used_query": used_query,
        "scores": scores,
        "snippets": [(d.page_content[:200] if getattr(d, "page_content", None) else "") for d in docs],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    raw_query = normalize(req.query)
    if not raw_query:
        return {"answer": "Ask a question.", "sources": []}

    # 0) Production junk guards
    if looks_like_id_or_code(raw_query):
        return {
            "answer": "I couldn’t understand that. Please rephrase (e.g., 'what is FRS', 'how to do eKYC', 'How many beneficiaries per Aadhaar?').",
            "sources": [],
        }

    if is_gibberish(raw_query):
        return {
            "answer": "I couldn’t understand that. Please rephrase (e.g., 'what is FRS', 'how to do eKYC', 'THR distribution').",
            "sources": [],
        }

    # 1) Fast defs (but do NOT hijack explain/procedure queries)
    fast = maybe_fast_def(raw_query)
    if fast and not is_explainish(raw_query):
        return {"answer": fast, "sources": ["builtin"]}

    # 2) Retrieval (robust: try raw + expanded, dynamic gating)
    raw_words = raw_query.split()
    is_short = len(raw_words) <= 3

    k = 12 if len(raw_words) <= 2 else (8 if len(raw_words) <= 3 else 5)

    used_query, docs, scores = pick_best_retrieval(raw_query, k=k)

    if not docs:
        return {"answer": "This information is not available in the PT FAQ document.", "sources": []}

    best = min(scores) if scores else 999.0
    max_dist_ok = (MAX_DISTANCE_OK + SHORT_QUERY_DISTANCE_BONUS) if is_short else MAX_DISTANCE_OK

    min_chars_needed = 40 if is_short else 220
    if weak_retrieval(docs, min_total_chars=min_chars_needed):
        return {"answer": "This information is not available in the PT FAQ document.", "sources": []}

    if best > max_dist_ok:
        return {"answer": "This information is not available in the PT FAQ document.", "sources": []}

    # 2.1) Retrieval-aware "unknown acronym" guard for definition queries
    unknown_term = extract_unknown_term_if_definition(raw_query)
    if unknown_term:
        term = unknown_term.strip()
        acronymish = looks_like_acronymish(term) and term.lower() not in FAST_DEFS
        if acronymish:
            # extra strict for acronymish defs: must have strong retrieval
            if weak_retrieval(docs, 160) or best > (max_dist_ok - 0.05):
                return {"answer": "This information is not available in the PT FAQ document.", "sources": []}

    # 3) Precision snippet mode for:
    #    - ultra-short queries (1–2 tokens)
    #    - numbers/limits questions (avoid paraphrase errors)
    if len(raw_words) <= 2 or is_numbers_intent(raw_query):
        s = snippet_answer(docs, max_docs=2, max_each=900)
        if s:
            return {"answer": s, "sources": top_sources(docs, 1)}
        return {
            "answer": "Please specify what you mean (beneficiary registration, Aadhaar eKYC, face capture/FRS, THR, etc).",
            "sources": [],
        }

    # 4) LLM (grounded), BUT if it times out -> fallback to snippets
    context = build_context(docs, max_chars=2800)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {raw_query}\n\nContext:\n{context}"},
    ]

    try:
        answer = call_ollama(messages).strip()
        if not answer:
            answer = "This information is not available in the PT FAQ document."
        return {"answer": answer, "sources": top_sources(docs, 3)}
    except Exception:
        s = snippet_answer(docs, max_docs=2, max_each=900)
        if s:
            return {"answer": s, "sources": top_sources(docs, 1)}
        return {"answer": "This information is not available in the PT FAQ document.", "sources": []}