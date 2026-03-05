# backend/app.py

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rapidfuzz import process, fuzz

# ================= CONFIG =================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "12"))

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
COLLECTION = os.getenv("CHROMA_COLLECTION", "pt_faq_v2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chroma distance: lower is better
MAX_DISTANCE_OK = float(os.getenv("MAX_DISTANCE_OK", "0.95"))
SHORT_QUERY_DISTANCE_BONUS = float(os.getenv("SHORT_QUERY_DISTANCE_BONUS", "0.15"))  # ✅ tightened
SHORT_QUERY_MAX_THRESHOLD = float(os.getenv("SHORT_QUERY_MAX_THRESHOLD", "1.05"))    # ✅ hard cap
SHORT_QUERY_MAX_TOKENS = int(os.getenv("SHORT_QUERY_MAX_TOKENS", "4"))

BASE_DIR = Path(__file__).resolve().parent
FAQ_INDEX_PATH = Path(os.getenv("FAQ_INDEX_PATH", str(BASE_DIR / "faq_index.json")))

BUILD = os.getenv("BUILD", "2026-02-26-ptfaq-v2-hybrid-startup-acronym")

# ================= APP =================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOGGING =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ptfaq")

def log(payload: Dict[str, Any]) -> None:
    payload["ts"] = datetime.utcnow().isoformat()
    logger.info(json.dumps(payload, ensure_ascii=False))

# ================= NORMALIZATION =================

_WS = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """
    Match faq_index.json by_question_norm keys:
    - lowercase
    - preserve hyphen '-' and slash '/'
    - normalize unicode dashes to '-'
    - remove other punctuation
    - collapse whitespace
    - remove spaces around '-' and '/'
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"[^a-z0-9\s/-]+", " ", s)
    s = _WS.sub(" ", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s

def tokenize(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (q or "").lower())

# ================= GLOBALS (startup init) =================

embeddings: Optional[HuggingFaceEmbeddings] = None
vectordb: Optional[Chroma] = None

# ================= FAQ INDEX =================

FAQ_BY_Q: Dict[str, Dict[str, Any]] = {}
ACRONYM_MAP: Dict[str, str] = {}

def load_faq_index() -> None:
    global FAQ_BY_Q, ACRONYM_MAP

    FAQ_BY_Q = {}
    ACRONYM_MAP = {}

    try:
        with open(FAQ_INDEX_PATH, "r", encoding="utf-8") as f:
            idx = json.load(f)
            FAQ_BY_Q = idx.get("by_question_norm", {}) or {}
            print(f"[FAQ] loaded {len(FAQ_BY_Q)} deterministic QAs")
    except Exception as e:
        print("[FAQ] failed:", e)
        return

    # Build acronym map automatically from questions like:
    # "What is FRS in Poshan Tracker?", "Full form of THR", etc.
    def build_acronym_map(faq_by_q: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        m: Dict[str, str] = {}
        for rec in faq_by_q.values():
            q = (rec.get("question") or "").strip()
            qn = normalize_text(q)
            mm = re.match(r"^(what is|full form of)\s+([a-z]{2,6})\b", qn)
            if mm:
                acr = mm.group(2).lower()
                if acr not in m:
                    m[acr] = q
        return m

    ACRONYM_MAP = build_acronym_map(FAQ_BY_Q)
    print(f"[FAQ] acronym map size: {len(ACRONYM_MAP)}")

def faq_exact_lookup(q: str) -> Optional[Tuple[str, str]]:
    key = normalize_text(q)
    hit = FAQ_BY_Q.get(key)
    if not hit:
        return None

    # IMPORTANT: preserve multiline answers if the index stores them.
    ans = hit.get("answer", "")
    ques = hit.get("question", "")

    # Sometimes older index answers are single-line; leave as-is.
    return ans, ques

# ================= CLARIFICATION =================

CLARIFY_BUCKETS = {
    "registration": [
        "Who can be registered",
        "Required documents",
        "eKYC / Face capture steps",
        "Child 0-6 months registration",
        "Supervisor approval / verification",
    ]
}

CANONICAL = {
    ("registration","who can be registered"):
        "How many kinds of beneficiaries can be registered in the Application?",
    ("registration","required documents"):
        "What documents are required for registration?",
    ("registration","ekyc face capture steps"):
        "How is face capture done during registration?",
}

def _nk(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def parse_followup(q: str) -> Optional[Tuple[str, str]]:
    if ":" in q:
        a, b = q.split(":", 1)
    elif " - " in q:
        a, b = q.split(" - ", 1)
    else:
        return None
    return _nk(a), _nk(b)

def canonicalize(q: str) -> Tuple[str, Optional[str], Optional[str]]:
    p = parse_followup(q)
    if not p:
        return q, None, None
    t, o = p
    canon = CANONICAL.get((t, o))
    if canon:
        return canon, t, o
    return q, t, o

# ================= FUZZY (light, safe) =================

KNOWN = ["registration", "beneficiary", "documents", "ekyc", "frs", "nominee", "thr", "attendance", "tracking"]

def fuzzy_fix(q: str) -> str:
    toks = tokenize(q)
    out = []
    for t in toks:
        hit = process.extractOne(t, KNOWN, scorer=fuzz.WRatio)
        if hit and hit[1] > 90:
            out.append(hit[0])
        else:
            out.append(t)
    return " ".join(out).strip()

# ================= REQUEST =================

class ChatRequest(BaseModel):
    query: str

# ================= HELPERS =================

ACRONYM_QUERY_RE = re.compile(r"^(what is|define|meaning of|full form of)\s+([a-z]{2,6})\b", re.I)

def maybe_rewrite_acronym(raw: str) -> Optional[str]:
    """
    Only rewrite when query is clearly an acronym-definition type.
    Does NOT restrict other questions.
    """
    qn = normalize_text(raw)

    # Case 1: just acronym like "frs"
    if re.fullmatch(r"[a-z]{2,6}", qn):
        return ACRONYM_MAP.get(qn)

    # Case 2: "what is frs" etc.
    m = ACRONYM_QUERY_RE.match(qn)
    if m:
        acr = m.group(2).lower()
        return ACRONYM_MAP.get(acr)

    return None

def extract_answer_from_page_content(page: str) -> str:
    """
    vectors store:
      Q: ...
      A: ...
    return A: part when possible.
    """
    txt = (page or "").strip()
    m = re.search(r"\nA:\s*(.*)$", txt, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return txt

def vector_retrieve(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Use distance gating (lower is better). Apply short-query bonus with cap.
    """
    if vectordb is None:
        return {"ok": False, "reason": "vectordb_not_ready", "hits": []}

    q = (query or "").strip()
    if not q:
        return {"ok": False, "reason": "empty_query", "hits": []}

    pairs = vectordb.similarity_search_with_score(q, k=k)
    hits = [{
        "distance": float(dist),
        "text": doc.page_content,
        "meta": doc.metadata or {}
    } for doc, dist in pairs]

    if not hits:
        return {"ok": False, "reason": "no_results", "hits": []}

    best = hits[0]["distance"]

    tok_len = len(tokenize(q))
    threshold = MAX_DISTANCE_OK
    if tok_len <= SHORT_QUERY_MAX_TOKENS:
        threshold = min(MAX_DISTANCE_OK + SHORT_QUERY_DISTANCE_BONUS, SHORT_QUERY_MAX_THRESHOLD)

    if best > threshold:
        return {"ok": False, "reason": f"too_far(best={best:.3f},thr={threshold:.3f})", "hits": hits}

    return {"ok": True, "reason": f"ok(best={best:.3f},thr={threshold:.3f})", "hits": hits}

# ================= STARTUP =================

@app.on_event("startup")
def startup():
    global embeddings, vectordb
    load_faq_index()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    print(f"[VECTOR] ready collection={COLLECTION} persist={PERSIST_DIR}")

# ================= ROUTES =================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "faq_items": len(FAQ_BY_Q),
        "collection": COLLECTION,
        "persist_dir": PERSIST_DIR,
        "embed_model": EMBED_MODEL,
        "acronyms": len(ACRONYM_MAP),
        "build": BUILD,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    raw = (req.query or "").strip()
    rid = str(uuid4())[:8]

    # 0) Acronym rewrite (only when query is clearly definition-type)
    rewritten = maybe_rewrite_acronym(raw)
    if rewritten:
        raw_for_pipeline = rewritten
    else:
        raw_for_pipeline = raw

    # 1) Canonicalize FIRST (before fuzzy destroys followup markers)
    canonical, topic, opt = canonicalize(raw_for_pipeline)

    # 2) Fuzzy only if NOT a followup
    if not topic:
        canonical = fuzzy_fix(canonical)

    log({"rid": rid, "raw": raw, "rewritten": rewritten, "canonical": canonical, "topic": topic, "opt": opt})

    # 3) Deterministic FAQ exact hit
    hit = faq_exact_lookup(canonical)
    if hit:
        ans, q = hit
        log({"rid": rid, "stage": "faq_exact"})
        return {"answer": ans, "mode": "faq_exact"}

    # 4) Clarify bucket
    if _nk(canonical) == "registration":
        log({"rid": rid, "stage": "clarify"})
        return {
            "mode": "clarify",
            "question": "What exactly do you want to know about REGISTRATION?",
            "options": CLARIFY_BUCKETS["registration"],
        }

    # 5) Vector gated retrieval
    vr = vector_retrieve(canonical, k=4)
    log({"rid": rid, "stage": "vector", "ok": vr["ok"], "reason": vr["reason"],
         "best_distance": (vr["hits"][0]["distance"] if vr["hits"] else None)})

    if not vr["ok"]:
        return {"answer": "This information is not available in the PT FAQ document.", "mode": "not_found"}

    top = vr["hits"][0]
    ans = extract_answer_from_page_content(top["text"])[:2000]  # ✅ longer

    return {
        "answer": ans,
        "mode": "vector",
        "debug": {
            "reason": vr["reason"],
            "best_distance": top["distance"],
            "matched_question": top["meta"].get("question", ""),
            "section_path": top["meta"].get("section_path", ""),
            "tags": top["meta"].get("tags", ""),
        }
    }