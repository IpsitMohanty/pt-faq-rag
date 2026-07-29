"""Smoke tests at the service's seams: the pure query-normalization
functions, and the /health and /chat endpoints with the vector store
and LLM mocked out. Not an evaluation suite (that discipline lives in
rag-ingestion-evaluation, a different repo) -- this exists to prove the
guardrail pipeline actually behaves as documented (exact-match short-
circuits before vector retrieval, clarification before guessing,
graceful degradation when the vector store or Ollama aren't available),
not to grade retrieval quality.

No real Chroma, HuggingFace, or Ollama required: app.py's startup event
(which builds those) is never triggered, and vectordb/llm are set
directly on the module for the tests that need a specific behavior.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def reset_module_state():
    """app.py keeps embeddings/vectordb/llm/FAQ_BY_Q/ACRONYM_MAP as
    module globals, mutated by the startup event and by /chat itself.
    Reset them before and after every test so tests can't leak state
    into each other regardless of order."""
    app_module.FAQ_BY_Q = {}
    app_module.ACRONYM_MAP = {}
    app_module.vectordb = None
    app_module.llm = None
    yield
    app_module.FAQ_BY_Q = {}
    app_module.ACRONYM_MAP = {}
    app_module.vectordb = None
    app_module.llm = None


@pytest.fixture
def client():
    return TestClient(app_module.app)


# --- pure normalization / canonicalization functions ---------------------

class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert app_module.normalize_text("What is FRS?") == "what is frs"

    def test_preserves_hyphen_and_slash(self):
        assert app_module.normalize_text("eKYC/Face-capture") == "ekyc/face-capture"

    def test_collapses_whitespace(self):
        assert app_module.normalize_text("  a   b  ") == "a b"


class TestParseFollowup:
    def test_splits_on_colon(self):
        assert app_module.parse_followup("registration: required documents") == (
            "registration", "required documents",
        )

    def test_splits_on_dash_with_spaces(self):
        assert app_module.parse_followup("registration - required documents") == (
            "registration", "required documents",
        )

    def test_returns_none_without_a_followup_marker(self):
        assert app_module.parse_followup("what is frs") is None


class TestCanonicalize:
    def test_maps_known_followup_to_canonical_question(self):
        canon, topic, opt = app_module.canonicalize("registration: who can be registered")
        assert canon == "How many kinds of beneficiaries can be registered in the Application?"
        assert topic == "registration"

    def test_unknown_followup_falls_through_unchanged(self):
        canon, topic, opt = app_module.canonicalize("registration: something unheard of")
        assert topic == "registration"
        assert opt == "something unheard of"


class TestMaybeRewriteAcronym:
    def test_bare_acronym_expands_via_acronym_map(self):
        app_module.ACRONYM_MAP = {"frs": "What is FRS in Poshan Tracker?"}
        assert app_module.maybe_rewrite_acronym("frs") == "What is FRS in Poshan Tracker?"

    def test_what_is_phrasing_also_expands(self):
        app_module.ACRONYM_MAP = {"thr": "What is THR in Poshan Tracker?"}
        assert app_module.maybe_rewrite_acronym("what is thr") == "What is THR in Poshan Tracker?"

    def test_unrelated_query_is_not_rewritten(self):
        app_module.ACRONYM_MAP = {"frs": "What is FRS in Poshan Tracker?"}
        assert app_module.maybe_rewrite_acronym("how do I register a beneficiary") is None


# --- /health -------------------------------------------------------------

def test_health_reports_status_and_counts_without_any_backend_running(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["faq_items"] == 0  # nothing loaded -- startup never ran


# --- /chat: the guardrail pipeline's documented modes ---------------------

def test_chat_exact_faq_match_short_circuits_before_vector_retrieval(client):
    app_module.FAQ_BY_Q = {
        "what is frs": {"question": "What is FRS?", "answer": "Facial Recognition System."},
    }
    response = client.post("/chat", json={"query": "What is FRS?"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "faq_exact"
    assert body["answer"] == "Facial Recognition System."


def test_chat_bare_registration_query_returns_clarify_not_a_guess(client):
    response = client.post("/chat", json={"query": "registration"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "clarify"
    assert "options" in body and len(body["options"]) > 0


def test_chat_degrades_to_not_found_when_vector_store_is_unavailable(client):
    """The documented fallback: no FAQ hit, not a clarify bucket, and no
    vector store running (vectordb is None, exactly as it is before
    startup / when Chroma can't be reached) -- the service must respond
    cleanly, never raise, and say so rather than fabricate an answer."""
    response = client.post("/chat", json={"query": "a completely unrelated question"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "not_found"
    assert "answer" in body


def test_vector_retrieve_gates_on_distance_threshold():
    """Distance gating (app_module.vector_retrieve) with a fake vector
    store standing in for Chroma -- verifies the guardrail's actual
    boundary condition, not just that /chat doesn't crash."""

    class FakeDoc:
        def __init__(self, text):
            self.page_content = text
            self.metadata = {}

    class FakeVectorDB:
        def __init__(self, distance):
            self._distance = distance

        def similarity_search_with_score(self, query, k=4):
            return [(FakeDoc("Q: x\nA: relevant answer"), self._distance)]

    app_module.vectordb = FakeVectorDB(distance=0.5)
    result = app_module.vector_retrieve("a normal length question about something")
    assert result["ok"] is True

    app_module.vectordb = FakeVectorDB(distance=0.99)
    result = app_module.vector_retrieve("a normal length question about something")
    assert result["ok"] is False
