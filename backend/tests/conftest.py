import pytest

from services import rag_service


@pytest.fixture(autouse=True)
def no_ingredient_rag(monkeypatch):
    """Endpoint tests never embed queries against OpenAI; tests that care override this."""
    monkeypatch.setattr(rag_service, "retrieve_ingredient_context", lambda query, n_results=4: [])
