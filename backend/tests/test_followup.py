from fastapi.testclient import TestClient

import main
from services import conversation_service, rag_service

ROUTINE = [{
    "product_name": "Niacinamide 10% + Zinc 1%", "brand": "The Ordinary", "amazon_asin": "B01MDTVZTZ",
    "routine_step": "serum", "why_recommended": "w", "key_ingredients": ["Niacinamide"],
}]


def _post(routine):
    body = {"conversation_history": [], "skin_profile": {"monk_tone": 6}, "user_message": "What does niacinamide do?"}
    if routine is not None:
        body["routine"] = routine
    return TestClient(main.app).post("/chat", json=body)


def test_followup_answers_without_rebuilding(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag_service, "retrieve_for_question", lambda q, asins: seen.update(asins=asins) or ["doc"])
    monkeypatch.setattr(conversation_service, "answer_followup", lambda *a: "It helps fade dark marks.")
    # Would fail loudly if the profile-gathering path ran.
    monkeypatch.setattr(conversation_service, "get_next_response", lambda *a: (_ for _ in ()).throw(AssertionError))

    res = _post(ROUTINE)

    assert res.status_code == 200
    data = res.json()
    assert data["assistant_response"] == "It helps fade dark marks."
    assert data["recommendations"] == [] and data["is_complete"] is True
    assert seen["asins"] == ["B01MDTVZTZ"]


def test_without_routine_uses_profile_conversation(monkeypatch):
    monkeypatch.setattr(rag_service, "retrieve_relevant_products", lambda profile: [])
    monkeypatch.setattr(
        conversation_service, "get_next_response", lambda h, p, m, c: ("Any allergies?", p, False)
    )
    assert _post(None).json()["assistant_response"] == "Any allergies?"
