from fastapi.testclient import TestClient
from openai import OpenAIError

import main
from models.schemas import RecommendationResponse
from services import conversation_service, rag_service


def _rec(asin, step, price):
    return RecommendationResponse(product_name=asin, brand="b", amazon_asin=asin, routine_step=step,
                                  why_recommended="w", key_ingredients=[], price=f"${price}", price_usd=price)


def test_recommendations_builds_budget_fitted_routine(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag_service, "retrieve_routine_candidates",
                        lambda profile, per_step=2, environment=None: seen.setdefault("profile", profile) and {})
    monkeypatch.setattr(conversation_service, "recommend", lambda profile, candidates, env=None: [
        _rec("c1", "cleanser", 12), _rec("m1", "moisturizer", 15), _rec("s1", "spf", 14),
    ])
    profile = {"monk_tone": 7, "observed_concerns": ["dark spots"], "skin_type": "oily",
               "confirmed_concerns": ["acne"], "allergies": ["parabens"], "budget_monthly": 40,
               "fragrance_free": True, "products_liked": [], "products_disliked": []}

    res = TestClient(main.app).post("/recommendations", json={"skin_profile": profile})

    assert res.status_code == 200
    body = res.json()
    assert [r["amazon_asin"] for r in body["recommendations"]] == ["c1", "m1", "s1"]
    assert body["budget"]["budget"] == 40 and body["budget"]["total"] == 41
    assert seen["profile"].monk_tone == 7 and seen["profile"].allergies == ["parabens"]


def test_recommendations_returns_502_on_llm_failure(monkeypatch):
    monkeypatch.setattr(rag_service, "retrieve_routine_candidates", lambda *a, **k: {})

    def boom(*a, **k):
        raise OpenAIError("down")
    monkeypatch.setattr(conversation_service, "recommend", boom)

    res = TestClient(main.app).post("/recommendations", json={"skin_profile": {"monk_tone": 5}})
    assert res.status_code == 502
