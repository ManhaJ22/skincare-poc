from models.schemas import SkinProfile
from services import conversation_service
from services.conversation_service import _has_required_fields
from services.rag_service import build_query


def complete(**overrides):
    fields = dict(
        monk_tone=6, confirmed_concerns=["dark marks"], allergies=[], products_liked=[],
        products_disliked=[], budget_monthly=50, fragrance_free=True,
    )
    return SkinProfile(**{**fields, **overrides})


def test_unasked_fields_default_to_none():
    profile = SkinProfile()
    assert profile.allergies is None
    assert profile.products_liked is None and profile.products_disliked is None


def test_ready_only_once_everything_is_answered():
    assert _has_required_fields(complete())


def test_not_ready_until_sensitivities_asked():
    assert not _has_required_fields(complete(allergies=None))


def test_not_ready_until_product_history_asked():
    assert not _has_required_fields(complete(products_liked=None, products_disliked=None))


def test_either_product_list_counts_as_answered():
    assert _has_required_fields(complete(products_liked=None, products_disliked=["niacinamide serum"]))


def test_query_avoids_disliked_and_mentions_liked():
    query = build_query(complete(allergies=["fragrance"], products_disliked=["niacinamide"], products_liked=["CeraVe cleanser"]))
    assert "avoid: fragrance, niacinamide" in query
    assert "likes: CeraVe cleanser" in query


def _reply(monkeypatch, text, flag, profile):
    fake = {"assistant_response": text, "updated_profile": profile.model_dump(), "is_ready_to_recommend": flag}
    monkeypatch.setattr(conversation_service, "_chat_json", lambda *a, **k: fake)
    monkeypatch.setattr(conversation_service, "typical_routine_costs", lambda: {})
    _, _, ready = conversation_service.get_next_response([], profile, "hi", [])
    return ready


def test_sign_off_without_flag_counts_as_ready(monkeypatch):
    assert _reply(monkeypatch, "I'll put your routine together now. Patch test new products!", False, complete())


def test_follow_up_question_without_flag_is_not_ready(monkeypatch):
    assert not _reply(monkeypatch, "What would you love your skin to feel like?", False, complete())


def test_flag_ignored_while_answers_missing(monkeypatch):
    assert not _reply(monkeypatch, "Building your routine now!", True, complete(products_liked=None, products_disliked=None))
