from models.schemas import SkinProfile
from services import conversation_service
from services.conversation_service import _has_required_fields
from services.rag_service import build_query


def complete(**overrides):
    fields = dict(
        monk_tone=6, confirmed_concerns=["dark marks"], allergies=[], products_liked=[],
        products_disliked=[], budget_monthly=50, fragrance_free=True, goals=[],
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


# --- Skin goals: always asked once, answering optional ------------------------------------------
from models.schemas import ConversationTurn  # noqa: E402


def test_goals_default_to_not_asked():
    assert SkinProfile().goals is None


def test_not_ready_until_goals_question_asked():
    assert not _has_required_fields(complete(goals=None))
    assert _has_required_fields(complete(goals=[]))  # asked, nothing specific: fine


def test_missing_fields_in_asking_order_goals_before_budget():
    labels = conversation_service.missing_fields(SkinProfile(monk_tone=6))
    keys = ["concerns", "sensitivities", "fragrance", "products", "skin goals", "budget"]
    assert [next(k for k in keys if k in label) for label in labels] == keys
    assert conversation_service.GOALS_QUESTION in labels[4]
    assert conversation_service.missing_fields(complete()) == []


def _run(monkeypatch, history, profile, model_profile, text="Got it! And roughly what budget feels right?"):
    seen = {}

    def fake(messages, temperature):
        seen["system"] = messages[0]["content"]
        return {"assistant_response": text, "updated_profile": model_profile.model_dump(), "is_ready_to_recommend": False}

    monkeypatch.setattr(conversation_service, "_chat_json", fake)
    monkeypatch.setattr(conversation_service, "typical_routine_costs", lambda: {})
    reply, updated, ready = conversation_service.get_next_response(
        history, profile, "lol idk", [], ingredient_context=["[niacinamide knowledge base — SECTION 3]\nfades marks in 8-12 weeks"]
    )
    return seen["system"], updated, ready


def test_ignored_goals_question_counts_as_asked(monkeypatch):
    history = [ConversationTurn(role="assistant", content=f"Love that. {conversation_service.GOALS_QUESTION}")]
    profile = complete(goals=None, budget_monthly=None)
    _, updated, _ = _run(monkeypatch, history, profile, profile)  # model left goals null
    assert updated.goals == []


def test_goals_not_marked_asked_by_other_questions(monkeypatch):
    history = [ConversationTurn(role="assistant", content="Have you tried anything for those dark marks?")]
    profile = complete(goals=None)
    _, updated, ready = _run(monkeypatch, history, profile, profile)
    assert updated.goals is None and not ready


def test_prompt_lists_missing_fields_and_ingredient_science(monkeypatch):
    profile = complete(allergies=None, goals=None)
    system, _, _ = _run(monkeypatch, [], profile, profile)
    assert "STILL UNANSWERED (in order" in system
    unanswered = system.split("STILL UNANSWERED")[1].split("\n")[0]
    assert "sensitivities" in unanswered and "skin goals" in unanswered and "budget" not in unanswered
    assert "MISSED OR UNCLEAR ANSWERS" in system
    assert "fades marks in 8-12 weeks" in system and "Ingredient science" in system


def test_prompt_without_ingredient_chunks_has_no_empty_section(monkeypatch):
    seen = {}
    monkeypatch.setattr(conversation_service, "_chat_json",
                        lambda m, t: seen.update(system=m[0]["content"]) or {"assistant_response": "Hi?"})
    monkeypatch.setattr(conversation_service, "typical_routine_costs", lambda: {})
    conversation_service.get_next_response([], SkinProfile(monk_tone=6), "hi", [])
    assert "Ingredient science" not in seen["system"]


# --- Awkward moments: piggybacked "none" answers and deferred sign-offs ---------------------------
def _turn(monkeypatch, history, before, after, text="Got it! What else?", flag=False):
    from models.schemas import ConversationTurn
    fake = {"assistant_response": text, "updated_profile": after.model_dump(), "is_ready_to_recommend": flag}
    monkeypatch.setattr(conversation_service, "_chat_json", lambda *a, **k: fake)
    monkeypatch.setattr(conversation_service, "typical_routine_costs", lambda: {})
    turns = [ConversationTurn(role="assistant", content=history)]
    return conversation_service.get_next_response(turns, before, "nope nothing really reacts", [])


def test_none_to_allergy_question_does_not_also_answer_products(monkeypatch):
    before = SkinProfile(monk_tone=6, confirmed_concerns=["dark marks"])
    after = before.model_copy(update={"allergies": [], "products_liked": [], "products_disliked": []})
    _, profile, _ = _turn(monkeypatch, "Does anything make your skin sting or turn red?", before, after)
    assert profile.allergies == []
    assert profile.products_liked is None and profile.products_disliked is None


def test_none_to_combined_question_keeps_both(monkeypatch):
    before = SkinProfile(monk_tone=6, confirmed_concerns=["dark marks"])
    after = before.model_copy(update={"allergies": [], "products_liked": [], "products_disliked": []})
    _, profile, _ = _turn(monkeypatch, "Have you tried any products, and does anything react?", before, after)
    assert profile.products_liked == [] and profile.products_disliked == []


def test_named_products_are_never_dropped(monkeypatch):
    before = SkinProfile(monk_tone=6, confirmed_concerns=["dark marks"])
    after = before.model_copy(update={"allergies": [], "products_liked": ["Ordinary niacinamide"]})
    _, profile, _ = _turn(monkeypatch, "Does anything make your skin sting?", before, after)
    assert profile.products_liked == ["Ordinary niacinamide"]


def test_in_touch_sign_off_is_replaced(monkeypatch):
    reply, _, ready = _turn(
        monkeypatch, "What's your budget?", complete(), complete(),
        text="Perfect, I'll be in touch soon with your routine! Remember to patch test.", flag=True,
    )
    assert ready and reply == conversation_service.READY_MESSAGE
    assert "below" in reply


def test_goals_question_is_exact_and_last():
    fix = conversation_service._exact_goals_question
    q = conversation_service.GOALS_QUESTION
    assert fix("Thanks! Let's talk about what you hope to achieve with your skincare. Any specific goals?") == f"Thanks! {q}"
    assert fix("Got it. What do you hope to achieve with your new skincare? Any goals, or just overall?") == f"Got it. {q}"
    assert fix("What do you hope to achieve?") == q
    assert fix("What's your budget?") == "What's your budget?"


def test_premature_sign_off_asks_for_missing_budget(monkeypatch):
    before = complete(budget_monthly=None)
    reply, _, ready = _turn(monkeypatch, "Roughly what would you like to spend?", before, before,
                            text="No worries! I have what I need now, and your routine will appear just below in a moment.")
    assert not ready
    assert "?" in reply and "spend" in reply and "below" not in reply


def test_listening_reply_without_sign_off_is_left_alone(monkeypatch):
    before = complete(budget_monthly=None)
    reply, _, _ = _turn(monkeypatch, "How's your week?", before, before, text="Ugh, that sounds exhausting. Take a breath.")
    assert reply == "Ugh, that sounds exhausting. Take a breath."
