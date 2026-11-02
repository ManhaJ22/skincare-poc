import pytest

from services import conversation_service, rag_service
from services.rag_service import KB_DIR, ROUTINE_STEPS_PRIORITY, _parse_listing, _parse_routine_step

PRODUCTS = sorted((KB_DIR / "products").glob("*.txt"))


@pytest.mark.parametrize(
    "label, step",
    [
        ("Routine step: Spot treatment", "spot treatment"),
        ("Routine step: Eye cream", "eye cream"),
        ("Routine step: Treatment / Serum", "treatment"),
        ("Routine step: Moisturizer + SPF (combined)", "spf"),
        ("Routine step: Serum / Essence", "serum"),
        ("ROUTINE STEP: cleanser", "cleanser"),
    ],
)
def test_routine_step_keywords(label, step):
    assert _parse_routine_step(label) == step


@pytest.mark.parametrize("path", PRODUCTS, ids=lambda p: p.name)
def test_every_product_has_known_step_and_listing(path):
    text = path.read_text(encoding="utf-8")
    assert _parse_routine_step(text) in ROUTINE_STEPS_PRIORITY
    listing = _parse_listing(text, path.name)
    assert listing["asin"] and listing["price"].startswith(("$", "~$"))


def _hit(asin, step):
    return {"doc": f"Product {asin}", "meta": {"asin": asin, "routine_step": step, "price": "~$10"}}


def test_group_by_step_keeps_rank_order_and_routine_order():
    ranked = [_hit("spf1", "spf"), _hit("cl1", "cleanser"), _hit("spf2", "spf"), _hit("spf3", "spf")]
    groups = rag_service.group_by_step(ranked, per_step=2)
    assert list(groups) == ["cleanser", "spf"]
    assert [h["meta"]["asin"] for h in groups["spf"]] == ["spf1", "spf2"]


def test_recommend_marks_primary_and_drops_unknown(monkeypatch):
    candidates = {"cleanser": [_hit("cl1", "cleanser"), _hit("cl2", "cleanser")], "spf": [_hit("spf1", "spf")]}
    item = {"product_name": "x", "brand": "b", "why_recommended": "w", "key_ingredients": []}
    # Model returns them out of order, skips cl1 (conflict), and invents an ASIN.
    fake = {"recommendations": [
        {**item, "amazon_asin": "spf1"},
        {**item, "amazon_asin": "cl2"},
        {**item, "amazon_asin": "MADE_UP"},
    ]}
    monkeypatch.setattr(conversation_service, "_chat_json", lambda *args, **kwargs: fake)

    recs = conversation_service.recommend(conversation_service.SkinProfile(monk_tone=5), candidates)

    assert [(r.amazon_asin, r.routine_step, r.is_alternative) for r in recs] == [
        ("cl2", "cleanser", False),  # cl1 was skipped, so cl2 becomes the primary
        ("spf1", "spf", False),
    ]
    assert recs[0].price == "~$10"
