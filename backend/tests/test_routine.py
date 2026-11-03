import pytest

from services import conversation_service, rag_service
from services.rag_service import (
    KB_DIR,
    ROUTINE_STEPS_PRIORITY,
    _parse_listing,
    _parse_routine_step,
    _parse_routine_time,
    _parse_usage,
)

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
    usage = _parse_usage(text, path.name)
    assert usage["routine_time"] in ("AM", "PM", "AM & PM")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("both (morning and evening)", "AM & PM"),
        ("Morning and evening", "AM & PM"),
        ("Morning or evening (or both, if tolerated)", "AM & PM"),
        ("Evening only (1-2x per week to start)", "PM"),
        ("evening preferred (start 2-3 nights). If used in the morning, follow with SPF.", "PM"),
        ("Morning (last step before makeup)", "AM"),
    ],
)
def test_routine_time_mapping(value, expected):
    assert _parse_routine_time(value) == expected


def test_parse_usage_requires_amount():
    text = "Routine time: Morning\nFrequency: daily\nHow to use: Apply."
    with pytest.raises(ValueError, match="Amount"):
        _parse_usage(text, "x.txt")


def _hit(asin, step):
    return {"doc": f"Product {asin}", "meta": {"asin": asin, "routine_step": step, "price": "~$10", "price_usd": 10.0, "targets": "acne",
        "routine_time": "PM", "amount": "a pea-sized amount", "frequency": "nightly", "how_to_use": "Apply to dry skin.",
    }}


def test_group_by_step_keeps_rank_order_and_routine_order():
    ranked = [_hit("spf1", "spf"), _hit("cl1", "cleanser"), _hit("spf2", "spf"), _hit("spf3", "spf")]
    groups = rag_service.group_by_step(ranked, per_step=2)
    assert list(groups) == ["cleanser", "spf"]
    assert [h["meta"]["asin"] for h in groups["spf"]] == ["spf1", "spf2"]


def test_recommend_marks_primary_and_drops_unknown(monkeypatch):
    candidates = {"cleanser": [_hit("cl1", "cleanser"), _hit("cl2", "cleanser")], "spf": [_hit("spf1", "spf")]}
    # The model's own usage fields are ignored in favor of the KB's.
    item = {"product_name": "x", "brand": "b", "why_recommended": "w", "routine_time": "AM", "amount": "lots",
            "key_ingredients": ["Ceramides 1, 3, 6-II", "Glycerin"],
            "ingredient_tags": ["Contains niacinamide for brightening", "Niacinamide 10%"]}
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
    assert (recs[0].routine_time, recs[0].amount) == ("PM", "a pea-sized amount")
    assert recs[0].ingredient_tags == ["Niacinamide", "Ceramides", "Glycerin"]


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Niacinamide 10%", "Niacinamide"),
        ("Ceramides (1, 3, 6-II)", "Ceramides"),
        ("Zinc PCA 1%", "Zinc PCA"),
        ("spf30", "SPF 30"),
        ("hyaluronic acid", "Hyaluronic Acid"),
        ("Contains niacinamide for brightening", ""),  # a description, not a name
    ],
)
def test_short_tag(text, tag):
    assert conversation_service._short_tag(text) == tag


def test_ingredient_tags_dedupes_caps_and_tops_up():
    tags = conversation_service._ingredient_tags(
        ["niacinamide", "Niacinamide 10%", "a long made up description here"],
        ["Zinc PCA 1%", "Glycerin", "Squalane"],
    )
    assert tags == ["Niacinamide", "Zinc PCA", "Glycerin"]
    assert conversation_service._ingredient_tags(None, None) == []


def test_ingredient_tags_skips_contained_duplicates():
    assert conversation_service._ingredient_tags(["Hydrocolloid", "Medical-grade hydrocolloid"], []) == ["Hydrocolloid"]
