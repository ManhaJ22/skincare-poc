import pytest

from models.schemas import RecommendationResponse
from services import budget
from services.rag_service import KB_DIR, _parse_listing, _parse_price_usd


def rec(asin, step, price, targets=()):
    return RecommendationResponse(
        product_name=asin, brand="b", amazon_asin=asin, routine_step=step,
        why_recommended="w", key_ingredients=[], price_usd=price, targets=list(targets),
    )


def routine():
    # Ranked best-first within each step, as recommend() returns them.
    return [
        rec("cl1", "cleanser", 14), rec("cl2", "cleanser", 9),
        rec("toner1", "toner", 23),
        rec("serum1", "serum", 14, ["hyperpigmentation"]),
        rec("tx1", "treatment", 12, ["acne"]),
        rec("mo1", "moisturizer", 20), rec("mo2", "moisturizer", 12),
        rec("spf1", "spf", 15), rec("spf2", "spf", 38),
        rec("mask1", "mask", 12),
    ]


def picked(recs):
    return {r.amazon_asin for r in recs if not r.is_alternative and not r.budget_left_out}


@pytest.mark.parametrize("text, value", [("~$14", 14), ("$25-$37", 31), ("$6-$12", 9), ("$12.50", 12.5), ("", None)])
def test_parse_price_usd(text, value):
    assert _parse_price_usd(text) == value


def test_every_product_has_a_price():
    for path in (KB_DIR / "products").glob("*.txt"):
        assert _parse_listing(path.read_text(encoding="utf-8"), path.name)["price_usd"] > 0, path.name


def test_tight_budget_keeps_essentials_first():
    recs, summary = budget.fit_routine(routine(), 30, ["dark marks"])
    kept = picked(recs)
    assert {r.routine_step for r in recs if r.amazon_asin in kept} <= {"spf", "cleanser", "moisturizer"}
    assert "spf1" in kept
    assert summary.total <= 40
    assert "mask" in summary.left_out_steps and "toner" in summary.left_out_steps


def test_cheaper_alternative_swapped_in_to_fit_another_essential():
    # Cap is $38. Top picks (cl1 14 + mo1 20 + spf1 15 = 49) don't fit, and neither does
    # cl1 + mo2 + spf1 = 41, but cl2 9 + mo2 12 + spf1 15 = 36 keeps all three essentials.
    recs, summary = budget.fit_routine(routine(), 28, [])
    kept = picked(recs)
    assert kept == {"cl2", "mo2", "spf1"}
    assert summary.status == "within"


def test_concern_treatment_beats_extras():
    recs, _ = budget.fit_routine(routine(), 60, ["dark marks"])
    kept = picked(recs)
    assert "serum1" in kept  # targets their concern
    assert "toner1" not in kept and "mask1" not in kept


def test_generous_budget_keeps_every_top_pick():
    recs, summary = budget.fit_routine(routine(), 200, [])
    assert picked(recs) == {"cl1", "toner1", "serum1", "tx1", "mo1", "spf1", "mask1"}
    assert summary.left_out_steps == []
    assert summary.status == "under"
    assert summary.total == summary.full_total == 110


def test_no_budget_keeps_top_picks():
    recs, summary = budget.fit_routine(routine(), None, [])
    assert summary.status == "no_budget"
    assert summary.total == 110


def test_nothing_fits_keeps_cheapest_most_important():
    recs, summary = budget.fit_routine([rec("spf1", "spf", 40), rec("cl1", "cleanser", 30)], 5, [])
    assert picked(recs) == {"spf1"}
    assert summary.status == "over"


def test_concern_wording_maps_to_targets():
    assert budget.concern_targets(["dark marks", "breakouts"]) == {"hyperpigmentation", "acne"}


def test_typical_costs():
    costs = budget.typical_costs([("cleanser", 10), ("cleanser", 20), ("moisturizer", 20), ("spf", 15), ("mask", 12)])
    assert costs == {"typical_basic_low": 45, "typical_basic": 50, "typical_full": 60}
