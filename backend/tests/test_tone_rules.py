from pathlib import Path

import pytest

from services import tone_rules
from services.rag_service import KB_DIR, _parse_monk_range, _parse_tone_fields

PRODUCTS = sorted((KB_DIR / "products").glob("*.txt"))


def _meta(filename: str) -> dict:
    text = (KB_DIR / "products" / filename).read_text(encoding="utf-8")
    return _parse_tone_fields(text, filename)


@pytest.mark.parametrize(
    "tone, band",
    [(0, None), (1, "light"), (3, "light"), (4, "medium"), (6, "medium"), (7, "deep"), (10, "deep")],
)
def test_tone_band_boundaries(tone, band):
    assert tone_rules.tone_band(tone) == band


def test_deep_tone_excludes_sunscreen_with_cast():
    tinted_mineral = _meta("la_roche_posay_spf50.txt")
    assert tone_rules.is_excluded(tinted_mineral, 8)
    assert not tone_rules.is_excluded(tinted_mineral, 2)


def test_visible_cast_sunscreen_only_for_light_tones():
    sheer_zinc = _meta("neutrogena_sheer_zinc_spf50.txt")
    assert not tone_rules.is_excluded(sheer_zinc, 3)
    assert tone_rules.is_excluded(sheer_zinc, 4)


def test_invisible_sunscreen_allowed_for_every_tone():
    unseen = _meta("supergoop_unseen_spf40.txt")
    assert not any(tone_rules.is_excluded(unseen, tone) for tone in range(1, 11))


def test_unknown_tone_applies_no_rules():
    meta = _meta("neutrogena_salicylic_cleanser.txt")
    assert not tone_rules.is_excluded(meta, 0)
    assert tone_rules.score(meta, 0.8, 0) == 0.8


def test_best_match_outranks_merely_suitable_for_deep_tone():
    made_for_deep = _meta("black_girl_sunscreen_spf30.txt")
    hybrid = _meta("eltamd_uv_clear_spf46.txt")
    assert tone_rules.score(made_for_deep, 0.8, 6) > tone_rules.score(hybrid, 0.8, 6)


def test_deep_tone_penalizes_high_irritation():
    harsh = _meta("panoxyl_acne_foaming_wash_10.txt")
    gentle = _meta("cetaphil_gentle_cleanser.txt")
    assert tone_rules.score(harsh, 0.8, 9) < tone_rules.score(gentle, 0.8, 9)
    # On light skin the same wash is a best match and carries no irritation penalty.
    assert tone_rules.score(harsh, 0.8, 2) == pytest.approx(0.8 + tone_rules.BEST_MATCH_BOOST)


@pytest.mark.parametrize("path", PRODUCTS, ids=lambda p: p.name)
def test_every_product_has_tone_fields(path: Path):
    text = path.read_text(encoding="utf-8")
    fields = _parse_tone_fields(text, path.name)
    suitable_min, suitable_max = _parse_monk_range(text, path.name)
    # "Best" must sit inside "suitable", or the hard filter would hide a best match.
    assert suitable_min <= fields["monk_best_min"] <= fields["monk_best_max"] <= suitable_max
    assert fields["targets"]
