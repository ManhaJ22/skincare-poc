"""
Deterministic Monk-tone rules for product ranking.

Tones are grouped into bands (light 1-3, medium 4-6, deep 7-10). Each band can hard-exclude
products (e.g. sunscreens that leave a cast on deep skin) and nudges the semantic similarity
score up or down. Product metadata comes from the structured lines in each KB product file
(see rag_service.init_collection).
"""

# Similarity scores from ada-002 sit in a narrow band (~0.7-0.9), so adjustments are small.
BEST_MATCH_BOOST = 0.04
PRIORITY_TARGET_BOOST = 0.03

BAND_RULES = {
    "light": {
        "priority_targets": ["redness", "sensitivity"],
        "excluded_casts": [],
        "cast_penalty": {},
        "irritation_penalty": {},
        "guidance": (
            "Lighter skin (Monk 1-3) shows redness and sun damage readily and tends to be more reactive. "
            "Mineral and tinted sunscreens are fine here; favor calming, barrier-friendly products."
        ),
    },
    "medium": {
        "priority_targets": ["hyperpigmentation"],
        "excluded_casts": ["visible"],
        "cast_penalty": {"slight": 0.03},
        "irritation_penalty": {"high": 0.03},
        "guidance": (
            "Medium skin (Monk 4-6) is prone to post-inflammatory hyperpigmentation (PIH), so dark marks "
            "after breakouts or irritation can linger. Favor pigment-fading actives and gentle formulas, "
            "and sunscreens that blend without a cast."
        ),
    },
    "deep": {
        "priority_targets": ["hyperpigmentation"],
        "excluded_casts": ["slight", "visible"],
        "cast_penalty": {},
        "irritation_penalty": {"high": 0.06, "medium": 0.02},
        "guidance": (
            "Deeper skin (Monk 7-10) has the highest risk of PIH, and irritation itself can cause new dark marks. "
            "Favor low-irritation, pigment-fading actives and sunscreens with no white or grey cast."
        ),
    },
}


def tone_band(monk_tone: int | None) -> str | None:
    """Map a Monk tone to its band. 0/None means the tone is unknown, so no band rules apply."""
    if not monk_tone:
        return None
    if monk_tone <= 3:
        return "light"
    if monk_tone <= 6:
        return "medium"
    return "deep"


def is_excluded(meta: dict, monk_tone: int | None) -> bool:
    band = tone_band(monk_tone)
    if band is None:
        return False
    return meta.get("white_cast") in BAND_RULES[band]["excluded_casts"]


def score(meta: dict, similarity: float, monk_tone: int | None) -> float:
    """Semantic similarity adjusted for how well the product fits the user's tone."""
    band = tone_band(monk_tone)
    if band is None:
        return similarity

    rules = BAND_RULES[band]
    adjusted = similarity
    if meta.get("monk_best_min", 0) <= monk_tone <= meta.get("monk_best_max", 0):
        adjusted += BEST_MATCH_BOOST
    targets = meta.get("targets", "")
    if any(target in targets for target in rules["priority_targets"]):
        adjusted += PRIORITY_TARGET_BOOST
    adjusted -= rules["cast_penalty"].get(meta.get("white_cast"), 0)
    adjusted -= rules["irritation_penalty"].get(meta.get("irritation"), 0)
    return adjusted


def guidance(monk_tone: int | None) -> str:
    band = tone_band(monk_tone)
    return BAND_RULES[band]["guidance"] if band else "Skin tone unknown; no tone-specific guidance."
