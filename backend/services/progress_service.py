"""Predicted progress: when each product in the routine should start showing its effect.

Grounded in the ingredient knowledge base (the "ONSET TIMELINES", Monk-scale and severity
sections of backend/knowledge_base/ingredients_progress_kb/), retrieved per key ingredient.
"""
from models.schemas import Milestone, RecommendationResponse, SkinProfile
from services import conversation_service, rag_service

TIMELINE_TEMPERATURE = 0.2
MAX_WEEKS = 52
CHUNKS_PER_INGREDIENT = 3
MAX_INGREDIENTS = 8

TIMELINE_PROMPT = """You estimate when each product in someone's skincare routine should start showing results, for a progress tracker.

Use ONLY the timelines in the knowledge base text below. Apply its skin tone (Monk scale) and severity notes where they fit (e.g. deeper or longer-standing dark marks take longer). Never promise results.

For every product, return:
- asin: copied exactly from the routine
- ingredient: the active doing the work, in plain words ("Niacinamide")
- concern: the concern it targets, in everyday words ("dark marks", "shine", "breakouts", "dryness"); never clinical labels like "hyperpigmentation" or "PIH"
- first_signs_weeks: [low, high] week range when the FIRST visible change usually shows (the earliest measurable improvement in the text, e.g. "measurable change at ~4 weeks")
- full_effect_weeks: [low, high] week range for the fuller, more obvious result (adjusted for their tone and severity). For actives this must come later than first signs: the two ranges are never identical
- what_to_look_for: one short, friendly sentence on what change to watch for in photos, in everyday words
- kind: "active" for treatments with a visible effect; "supportive" for cleansers / moisturizers that mainly keep skin comfortable (short ranges like [1, 2] / [2, 4]); "protective" for sunscreen (what_to_look_for explains it keeps marks from darkening; no visible "effect" to wait for, use [0, 0] / [0, 0])

Also write conversation_summary: 2-3 plain sentences summing up the person (concerns, sensitivities, goals, budget) for the top of their progress page, addressed to them ("You wanted to...").

Person:
{person}

Routine:
{routine}

Ingredient knowledge base:
{chunks}

Return JSON: {{"conversation_summary": "...", "milestones": [{{...}}, ...]}}
"""


def _person(profile: SkinProfile) -> str:
    return "\n".join([
        f"- Monk tone: {profile.monk_tone or 'unknown'}",
        f"- Concerns: {', '.join(profile.confirmed_concerns or profile.observed_concerns) or 'none given'}",
        f"- Skin type: {profile.skin_type or 'unknown'}",
        f"- Sensitivities: {', '.join(profile.allergies or []) or 'none'}",
        f"- Goals: {', '.join(profile.goals or []) or 'not specified'}",
        f"- Budget: {'$' + str(profile.budget_monthly) if profile.budget_monthly is not None else 'not given'}",
    ])


def _ingredients(recs: list[RecommendationResponse]) -> list[str]:
    seen = {}
    for rec in recs:
        for name in rec.ingredient_tags or rec.key_ingredients:
            seen.setdefault(name.lower(), name)
    return list(seen.values())[:MAX_INGREDIENTS]


def retrieval_query(ingredient: str, profile: SkinProfile) -> str:
    concerns = ", ".join(profile.confirmed_concerns or profile.observed_concerns) or "general skin health"
    return f"onset timeline: how many weeks until {ingredient} shows results for {concerns}; Monk tone {profile.monk_tone}"


def _retrieve(recs: list[RecommendationResponse], profile: SkinProfile) -> list[str]:
    chunks = []
    for ingredient in _ingredients(recs):
        for chunk in rag_service.retrieve_ingredient_context(retrieval_query(ingredient, profile), n_results=CHUNKS_PER_INGREDIENT):
            if chunk not in chunks:
                chunks.append(chunk)
    return chunks


def _weeks(value, fallback: list[int]) -> list[int]:
    try:
        low, high = (max(0, min(MAX_WEEKS, int(round(float(v))))) for v in list(value)[:2])
    except (TypeError, ValueError):
        return list(fallback)
    return [min(low, high), max(low, high)]


def _fallback(rec: RecommendationResponse) -> Milestone:
    step = rec.routine_step.lower()
    if step == "spf":
        return Milestone(asin=rec.amazon_asin, product_name=rec.product_name, ingredient="Sunscreen", concern="protection",
                         first_signs_weeks=[0, 0], full_effect_weeks=[0, 0], kind="protective",
                         what_to_look_for="Keeps dark marks from getting darker; there's no visible effect to wait for.")
    if step in ("cleanser", "moisturizer"):
        return Milestone(asin=rec.amazon_asin, product_name=rec.product_name, ingredient=(rec.ingredient_tags or [""])[0],
                         concern="comfort", first_signs_weeks=[1, 2], full_effect_weeks=[2, 4], kind="supportive",
                         what_to_look_for="Skin should feel calmer and more comfortable.")
    return Milestone(asin=rec.amazon_asin, product_name=rec.product_name, ingredient=(rec.ingredient_tags or [""])[0],
                     concern=", ".join(rec.targets[:2]), first_signs_weeks=[4, 8], full_effect_weeks=[8, 12],
                     what_to_look_for="Look for gradual change in the area this product targets.")


def clean_milestones(raw, recs: list[RecommendationResponse]) -> list[Milestone]:
    """The model's milestones, made safe: routine ASINs only, weeks clamped and ordered, one per product."""
    by_asin = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        rec = next((r for r in recs if r.amazon_asin == item.get("asin")), None)
        if rec is None or rec.amazon_asin in by_asin:
            continue
        base = _fallback(rec)
        kind = item.get("kind") if item.get("kind") in ("active", "supportive", "protective") else base.kind
        first = _weeks(item.get("first_signs_weeks"), base.first_signs_weeks)
        full = _weeks(item.get("full_effect_weeks"), base.full_effect_weeks)
        full = [max(full[0], first[0]), max(full[1], first[1])]
        by_asin[rec.amazon_asin] = Milestone(
            asin=rec.amazon_asin, product_name=rec.product_name,
            ingredient=str(item.get("ingredient") or base.ingredient)[:60],
            concern=str(item.get("concern") or base.concern)[:60],
            first_signs_weeks=first, full_effect_weeks=full,
            what_to_look_for=str(item.get("what_to_look_for") or base.what_to_look_for)[:300],
            kind=kind,
        )
    return [by_asin.get(rec.amazon_asin) or _fallback(rec) for rec in recs]


def build_timeline(profile: SkinProfile, recs: list[RecommendationResponse]) -> tuple[list[Milestone], str]:
    """(milestones in routine order, conversation_summary). Raises OpenAIError if the model is unavailable."""
    if not recs:
        return [], ""
    routine = "\n".join(
        f"- asin {r.amazon_asin} | {r.routine_step} | {r.brand} {r.product_name} | key ingredients: "
        f"{', '.join(r.key_ingredients) or 'n/a'} | targets: {', '.join(r.targets) or 'n/a'}"
        for r in recs
    )
    prompt = TIMELINE_PROMPT.format(
        person=_person(profile), routine=routine, chunks="\n\n---\n\n".join(_retrieve(recs, profile)) or "(none)",
    )
    data = conversation_service._chat_json([{"role": "system", "content": prompt}], TIMELINE_TEMPERATURE)
    summary = str(data.get("conversation_summary") or "").strip()[:600]
    return clean_milestones(data.get("milestones"), recs), summary
