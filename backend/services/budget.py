"""
Fit a routine to the user's budget.

Budget is compared against the routine's upfront cost (the sum of listed prices), with a
±TOLERANCE window. When everything doesn't fit, the most important steps are kept: essentials
first, then a treatment for the user's concern, then extras. Within a step, a cheaper
alternative can replace the top pick if that frees room for another step.
"""

from statistics import median

from models.schemas import BudgetSummary, RecommendationResponse

TOLERANCE = 10

ESSENTIAL_STEPS = ["cleanser", "moisturizer", "spf"]

# How much keeping each step is worth; higher survives a tight budget.
STEP_WEIGHTS = {
    "spf": 100,
    "cleanser": 95,
    "moisturizer": 90,
    "treatment": 35,
    "serum": 35,
    "spot treatment": 25,
    "toner": 12,
    "exfoliant": 12,
    "eye cream": 8,
    "mask": 8,
}
CONCERN_MATCH_WEIGHT = 60  # a treatment/serum that targets one of the user's concerns
CONCERN_STEPS = {"treatment", "serum"}
# Each alternative is worth this much less than the pick ranked above it.
ALTERNATIVE_DISCOUNT = 0.85

# User wording -> KB "Targets" vocabulary.
CONCERN_KEYWORDS = {
    "hyperpigmentation": ["hyperpigment", "dark mark", "dark spot", "pih", "uneven", "discolor", "melasma", "mark"],
    "acne": ["acne", "breakout", "pimple", "blemish", "zit", "whitehead", "blackhead"],
    "oiliness": ["oil", "shine", "greasy"],
    "dryness": ["dry", "flak", "tight"],
    "dehydration": ["dehydrat"],
    "redness": ["redness", "red patch", "rosacea", "flush"],
    "sensitivity": ["sensitiv", "irritat", "reactive"],
    "texture": ["texture", "rough", "bump", "pore"],
    "aging": ["aging", "ageing", "wrinkle", "fine line"],
    "dark circles": ["dark circle", "under-eye", "under eye"],
}


def concern_targets(concerns: list[str]) -> set[str]:
    """Map free-text concerns ("dark marks", "breakouts") to KB target names."""
    text = " ".join(concerns).lower()
    return {target for target, words in CONCERN_KEYWORDS.items() if any(w in text for w in words)}


def _weight(rec: RecommendationResponse, wanted: set[str]) -> float:
    if rec.routine_step in CONCERN_STEPS and wanted & set(rec.targets):
        return CONCERN_MATCH_WEIGHT
    return STEP_WEIGHTS.get(rec.routine_step, 5)


def summarize(total: float, budget: int | None) -> dict:
    """Where a routine total sits relative to budget ± TOLERANCE."""
    if budget is None:
        return {"budget": None, "low": None, "high": None, "status": "no_budget"}
    low, high = budget - TOLERANCE, budget + TOLERANCE
    status = "over" if total > high else "under" if total < low else "within"
    return {"budget": budget, "low": low, "high": high, "status": status}


def typical_costs(products: list[tuple[str, float]]) -> dict:
    """What routines usually cost, from (step, price) pairs, rounded to $5 for display."""
    by_step: dict[str, list[float]] = {}
    for step, price in products:
        by_step.setdefault(step, []).append(price)

    def round5(x: float) -> float:
        return 5 * round(x / 5)

    essentials = [by_step[s] for s in ESSENTIAL_STEPS if s in by_step]
    return {
        "typical_basic_low": round5(sum(min(p) for p in essentials)),
        "typical_basic": round5(sum(median(p) for p in essentials)),
        "typical_full": round5(sum(median(p) for p in by_step.values())),
    }


def fit_routine(
    recs: list[RecommendationResponse],
    budget: int | None,
    concerns: list[str],
    typical: dict | None = None,
) -> tuple[list[RecommendationResponse], BudgetSummary]:
    """
    Pick at most one product per step, maximizing importance while the total stays at or below
    budget + TOLERANCE. Recs keep their order (ranked best-first within each step); the chosen
    one per step is marked is_alternative=False and skipped steps get budget_left_out=True.
    """
    steps: dict[str, list[RecommendationResponse]] = {}
    for rec in recs:
        steps.setdefault(rec.routine_step, []).append(rec)
    wanted = concern_targets(concerns)
    full_total = sum(options[0].price_usd for options in steps.values())

    if budget is None:
        chosen = {step: options[0] for step, options in steps.items()}
    else:
        chosen = _best_within(steps, budget + TOLERANCE, wanted)
        if not chosen and steps:
            # Nothing fits even alone: keep the single most important product, cheapest option.
            step, options = max(steps.items(), key=lambda kv: max(_weight(r, wanted) for r in kv[1]))
            chosen = {step: min(options, key=lambda r: r.price_usd)}

    for step, options in steps.items():
        for rec in options:
            rec.is_alternative = rec is not chosen.get(step)
            rec.budget_left_out = step not in chosen

    total = round(sum(rec.price_usd for rec in chosen.values()), 2)
    summary = BudgetSummary(
        **summarize(total, budget),
        total=total,
        full_total=round(full_total, 2),
        left_out_steps=[step for step in steps if step not in chosen],
        **(typical or {}),
    )
    return recs, summary


def _best_within(
    steps: dict[str, list[RecommendationResponse]], cap: float, wanted: set[str]
) -> dict[str, RecommendationResponse]:
    """Knapsack over whole-dollar costs: one option (or none) per step, best total score under cap."""
    cap = int(cap)
    # cost -> (score, {step: rec}); only reachable costs are stored.
    best: dict[int, tuple[float, dict]] = {0: (0.0, {})}
    for step, options in steps.items():
        nxt = dict(best)  # skipping this step keeps every existing state
        for cost, (score, picks) in best.items():
            for rank, rec in enumerate(options):
                new_cost = cost + round(rec.price_usd)
                if new_cost > cap:
                    continue
                new_score = score + _weight(rec, wanted) * ALTERNATIVE_DISCOUNT**rank
                if new_cost not in nxt or new_score > nxt[new_cost][0]:
                    nxt[new_cost] = (new_score, {**picks, step: rec})
        best = nxt
    # Highest score wins; among equal scores, the cheaper routine.
    _, (_, picks) = max(best.items(), key=lambda kv: (kv[1][0], -kv[0]))
    return picks
