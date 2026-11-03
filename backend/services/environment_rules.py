"""
Climate and environment rules: the one place that turns an EnvironmentProfile into advice.

Everything here is a suggestion about factors that can affect skin, never a cause or a diagnosis.
Each rule says when it applies, what the consultant should keep in mind, which routine steps get
a short card tag and a plain-language note, and which KB `Targets` get a small ranking nudge.
Edit the constants and RULES below to tune behavior; tests live in tests/test_environment.py.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from models.schemas import EnvironmentProfile

# --- Climate classification ---------------------------------------------------------------
# Means over the last ~2 weeks. Dry vs humid uses the dew point (how much moisture the air
# actually holds), because cold air often shows a high relative humidity while being dry for skin.
HOT_TEMP_C = 24          # mean temperature at or above this is "hot"
COLD_TEMP_C = 10         # mean temperature at or below this is "cold"
HUMID_DEW_POINT_C = 16   # hot and dew point at or above this: "hot and humid"
COLD_DRY_DEW_POINT_C = 4 # cold and dew point below this: "cold and dry"
DRY_HUMIDITY_PCT = 40    # mean relative humidity below this counts as dry air in any climate
HIGH_UV = 6              # WHO "high" starts at 6

CLIMATE_LABELS = {
    "hot_humid": "hot and humid",
    "hot_dry": "hot and dry",
    "mild": "mild",
    "cold_dry": "cold and dry",
    "cold_humid": "cold and damp",
}

# Ranking nudges, in the same small units as tone_rules (similarities sit around 0.7-0.9).
CLIMATE_BOOST = 0.03
CLIMATE_PENALTY = 0.02


def climate_label(avg_temp_c: float, dew_point_c: float, avg_humidity: float | None = None) -> str:
    if avg_temp_c >= HOT_TEMP_C:
        return "hot_humid" if dew_point_c >= HUMID_DEW_POINT_C else "hot_dry"
    if avg_temp_c <= COLD_TEMP_C:
        dry = dew_point_c < COLD_DRY_DEW_POINT_C or (avg_humidity is not None and avg_humidity < DRY_HUMIDITY_PCT)
        return "cold_dry" if dry else "cold_humid"
    return "mild"


TROPICS_LAT = 23.5  # inside the tropics, four seasons don't mean much; the climate label covers it


def season(lat: float, today: date) -> str | None:
    """Meteorological season for the hemisphere (Dec-Feb is winter in the north, summer in the south)."""
    if abs(lat) < TROPICS_LAT:
        return None
    north = ("winter", "winter", "spring", "spring", "spring", "summer",
             "summer", "summer", "autumn", "autumn", "autumn", "winter")[today.month - 1]
    if lat >= 0:
        return north
    return {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}[north]


def uv_word(uv: float) -> str:
    if uv < 3:
        return "low"
    if uv < HIGH_UV:
        return "moderate"
    if uv < 8:
        return "high"
    return "very high"


def summary(env: EnvironmentProfile) -> str:
    """Short plain-language line for the user, e.g. 'Currently cold and dry where you are, UV is low.'"""
    parts = []
    if env.climate_label:
        parts.append(f"Currently {CLIMATE_LABELS[env.climate_label]} where you are")
    if env.uv_index_max is not None:
        parts.append(f"UV is {uv_word(env.uv_index_max)}")
    return (", ".join(parts) + ".") if parts else ""


# --- Rules ---------------------------------------------------------------------------------
def _dry_air(env: EnvironmentProfile) -> bool:
    return (
        env.climate_label in ("cold_dry", "hot_dry")
        or (env.avg_humidity is not None and env.avg_humidity < DRY_HUMIDITY_PCT)
        or env.indoor == "heated"
    )


def _humid_heat(env: EnvironmentProfile) -> bool:
    return env.climate_label == "hot_humid"


def _strong_sun(env: EnvironmentProfile) -> bool:
    return (env.uv_index_max is not None and env.uv_index_max >= HIGH_UV) or env.outdoor_time == "over_2h"


@dataclass(frozen=True)
class StepNote:
    tag: str    # 1-3 words for the card, e.g. "Dry-air pick"; "" for no tag
    note: str   # one plain sentence for Details and the How-to-use panel
    requires: tuple[str, ...] = ()  # only when the product targets one of these (empty: any product)


@dataclass(frozen=True)
class Rule:
    id: str
    applies: Callable[[EnvironmentProfile], bool]
    guidance: str                                   # for the consultant's context
    notes: dict[str, list[StepNote]] = field(default_factory=dict)  # step -> first matching note wins
    boost_steps: tuple[str, ...] = ()
    boost_targets: tuple[str, ...] = ()
    penalty_targets: tuple[str, ...] = ()


HYDRATING = ("dryness", "barrier", "dehydration")

RULES: list[Rule] = [
    Rule(
        id="dry_air",
        applies=_dry_air,
        guidance=(
            "Dry air (from the climate or indoor heating) is a factor that can leave skin feeling tight. "
            "Suggest richer moisturizers and barrier-supporting ingredients (ceramides, glycerin, hyaluronic acid), "
            "a little extra moisturizer at night, and going easy on exfoliating."
        ),
        notes={
            "moisturizer": [
                StepNote("Dry-air pick", "Dry air can leave skin feeling tight, and this richer, barrier-supporting "
                         "moisturizer suits it. On dry days, a little extra at night can help.", HYDRATING),
                StepNote("", "In dry air, a little extra moisturizer at night can help."),
            ],
            "cleanser": [StepNote("Gentle for dry air", "A gentle, non-stripping cleanser is a good match for dry air.", HYDRATING)],
            "serum": [StepNote("Adds hydration", "This adds a layer of hydration, which helps in dry air.", ("dehydration", "dryness"))],
            "exfoliant": [StepNote("Go easy", "In dry air, exfoliating less often (once or twice a week to start) is usually kinder to skin.")],
        },
        boost_steps=("moisturizer", "cleanser", "serum"),
        boost_targets=HYDRATING,
    ),
    Rule(
        id="humid_heat",
        applies=_humid_heat,
        guidance=(
            "Hot, humid weather is a factor that can make heavier products feel greasy. "
            "Suggest lightweight, non-greasy textures; a thin layer of moisturizer is usually enough."
        ),
        notes={
            "moisturizer": [
                StepNote("Light for humidity", "In hot, humid weather a lightweight, non-greasy moisturizer like this "
                         "one is usually enough.", ("oiliness", "dehydration")),
                StepNote("", "In hot, humid weather a thin layer is usually enough."),
            ],
        },
        boost_steps=("moisturizer", "cleanser"),
        # In the KB, "dehydration" marks lightweight hydrators (gel creams), "dryness" the rich creams.
        boost_targets=("oiliness", "dehydration"),
        penalty_targets=("dryness",),
    ),
    Rule(
        id="strong_sun",
        applies=_strong_sun,
        guidance=(
            "Strong sun or long hours outdoors are factors that can affect skin, including dark marks. "
            "Stress broad-spectrum sunscreen every morning and reapplying about every 2 hours outdoors."
        ),
        notes={
            "spf": [StepNote("Sun-ready", "With strong sun or long hours outside, broad-spectrum sunscreen every morning, "
                             "reapplied about every 2 hours outdoors, is one of the most helpful habits.")],
        },
    ),
    Rule(
        id="ac_indoors",
        applies=lambda env: env.indoor == "ac",
        guidance="Air-conditioning can dry the air a little; a hydrating layer can help.",
    ),
    Rule(
        id="travel",
        applies=lambda env: env.travels is True,
        guidance=(
            "They often travel between very different climates. Suggest a flexible routine: keep the basics, "
            "and switch to a richer moisturizer in dry or cold places and a lighter one in humid heat."
        ),
    ),
]

SUNSCREEN_HABIT_GUIDANCE = {
    "rarely": "They rarely or never wear sunscreen. Gently encourage a daily habit without lecturing, "
              "e.g. by finding a sunscreen they'd actually enjoy wearing.",
    "sometimes": "They wear sunscreen sometimes; a friendly nudge toward every morning fits naturally.",
}

INDOOR = {"heated": "mostly heated", "ac": "mostly air-conditioned", "mixed": "mixed"}
OUTDOOR = {"under_30m": "under 30 minutes", "30m_2h": "30 minutes to 2 hours", "over_2h": "over 2 hours"}
MAX_LIFESTYLE_CONTEXT = 500


def active_rules(env: EnvironmentProfile | None) -> list[Rule]:
    return [rule for rule in RULES if rule.applies(env)] if env else []


def guidance(env: EnvironmentProfile | None) -> str:
    """Context block for the consultant's prompts. Never includes the place name or coordinates."""
    if env is None:
        return "Not shared. Don't ask where they live."
    facts = []
    if env.climate_label:
        facts.append(f"climate right now: {CLIMATE_LABELS[env.climate_label]}")
    if env.avg_temp_c is not None:
        facts.append(f"recent average {env.avg_temp_c:.0f}°C")
    if env.avg_humidity is not None:
        facts.append(f"humidity {env.avg_humidity:.0f}%")
    if env.uv_index_max is not None:
        facts.append(f"UV {uv_word(env.uv_index_max)}")
    if env.season:
        facts.append(f"season: {env.season}")
    if env.indoor:
        facts.append(f"indoors: {INDOOR[env.indoor]}")
    if env.outdoor_time:
        facts.append(f"outdoors: {OUTDOOR[env.outdoor_time]} a day")
    if env.sunscreen_habit:
        facts.append(f"sunscreen: {'rarely or never' if env.sunscreen_habit == 'rarely' else env.sunscreen_habit}")
    if env.travels is not None:
        facts.append(f"travels between climates: {'often' if env.travels else 'rarely'}")

    lines = ["Their environment: " + ("; ".join(facts) if facts else "partly shared") + "."]
    lines += [f"- {rule.guidance}" for rule in active_rules(env)]
    if env.sunscreen_habit in SUNSCREEN_HABIT_GUIDANCE:
        lines.append(f"- {SUNSCREEN_HABIT_GUIDANCE[env.sunscreen_habit]}")
    for module_id, module in (env.lifestyle or {}).items():
        if module.context:
            lines.append(f"- Lifestyle ({module_id}): {module.context[:MAX_LIFESTYLE_CONTEXT]}")
    lines.append(
        "When it fits, mention it naturally (e.g. \"It's dry and cold where you are right now, so...\"). "
        "Talk about these as factors that can affect skin and as suggestions; never say the climate or their "
        "lifestyle causes a condition."
    )
    return "\n".join(lines)


def score(meta: dict, env: EnvironmentProfile | None) -> float:
    """Ranking adjustment for one product (added to the tone-adjusted score)."""
    targets = meta.get("targets", "")
    step = meta.get("routine_step")
    adjust = 0.0
    for rule in active_rules(env):
        if step not in rule.boost_steps:
            continue
        if any(t in targets for t in rule.boost_targets):
            adjust += CLIMATE_BOOST
        if any(t in targets for t in rule.penalty_targets):
            adjust -= CLIMATE_PENALTY
    return adjust


def notes_for(step: str, targets: list[str], env: EnvironmentProfile | None) -> tuple[str, str]:
    """(card tag, plain sentence) for a product in this step, or ("", "") when no rule applies."""
    for rule in active_rules(env):
        for note in rule.notes.get(step, []):
            if not note.requires or any(t in targets for t in note.requires):
                return note.tag, note.note
    return "", ""
