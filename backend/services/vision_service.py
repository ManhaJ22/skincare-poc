import json
from dataclasses import dataclass

from openai import OpenAI
from pydantic import ValidationError

from models.schemas import SkinProfile

VISION_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a cosmetic skin analysis assistant for a skincare consultation app. You are NOT a doctor and must never make medical diagnoses (do not name conditions like rosacea, eczema, melasma, or dermatitis). Describe only visible, cosmetic skin characteristics.

Analyze the face photo and:
1. Estimate the Monk Skin Tone on a 1-10 scale, where 1 is lightest and 10 is deepest.
2. Identify visible skin concerns, using only these labels: "hyperpigmentation", "texture", "dryness", "oiliness", "redness", "acne".
3. Estimate skin type: one of "oily", "dry", "combination", "normal".

If the image is too low quality to analyze (blurry, too dark, face not visible, not a face), return monk_tone: 0 with empty observed_concerns and empty skin_type.

Return ONLY a JSON object matching this schema, with every other field left at its empty default:
{
  "monk_tone": <int 0-10>,
  "observed_concerns": [<strings>],
  "skin_type": "<string>",
  "confirmed_concerns": [],
  "allergies": null,
  "budget_monthly": null,
  "goals": null,
  "fragrance_free": null,
  "products_liked": null,
  "products_disliked": null
}"""

# Batch testing on FairFace (scripts/fairface_harness.py), opted into with "prompt": "fairface" on /analyze.
# Not used by the app: it guesses on poor photos instead of asking the user to retake them.
FAIRFACE_PROMPT = """You are a skin analysis assistant. Analyze the provided face photo and return a JSON object.

MONK SKIN TONE SCALE — use these descriptions to select the closest number:
  1 — Porcelain, very fair, almost no visible melanin. Burns immediately, never tans.
  2 — Fair, light ivory. Burns easily, rarely tans.
  3 — Light beige, some warmth. Burns sometimes, tans minimally.
  4 — Light-medium, warm or peachy undertone. Sometimes burns, tans gradually.
  5 — Medium, golden or olive undertone. Rarely burns, tans easily.
  6 — Medium-deep, warm tan or caramel. Seldom burns, tans darkly.
  7 — Deep, rich brown. Almost never burns, always tans.
  8 — Deep brown, high melanin, warm undertone.
  9 — Very deep brown, minimal surface variation visible.
  10 — Deepest brown to ebony. Highest melanin concentration.

RULES:
- You MUST always return a monk_tone between 1 and 10. Never return 0 or null.
- If lighting is imperfect but a face is visible, make your best determination.
- When uncertain between two tones, pick the one closer to what you see; don't lean lighter or darker.
- Do not refuse to classify based on image quality unless the face is completely unidentifiable.

Analyze and return ONLY this JSON, no other text:
{
  "monk_tone": <integer 1-10>,
  "observed_concerns": [<list of strings, e.g. "dark spots", "texture", "redness", "acne">],
  "skin_type_estimate": "<oily|dry|combination|normal|unknown>",
  "confidence": "<high|medium|low>"
}

If no face is detected, return: {"monk_tone": 0, "observed_concerns": [], "skin_type_estimate": "unknown", "confidence": "none"}
Only return monk_tone 0 when there is literally no face in the image."""

PROMPTS = {"app": SYSTEM_PROMPT, "fairface": FAIRFACE_PROMPT}
CONFIDENCE_LEVELS = {"high", "medium", "low", "none"}


@dataclass
class VisionResult:
    profile: SkinProfile
    confidence: str | None = None  # only the fairface prompt asks for it


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _to_data_uri(image_base64: str) -> str:
    if image_base64.startswith("data:"):
        return image_base64
    return f"data:image/jpeg;base64,{image_base64}"


def analyze_skin_photo(image_base64: str) -> SkinProfile:
    return analyze_photo(image_base64).profile


def analyze_photo(image_base64: str, prompt: str = "app") -> VisionResult:
    response = _get_client().chat.completions.create(
        model=VISION_MODEL,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPTS[prompt]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the skin in this photo."},
                    {"type": "image_url", "image_url": {"url": _to_data_uri(image_base64), "detail": "high"}},
                ],
            },
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        # The app prompt answers "skin_type"; the fairface prompt "skin_type_estimate", which may be "unknown".
        skin_type = data.get("skin_type") or data.get("skin_type_estimate") or ""
        # Only trust the fields vision is responsible for; the rest come from the conversation.
        profile = SkinProfile(
            monk_tone=data.get("monk_tone", 0),
            observed_concerns=data.get("observed_concerns") or [],
            skin_type="" if skin_type == "unknown" else skin_type,
        )
        confidence = data.get("confidence")
        return VisionResult(profile, confidence if confidence in CONFIDENCE_LEVELS else None)
    except (json.JSONDecodeError, ValidationError, TypeError, AttributeError):
        # Treat an unparseable answer the same as an unreadable photo.
        return VisionResult(SkinProfile(monk_tone=0))
