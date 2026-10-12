import json

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
  "allergies": [],
  "budget_monthly": null,
  "goals": [],
  "fragrance_free": null
}"""

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
    response = _get_client().chat.completions.create(
        model=VISION_MODEL,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
        # Only trust the fields vision is responsible for; the rest come from the conversation.
        return SkinProfile(
            monk_tone=data.get("monk_tone", 0),
            observed_concerns=data.get("observed_concerns") or [],
            skin_type=data.get("skin_type") or "",
        )
    except (json.JSONDecodeError, ValidationError, TypeError):
        # Treat an unparseable answer the same as an unreadable photo.
        return SkinProfile(monk_tone=0)
