import json

from openai import OpenAI
from pydantic import ValidationError

from models.schemas import ConversationTurn, RecommendationResponse, SkinProfile
from services import tone_rules
from services.rag_service import ROUTINE_STEPS_PRIORITY

CHAT_MODEL = "gpt-4o"

CONVERSATION_TEMPERATURE = 0.7
RECOMMEND_TEMPERATURE = 0.4
OPENING_TEMPERATURE = 0.8

SYSTEM_PROMPT = """You are Dermalis, a warm, skincare-savvy friend helping someone build a routine. You've already looked at their photo; what you saw is in their profile below.

HOW YOU TALK
- Casual, kind, and encouraging, like a friend who happens to know a lot about skin. Light humor is welcome. Never clinical, never salesy. Use everyday words ("dark marks", "shine", "bumps") rather than labels like "hyperpigmentation".
- 2-4 short sentences per reply. Vary how you phrase things; don't fall into a script.
- Ask at most ONE question per reply (one question mark). It's fine to ask none if the moment calls for just listening.

LEAD WITH WHAT YOU NOTICED
- Don't ask open-ended questions like "What problems are you noticing?" Instead, bring up what the photo showed and make an educated guess they can confirm or correct, e.g. "Your T-zone looked a little shiny. Does it get oily by the afternoon?" or "I spotted a few dark marks on your cheeks. Do those tend to hang around after a breakout?"
- Build each follow-up on what they just told you, so it feels like a conversation, not a form.

EMPATHY AND SMALL TALK
- If they sound frustrated, self-conscious, or tired of their skin, acknowledge it first and reassure them (most concerns are really common and very manageable). Never shame or alarm.
- If they make small talk or go off-topic, respond genuinely and briefly, then gently steer back.

SUGGESTIONS
- You can drop a quick, helpful tip tied to what they said (an ingredient, a habit, why SPF matters for dark marks), but don't name specific products. Products come in their routine at the end.

WHAT YOU NEED TO LEARN (any order, whatever flows naturally; skip what's already known)
- Which concerns they actually care about (confirm or refine what you observed) -> confirmed_concerns. REQUIRED.
- Roughly how much they want to spend on skincare per month, in USD -> budget_monthly. REQUIRED.
- Whether they want fragrance-free products -> fragrance_free. REQUIRED. (This fits naturally alongside a question about sensitivities.)
- Any ingredient allergies or sensitivities -> allergies.
- What they'd love their skin to look or feel like -> goals.
- Products they already use and want to keep.

WRAPPING UP
Once confirmed_concerns, budget_monthly, and fragrance_free are all known, set is_ready_to_recommend to true. In that reply, don't recommend products and don't ask anything. Warmly let them know you have what you need and are putting their routine together, and remind them that results vary and to patch test new products.

SAFETY
Never diagnose medical conditions. Never say a product will definitely work.

Skin tone guidance (use it to make questions and tips relevant; never state tone numbers to the user):
{tone_guidance}

Current skin profile:
{skin_profile}

Relevant product and ingredient context from knowledge base:
{retrieved_context}

Update the skin profile JSON with any new information extracted from the user's message. Return your response as JSON:
{{
  "assistant_response": "your conversational response here",
  "updated_profile": {{ ...full updated SkinProfile as JSON... }},
  "is_ready_to_recommend": false
}}
"""

READY_MESSAGE = (
    "Thank you, I have everything I need and will generate your routine now. "
    "Results may vary from person to person, so please patch test each new product before using it on your full face."
)

OPENING_PROMPT = """You are Dermalis, a warm, skincare-savvy friend. You just looked at someone's photo and are starting a chat to help build their skincare routine.

Write your first message (2-4 short sentences, plain text, no lists):
- Greet them warmly and naturally.
- Mention one or two things you noticed (from the observations below) in plain, kind, non-judgmental language, using everyday words rather than the labels (e.g. "dark marks" not "hyperpigmentation", "a bit of shine" not "oiliness"), and reassure them these are common and manageable.
- End with ONE specific question that checks one observation with them, e.g. "Do those dark marks tend to hang around after a breakout?" Do NOT ask a generic question like "What problems are you noticing?"
- If nothing notable was observed, compliment something genuine and ask what they'd love their skin to do better.
- Don't recommend products. Never diagnose medical conditions. Never mention tone numbers or the Monk scale.

What you noticed in the photo:
- Visible concerns: {observed_concerns}
- Likely skin type: {skin_type}

Skin tone guidance (use only if it makes your question more relevant):
{tone_guidance}
"""

RECOMMEND_PROMPT = """You are Dermalis's product writer. Below are candidate products grouped by routine step. Within each step they are already filtered and ranked for this user's Monk skin tone, best fit first. Use ONLY these candidates and copy their ASINs exactly.

Return an entry for EVERY candidate except clear conflicts. Skip a candidate only when:
- its "NOT GOOD FOR" section directly names the user's skin type (e.g. "dry skin" for a dry-skinned user), one of their allergies or sensitivities, or their skin tone range; or
- they want fragrance-free and it isn't fragrance free.
Price, budget, and "optional step" are NEVER reasons to skip; mention them in why_recommended instead. When in doubt, include the product and note what to watch for. The user will browse and remove products themselves.

Fields:
- product_name: the product name WITHOUT the brand (brand goes in its own field).
- why_recommended: 1-2 sentences specific to this user's profile and concerns, grounded in the ingredient science from the candidate text. Where relevant, say why it suits their skin tone (e.g. no white cast, low risk of dark marks). If the step is optional for them (toner, exfoliant, spot treatment, eye cream, mask) or pushes their budget (${budget} per month), say so briefly. Never promise results.
- key_ingredients: up to 4 key ingredients.
- routine_time: "AM", "PM", or "AM & PM", from the candidate's time of use.
- tone_note: 1 short sentence on how it behaves on their skin tone, from the candidate's Monk tone notes.
- not_good_for: up to 3 short phrases from the candidate's "NOT GOOD FOR" section that the user should know about.

User profile:
{skin_profile}

Skin tone guidance:
{tone_guidance}

Candidate products:
{product_docs}

Return JSON:
{{
  "recommendations": [
    {{"amazon_asin": "", "product_name": "", "brand": "", "why_recommended": "", "key_ingredients": [""], "routine_time": "", "tone_note": "", "not_good_for": [""]}}
  ]
}}
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _chat_json(messages: list[dict], temperature: float) -> dict:
    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        temperature=temperature,
        messages=messages,
    )
    return json.loads(response.choices[0].message.content or "{}")


def get_opening_message(skin_profile: SkinProfile) -> str:
    """First chat message, written from what the photo analysis observed."""
    prompt = OPENING_PROMPT.format(
        observed_concerns=", ".join(skin_profile.observed_concerns) or "nothing notable",
        skin_type=skin_profile.skin_type or "unclear",
        tone_guidance=tone_rules.guidance(skin_profile.monk_tone),
    )
    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=OPENING_TEMPERATURE,
        messages=[{"role": "system", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def _has_required_fields(profile: SkinProfile) -> bool:
    return bool(profile.confirmed_concerns) and profile.budget_monthly is not None and profile.fragrance_free is not None


def get_next_response(
    conversation_history: list[ConversationTurn],
    skin_profile: SkinProfile,
    user_message: str,
    retrieved_context: list[str],
) -> tuple[str, SkinProfile, bool]:
    system = SYSTEM_PROMPT.format(
        tone_guidance=tone_rules.guidance(skin_profile.monk_tone),
        skin_profile=skin_profile.model_dump_json(indent=2),
        retrieved_context="\n\n---\n\n".join(retrieved_context),
    )
    messages = [{"role": "system", "content": system}]
    messages += [{"role": turn.role, "content": turn.content} for turn in conversation_history]
    messages.append({"role": "user", "content": user_message})

    data = _chat_json(messages, CONVERSATION_TEMPERATURE)
    assistant_response = data.get("assistant_response") or "Sorry, could you say that again?"

    try:
        updated = SkinProfile.model_validate(data.get("updated_profile") or {})
        # Photo-derived fields come from vision, not the chat; don't let the model drop them.
        updated.monk_tone = skin_profile.monk_tone
        updated.observed_concerns = skin_profile.observed_concerns
    except ValidationError:
        updated = skin_profile

    # Only trust "ready" once the required fields are actually filled in.
    is_ready = bool(data.get("is_ready_to_recommend")) and _has_required_fields(updated)
    # Keep the model's own (warmer) sign-off unless it still asks a follow-up or skips the patch-test reminder.
    if is_ready and ("?" in assistant_response or "patch test" not in assistant_response.lower()):
        assistant_response = READY_MESSAGE
    return assistant_response, updated, is_ready


def recommend(skin_profile: SkinProfile, candidates: dict[str, list[dict]]) -> list[RecommendationResponse]:
    """
    Write up the ranked per-step candidates. Code, not the model, decides step, price, and which
    pick is primary: the best-ranked surviving candidate per step; the rest become alternatives.
    """
    sections = []
    known = {}  # asin -> (step, rank, metadata)
    for step, hits in candidates.items():
        sections.append(f"### Step: {step}\n\n" + "\n\n---\n\n".join(hit["doc"] for hit in hits))
        for rank, hit in enumerate(hits):
            known[hit["meta"]["asin"]] = (step, rank, hit["meta"])

    prompt = RECOMMEND_PROMPT.format(
        budget=skin_profile.budget_monthly if skin_profile.budget_monthly is not None else "unknown",
        skin_profile=skin_profile.model_dump_json(indent=2),
        tone_guidance=tone_rules.guidance(skin_profile.monk_tone),
        product_docs="\n\n".join(sections),
    )
    data = _chat_json([{"role": "system", "content": prompt}], RECOMMEND_TEMPERATURE)

    ranked = []
    for item in data.get("recommendations", []):
        asin = item.get("amazon_asin") if isinstance(item, dict) else None
        # Drop anything the model invented that isn't a candidate, and duplicates.
        if asin not in known or any(rec.amazon_asin == asin for _, _, rec in ranked):
            continue
        step, rank, meta = known[asin]
        try:
            rec = RecommendationResponse.model_validate({**item, "routine_step": step, "price": meta["price"]})
        except ValidationError:
            continue
        ranked.append((ROUTINE_STEPS_PRIORITY.index(step), rank, rec))

    ranked.sort(key=lambda entry: entry[:2])
    seen_steps = set()
    for _, _, rec in ranked:
        rec.is_alternative = rec.routine_step in seen_steps
        seen_steps.add(rec.routine_step)
    return [rec for _, _, rec in ranked]
