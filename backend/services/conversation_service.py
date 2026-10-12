import json

from openai import OpenAI
from pydantic import ValidationError

from models.schemas import ConversationTurn, RecommendationResponse, SkinProfile

CHAT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are SkinSage, a knowledgeable and empathetic AI skincare consultant.

You have already analyzed the user's photo. Their current skin profile is provided to you.

Your job is to have a natural conversation to fill in the missing profile fields. Ask ONE question at a time. Never ask multiple questions in one turn. Keep responses to 2-3 sentences max.

Ask about in this order (skip if already known):
1. Their main skin concern right now
2. Any known allergies or sensitivities to ingredients
3. Monthly skincare budget in USD
4. Whether they prefer fragrance-free products
5. Any current products they are already using and want to keep

After you have collected at least: confirmed_concerns, budget_monthly, and fragrance_free — set is_ready_to_recommend to true in your response.

When ready to recommend, do not give recommendations in conversation. Just say you have everything you need and will generate their routine now.

Never diagnose medical conditions. Never say a product will definitely work. Always say results may vary and to patch test new products.

Current skin profile from photo analysis:
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

RECOMMEND_PROMPT = """You are SkinSage's product selector. Pick exactly 3 products for this user from the candidate products below. Use ONLY these candidates and copy their ASINs exactly.

Selection rules, in priority order:
1. Exclude any product whose "NOT GOOD FOR" section matches the user (skin type, skin tone, allergies, sensitivities).
2. If the user wants fragrance-free products, exclude any product that is not fragrance free.
3. Prefer a combination whose total price roughly fits the user's monthly budget.
4. Prefer covering different routine steps (e.g. cleanser + serum/treatment + spf/moisturizer) and the user's confirmed concerns.
If the rules leave fewer than 3 products, fill the remaining slots with the least-conflicting candidates and say what to watch for in why_recommended.

product_name: the product name WITHOUT the brand (brand goes in its own field).
why_recommended: 1-2 sentences specific to this user's profile, grounded in the ingredient science from the candidate text. Never promise results.

User profile:
{skin_profile}

Candidate products:
{product_docs}

Return JSON:
{{
  "recommendations": [
    {{"product_name": "", "brand": "", "amazon_asin": "", "routine_step": "", "why_recommended": "", "key_ingredients": [""]}}
  ]
}}
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _chat_json(messages: list[dict]) -> dict:
    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        temperature=0.4,
        messages=messages,
    )
    return json.loads(response.choices[0].message.content or "{}")


def _has_required_fields(profile: SkinProfile) -> bool:
    return bool(profile.confirmed_concerns) and profile.budget_monthly is not None and profile.fragrance_free is not None


def get_next_response(
    conversation_history: list[ConversationTurn],
    skin_profile: SkinProfile,
    user_message: str,
    retrieved_context: list[str],
) -> tuple[str, SkinProfile, bool]:
    system = SYSTEM_PROMPT.format(
        skin_profile=skin_profile.model_dump_json(indent=2),
        retrieved_context="\n\n---\n\n".join(retrieved_context),
    )
    messages = [{"role": "system", "content": system}]
    messages += [{"role": turn.role, "content": turn.content} for turn in conversation_history]
    messages.append({"role": "user", "content": user_message})

    data = _chat_json(messages)
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
    if is_ready:
        # The model sometimes flags ready but still asks a follow-up; close the conversation consistently.
        assistant_response = READY_MESSAGE
    return assistant_response, updated, is_ready


def recommend(skin_profile: SkinProfile, product_docs: list[str]) -> list[RecommendationResponse]:
    prompt = RECOMMEND_PROMPT.format(
        skin_profile=skin_profile.model_dump_json(indent=2),
        product_docs="\n\n---\n\n".join(product_docs),
    )
    data = _chat_json([{"role": "system", "content": prompt}])

    known_text = "\n".join(product_docs)
    recommendations = []
    for item in data.get("recommendations", []):
        try:
            rec = RecommendationResponse.model_validate(item)
        except ValidationError:
            continue
        # Drop anything the model invented that isn't in the knowledge base.
        if rec.amazon_asin and rec.amazon_asin in known_text:
            recommendations.append(rec)
    return recommendations[:3]
