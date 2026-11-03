import json
import re

from openai import OpenAI
from pydantic import ValidationError

from models.schemas import ConversationTurn, RecommendationResponse, SkinProfile
from services import tone_rules
from services.rag_service import ROUTINE_STEPS_PRIORITY, typical_routine_costs

CHAT_MODEL = "gpt-4o"

CONVERSATION_TEMPERATURE = 0.7
RECOMMEND_TEMPERATURE = 0.4
OPENING_TEMPERATURE = 0.8
FOLLOWUP_TEMPERATURE = 0.6

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
A field that is null in the profile has NOT been answered yet. Fill it in only from what the user actually said; "none" answers become []. Never infer an answer to a question they didn't address: if you asked about products and they only talked about allergies, record the allergies, leave the product fields null, and ask about products again.
- Which concerns they actually care about (confirm or refine what you observed) -> confirmed_concerns. REQUIRED.
- Any allergies, sensitivities, or ingredients their skin reacts to (stinging, redness, breakouts) -> allergies. REQUIRED. Ask explicitly; "none" -> [].
- Products they use now or have tried, and how those went -> products_liked (worked well, or they want to keep using) and products_disliked (broke them out, irritated, or did nothing). REQUIRED. Ask explicitly, ideally tied to their concern, e.g. "Have you tried anything for those dark marks before? What worked, and what didn't?" If they haven't tried anything, set both to [].
- Roughly how much they want to spend on their routine, in USD -> budget_monthly. REQUIRED. See BUDGET below.
- Whether they want fragrance-free products -> fragrance_free. REQUIRED. (This fits naturally alongside the sensitivities question, but still only one question mark per reply.)
- What they'd love their skin to look or feel like -> goals. Optional.

WRAPPING UP
Only when confirmed_concerns, allergies, product history (products_liked / products_disliked), budget_monthly, and fragrance_free are ALL known, set is_ready_to_recommend to true. Never wrap up early because the conversation feels done; ask about whatever is still null. In that reply, don't recommend products and don't ask anything. Warmly let them know you have what you need and are putting their routine together, and remind them that results vary and to patch test new products.

SAFETY
Never diagnose medical conditions. Never say a product will definitely work.

BUDGET
Typical costs: {typical_costs} Most products last 2-4 months, so this isn't a monthly bill. Their budget is compared to the total upfront cost of the routine, give or take $10.
- Whenever budget comes up, whether you're asking about it or they just told you a number, include the actual dollar figures above in your reply, e.g. "For reference, {typical_costs_example}" Don't paraphrase them away into "affordable" or "tight".
- If their budget is below what a basic routine costs, or well below a fuller routine, also reassure them: you'll keep the most important products (sunscreen, cleanser, moisturizer, then whatever targets their main concern) within their budget, and they can change their budget later to recalculate. Never make them feel bad about a small budget.

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

FOLLOWUP_PROMPT = """You are Dermalis, a warm, skincare-savvy friend. You've already built this person's routine (below) and they're now asking follow-up questions. Keep the conversation going naturally; this is an ongoing chat, not a one-and-done.

WHAT YOU CAN HELP WITH
- What an ingredient does and why it's in their routine ("What does niacinamide do?"), using the knowledge base text below.
- How and when to use their products: order of application, AM vs PM, how often, what not to layer together, how long until results.
- How long a product usually lasts: use any size/duration info in the knowledge base. If there isn't any, give a reasonable estimate from the product type and typical size (e.g. a 1 oz serum used daily usually lasts about 2-3 months) and say it's an estimate.
- Why a specific product was picked for them, using their profile and the product's "why" note.
- General skincare questions related to their skin and concerns.

HOW YOU ANSWER
- Same voice as before: casual, kind, everyday words ("dark marks", not "hyperpigmentation"). Keep it short: 2-4 sentences, or a short list when they ask for steps or an order of use. Lead with the direct answer.
- Ground facts in the knowledge base text. If it doesn't cover something, say so briefly and answer from general skincare knowledge with care, never inventing product-specific claims (exact percentages, sizes, prices) that aren't in the text.
- Refer to their actual products by name when relevant.
- Don't close replies with stock lines like "Let me know if you have more questions" or "Feel free to ask". The chat stays open on its own. Occasionally you can offer one specific, relevant next step instead (e.g. "Want the evening order too?").
- If they want to change the routine (swap or remove a product, different budget), tell them they can tap Remove on a product card, or change the budget in the routine panel to recalculate.

SAFETY
Never diagnose medical conditions or promise results. For anything that sounds like a reaction (burning, swelling, rash that won't settle) or a medical question, suggest stopping the product and checking with a dermatologist or pharmacist.

Skin tone guidance (never state tone numbers):
{tone_guidance}

Their profile:
{skin_profile}

Their routine (step: product, price, when to use):
{routine}

Knowledge base text (their routine's product files first, then the best matches for the question):
{context}
"""

RECOMMEND_PROMPT = """You are Dermalis's product writer. Below are candidate products grouped by routine step. Within each step they are already filtered and ranked for this user's Monk skin tone, best fit first. Use ONLY these candidates and copy their ASINs exactly.

Return an entry for EVERY candidate except clear conflicts. Skip a candidate only when:
- its "NOT GOOD FOR" section directly names the user's skin type (e.g. "dry skin" for a dry-skinned user), one of their allergies or sensitivities, or their skin tone range; or
- they want fragrance-free and it isn't fragrance free; or
- it is a product in their products_disliked, or one of its key ingredients is something in products_disliked or allergies (e.g. they said niacinamide broke them out).
If a candidate is (or closely matches) something in their products_liked, keep it and say so in why_recommended, e.g. "You mentioned this already works for you."
Price, budget, and "optional step" are NEVER reasons to skip; mention them in why_recommended instead. When in doubt, include the product and note what to watch for. The user will browse and remove products themselves.

Fields:
- product_name: the product name WITHOUT the brand (brand goes in its own field).
- why_recommended: 1-2 sentences specific to this user's profile and concerns, grounded in the ingredient science from the candidate text. Where relevant, say why it suits their skin tone (e.g. no white cast, low risk of dark marks). If the step is optional for them (toner, exfoliant, spot treatment, eye cream, mask), say so briefly; don't discuss price or budget (the app handles that). Never promise results.
- key_ingredients: up to 4 key ingredients.
- ingredient_tags: 3 distinct key ingredients as short tags of 1-3 words, the common ingredient name only (never the same ingredient twice under different names; fewer if it has fewer). No percentages (except a sunscreen's SPF value) and no descriptions: "Niacinamide", "Ceramides", "SPF 30", never "Contains niacinamide for brightening".
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
    {{"amazon_asin": "", "product_name": "", "brand": "", "why_recommended": "", "key_ingredients": [""], "ingredient_tags": ["", "", ""], "tone_note": "", "not_good_for": [""]}}
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


def has_product_history(profile: SkinProfile) -> bool:
    return profile.products_liked is not None or profile.products_disliked is not None


def _has_required_fields(profile: SkinProfile) -> bool:
    return (
        bool(profile.confirmed_concerns)
        and profile.allergies is not None
        and has_product_history(profile)
        and profile.budget_monthly is not None
        and profile.fragrance_free is not None
    )


def describe_typical_costs(costs: dict) -> str:
    """Plain-language price guide for the prompt, from budget.typical_costs()."""
    if not costs:
        return "A basic routine usually costs about $30-$50 upfront."
    return (
        f"a basic routine (cleanser, moisturizer, sunscreen) usually runs about "
        f"${costs['typical_basic_low']:.0f}-${costs['typical_basic']:.0f} upfront, and a fuller routine "
        f"with treatments and extras closer to ${costs['typical_full']:.0f}."
    )


def get_next_response(
    conversation_history: list[ConversationTurn],
    skin_profile: SkinProfile,
    user_message: str,
    retrieved_context: list[str],
) -> tuple[str, SkinProfile, bool]:
    typical = describe_typical_costs(typical_routine_costs())
    system = SYSTEM_PROMPT.format(
        typical_costs=typical[0].upper() + typical[1:],
        typical_costs_example=typical,
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

    # Only trust "ready" once the required fields are actually filled in. The model sometimes
    # signs off ("I'll put your routine together") without setting the flag; a reply with no
    # question after everything is answered counts as ready too, so the user isn't left waiting.
    model_ready = bool(data.get("is_ready_to_recommend")) or "?" not in assistant_response
    is_ready = model_ready and _has_required_fields(updated)
    # Keep the model's own (warmer) sign-off unless it still asks a follow-up or skips the patch-test reminder.
    if is_ready and ("?" in assistant_response or "patch test" not in assistant_response.lower()):
        assistant_response = READY_MESSAGE
    return assistant_response, updated, is_ready


def answer_followup(
    conversation_history: list[ConversationTurn],
    skin_profile: SkinProfile,
    user_message: str,
    routine: list[RecommendationResponse],
    context: list[str],
) -> str:
    """Answer a question about the finished routine; the profile and routine are left unchanged."""
    routine_lines = "\n".join(
        f"- {rec.routine_step}: {rec.brand} {rec.product_name}, {rec.price or 'price n/a'}, {rec.routine_time or 'time n/a'}"
        for rec in routine
    ) or "(they removed every product)"
    system = FOLLOWUP_PROMPT.format(
        tone_guidance=tone_rules.guidance(skin_profile.monk_tone),
        skin_profile=skin_profile.model_dump_json(indent=2),
        routine=routine_lines,
        context="\n\n---\n\n".join(context),
    )
    messages = [{"role": "system", "content": system}]
    messages += [{"role": turn.role, "content": turn.content} for turn in conversation_history]
    messages.append({"role": "user", "content": user_message})
    response = _get_client().chat.completions.create(
        model=CHAT_MODEL, temperature=FOLLOWUP_TEMPERATURE, messages=messages
    )
    return (response.choices[0].message.content or "").strip() or "Sorry, could you say that again?"


MAX_INGREDIENT_TAGS = 3
MAX_TAG_WORDS = 3


def _short_tag(text) -> str:
    """'Niacinamide 10%' -> 'Niacinamide', 'Ceramides (1, 3, 6-II)' -> 'Ceramides'; '' if it isn't a short name."""
    if not isinstance(text, str):
        return ""
    # Keep "SPF 30"; numbers are dropped everywhere else ("Ceramides 1, 3, 6-II").
    spf = re.search(r"\bSPF\s*(\d+)", text, re.IGNORECASE)
    if spf:
        return f"SPF {spf.group(1)}"
    tag = re.sub(r"\(.*?\)|\S*\d\S*%", " ", text)
    tag = " ".join(re.split(r"[,:;]|\s\d", tag)[0].split())
    if tag.islower():
        tag = tag.title()
    return tag if 0 < len(tag.split()) <= MAX_TAG_WORDS else ""


def _ingredient_tags(raw_tags, key_ingredients) -> list[str]:
    """Up to 3 short, distinct ingredient names for the card; tops up from key_ingredients if the model's are unusable."""
    tags = []
    for source in (raw_tags, key_ingredients):
        for text in source if isinstance(source, list) else []:
            tag = _short_tag(text)
            # "Medical-grade hydrocolloid" repeats "Hydrocolloid": skip tags that contain one another.
            if tag and not any(tag.lower() in t.lower() or t.lower() in tag.lower() for t in tags):
                tags.append(tag)
            if len(tags) == MAX_INGREDIENT_TAGS:
                return tags
    return tags


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
            rec = RecommendationResponse.model_validate({
                **item,
                "routine_step": step,
                "price": meta["price"],
                "price_usd": meta["price_usd"],
                "targets": [t.strip() for t in meta["targets"].split(",") if t.strip()],
                "ingredient_tags": _ingredient_tags(item.get("ingredient_tags"), item.get("key_ingredients")),
                # Usage facts come from the KB, never the model.
                **{key: meta[key] for key in ("routine_time", "amount", "frequency", "how_to_use")},
            })
        except ValidationError:
            continue
        ranked.append((ROUTINE_STEPS_PRIORITY.index(step), rank, rec))

    ranked.sort(key=lambda entry: entry[:2])
    seen_steps = set()
    for _, _, rec in ranked:
        rec.is_alternative = rec.routine_step in seen_steps
        seen_steps.add(rec.routine_step)
    return [rec for _, _, rec in ranked]
