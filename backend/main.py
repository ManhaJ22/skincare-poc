from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    EnvironmentLookupRequest,
    EnvironmentLookupResponse,
    EnvironmentProfile,
    FitBudgetRequest,
    FitBudgetResponse,
    RecommendationsRequest,
    RecommendationsResponse,
    SkinProfile,
)
from services import budget, conversation_service, environment, rag_service, vision_service

# Products per routine step: one primary pick plus alternatives the user can swap in.
CANDIDATES_PER_STEP = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = rag_service.init_collection()
    print(f"Knowledge base loaded: {count} documents")
    yield


app = FastAPI(title="Dermalis POC", lifespan=lifespan)

# POC only: allow every origin so frontend/index.html works from file://.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def build_opening_message(profile: SkinProfile, env: EnvironmentProfile | None = None) -> str:
    if profile.monk_tone == 0:
        return (
            "I couldn't get a clear enough look at your skin from that photo. "
            "Could you upload another one in natural daylight, facing the camera, without filters?"
        )
    try:
        return conversation_service.get_opening_message(profile, env) or _fallback_opening(profile)
    except OpenAIError:
        # The photo analysis already succeeded; don't fail the request over the greeting.
        return _fallback_opening(profile)


def _fallback_opening(profile: SkinProfile) -> str:
    observations = []
    if profile.observed_concerns:
        observations.append("some " + ", ".join(profile.observed_concerns))
    if profile.skin_type:
        observations.append(f"what looks like {profile.skin_type} skin")
    observed = " and ".join(observations) or "skin that looks fairly balanced overall"
    return (
        f"I can see {observed}. Let me ask you a few questions to tailor your routine. "
        "What's your main skin concern right now?"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    try:
        result = vision_service.analyze_photo(request.image_base64, request.prompt)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Image analysis failed: {e}")
    profile = result.profile
    # Batch testing never shows the greeting: skip that LLM call.
    opening = _fallback_opening(profile) if request.prompt == "fairface" else build_opening_message(profile, request.environment)
    return AnalysisResponse(skin_profile=profile, opening_message=opening, vision_confidence=result.confidence)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.routine is not None:
        return _followup(request)
    try:
        context = rag_service.retrieve_relevant_products(request.skin_profile)
        reply, profile, is_ready = conversation_service.get_next_response(
            request.conversation_history, request.skin_profile, request.user_message, context, request.environment
        )
        recommendations, budget_summary = [], None
        if is_ready:
            recommendations, budget_summary = _build_routine(profile, request.environment)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")

    return ChatResponse(
        assistant_response=reply,
        updated_skin_profile=profile,
        recommendations=recommendations,
        budget=budget_summary,
        is_complete=is_ready and bool(recommendations),
    )


def _build_routine(profile: SkinProfile, environment: EnvironmentProfile | None):
    """Profile -> budget-fitted routine: ranked candidates per step, the LLM write-up, then the budget fit."""
    candidates = rag_service.retrieve_routine_candidates(profile, per_step=CANDIDATES_PER_STEP, environment=environment)
    recommendations = conversation_service.recommend(profile, candidates, environment)
    return budget.fit_routine(
        recommendations, profile.budget_monthly, profile.confirmed_concerns, rag_service.typical_routine_costs()
    )


@app.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(request: RecommendationsRequest):
    """A complete profile straight to a routine, skipping the consultation (batch testing, e.g. scripts/fairface_harness.py)."""
    try:
        recs, summary = _build_routine(request.skin_profile, request.environment)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Recommendation failed: {e}")
    return RecommendationsResponse(recommendations=recs, budget=summary)


@app.post("/environment/lookup",response_model=EnvironmentLookupResponse)
def environment_lookup(request: EnvironmentLookupRequest):
    """City or postal code -> climate summary. POST so the place never shows up in access-log URLs."""
    try:
        return environment.lookup(request.query, request.lat, request.lon)
    except environment.EnvironmentLookupError as e:
        if e.reason == "not_found":
            raise HTTPException(status_code=404, detail="We couldn't find that place.")
        raise HTTPException(status_code=503, detail="Climate data is unavailable right now.")


@app.post("/fit-budget", response_model=FitBudgetResponse)
def fit_budget(request: FitBudgetRequest):
    """Re-fit an existing routine to a new budget (or after the user removed products). No LLM call."""
    recommendations, summary = budget.fit_routine(
        request.recommendations, request.budget, request.confirmed_concerns, rag_service.typical_routine_costs()
    )
    return FitBudgetResponse(recommendations=recommendations, budget=summary)


def _followup(request: ChatRequest) -> ChatResponse:
    """After the routine is built: answer questions about it, without rebuilding anything."""
    try:
        context = rag_service.retrieve_for_question(
            request.user_message, [rec.amazon_asin for rec in request.routine]
        )
        reply = conversation_service.answer_followup(
            request.conversation_history, request.skin_profile, request.user_message, request.routine, context,
            request.environment,
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")
    # No recommendations: the client keeps showing the routine it already has.
    return ChatResponse(assistant_response=reply, updated_skin_profile=request.skin_profile, is_complete=True)


# The page itself, on the same origin as the API, so one HTTPS URL is enough to test the
# camera on a phone (see "Camera capture" in README.md). Mounted last so it can't shadow a route.
app.mount("/app", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="app")
