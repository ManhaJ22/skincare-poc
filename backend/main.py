from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    PhotoMeta,
    ProgressSession,
    RecommendationsRequest,
    RecommendationsResponse,
    SaveSessionRequest,
    SkinProfile,
    UpdateSessionRequest,
)
from services import (
    budget,
    conversation_service,
    environment,
    progress_service,
    progress_store,
    rag_service,
    vision_service,
)

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
        ingredient_context = rag_service.retrieve_ingredient_context(_ingredient_query(request))
        reply, profile, is_ready = conversation_service.get_next_response(
            request.conversation_history, request.skin_profile, request.user_message, context, request.environment,
            ingredient_context=ingredient_context,
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


def _ingredient_query(request: ChatRequest) -> str:
    """The user's message plus their concerns: a bare "yes, that's right" retrieves nothing useful on its own."""
    profile = request.skin_profile
    concerns = list(dict.fromkeys(profile.confirmed_concerns + profile.observed_concerns))
    return f"{request.user_message}; concerns: {', '.join(concerns)}" if concerns else request.user_message


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
        ingredient_context = rag_service.retrieve_ingredient_context(request.user_message)
        reply = conversation_service.answer_followup(
            request.conversation_history, request.skin_profile, request.user_message, request.routine, context,
            request.environment, ingredient_context=ingredient_context,
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")
    # No recommendations: the client keeps showing the routine it already has.
    return ChatResponse(assistant_response=reply, updated_skin_profile=request.skin_profile, is_complete=True)


# --- Progress tracking: one saved session per finished routine, plus progress photos (no auth: demo) ---
def _session_or_404(session_id: str) -> ProgressSession:
    try:
        session = progress_store.get_session(session_id)
    except progress_store.InvalidId as e:
        raise HTTPException(status_code=400, detail=str(e))
    if session is None:
        raise HTTPException(status_code=404, detail="No saved routine found.")
    return session


@app.post("/sessions", response_model=ProgressSession)
def save_session(request: SaveSessionRequest):
    """Save (or re-save) the routine the user is looking at; builds the predicted timeline when the routine changed."""
    try:
        existing = progress_store.get_session(request.session_id)
    except progress_store.InvalidId as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = ProgressSession(**request.model_dump())
    routine = [rec for rec in request.recommendations if not rec.budget_left_out and not rec.is_alternative]
    session.recommendations = routine
    key = sorted(rec.amazon_asin for rec in routine)
    if existing and existing.timeline_key == key and existing.timeline:
        session.timeline, session.timeline_key = existing.timeline, existing.timeline_key
        session.conversation_summary = existing.conversation_summary
    else:
        profile = request.skin_profile or SkinProfile(
            monk_tone=request.monk_tone, observed_concerns=request.observed_concerns, skin_type=request.skin_type_estimate
        )
        try:
            session.timeline, session.conversation_summary = progress_service.build_timeline(profile, routine)
            session.timeline_key = key
        except OpenAIError as e:
            # Keep the session (and any photos); the timeline is rebuilt on the next save.
            session.timeline_error = f"Couldn't predict the timeline right now: {e}"
            if existing:
                session.conversation_summary = existing.conversation_summary
    return progress_store.save_session(session)


@app.get("/sessions/latest", response_model=ProgressSession)
def latest_session():
    session = progress_store.latest_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No saved routine yet.")
    return session


@app.get("/sessions/{session_id}", response_model=ProgressSession)
def get_session(session_id: str):
    return _session_or_404(session_id)


@app.patch("/sessions/{session_id}", response_model=ProgressSession)
def update_session(session_id: str, request: UpdateSessionRequest):
    _session_or_404(session_id)
    try:
        return progress_store.set_start_date(session_id, request.start_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Not a valid date.")


@app.post("/sessions/{session_id}/photos", response_model=PhotoMeta)
def upload_photo(session_id: str, file: UploadFile = File(...), taken_at: str | None = Form(None), note: str = Form("")):
    _session_or_404(session_id)
    data = file.file.read(progress_store.MAX_PHOTO_BYTES + 1)
    try:
        return progress_store.add_photo(session_id, data, taken_at, note)
    except progress_store.InvalidPhoto as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/sessions/{session_id}/photos/{photo_id}")
def get_photo(session_id: str, photo_id: str):
    try:
        path = progress_store.photo_path(session_id, photo_id)
    except progress_store.InvalidId as e:
        raise HTTPException(status_code=400, detail=str(e))
    if path is None:
        raise HTTPException(status_code=404, detail="Photo not found.")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


@app.delete("/sessions/{session_id}/photos/{photo_id}", status_code=204)
def delete_photo(session_id: str, photo_id: str):
    try:
        deleted = progress_store.delete_photo(session_id, photo_id)
    except progress_store.InvalidId as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Photo not found.")


# The page itself, on the same origin as the API, so one HTTPS URL is enough to test the
# camera on a phone (see "Camera capture" in README.md). Mounted last so it can't shadow a route.
app.mount("/app", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="app")
