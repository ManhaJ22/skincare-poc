from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError

from models.schemas import AnalysisRequest, AnalysisResponse, ChatRequest, ChatResponse, SkinProfile
from services import conversation_service, rag_service, vision_service

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


def build_opening_message(profile: SkinProfile) -> str:
    if profile.monk_tone == 0:
        return (
            "I couldn't get a clear enough look at your skin from that photo. "
            "Could you upload another one in natural daylight, facing the camera, without filters?"
        )
    try:
        return conversation_service.get_opening_message(profile) or _fallback_opening(profile)
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
        profile = vision_service.analyze_skin_photo(request.image_base64)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Image analysis failed: {e}")
    return AnalysisResponse(skin_profile=profile, opening_message=build_opening_message(profile))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        context = rag_service.retrieve_relevant_products(request.skin_profile)
        reply, profile, is_ready = conversation_service.get_next_response(
            request.conversation_history, request.skin_profile, request.user_message, context
        )
        recommendations = []
        if is_ready:
            candidates = rag_service.retrieve_routine_candidates(profile, per_step=CANDIDATES_PER_STEP)
            recommendations = conversation_service.recommend(profile, candidates)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")

    return ChatResponse(
        assistant_response=reply,
        updated_skin_profile=profile,
        recommendations=recommendations,
        is_complete=is_ready and bool(recommendations),
    )
