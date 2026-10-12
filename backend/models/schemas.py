from pydantic import BaseModel, Field


class SkinProfile(BaseModel):
    # 0 is reserved as the "image too low quality to analyze" signal; 1-10 is the Monk scale.
    monk_tone: int = Field(0, ge=0, le=10)
    observed_concerns: list[str] = Field(default_factory=list)
    skin_type: str = ""
    confirmed_concerns: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    # None means "not asked yet", which is distinct from a real 0 / False answer.
    budget_monthly: int | None = None
    goals: list[str] = Field(default_factory=list)
    fragrance_free: bool | None = None


class ConversationTurn(BaseModel):
    role: str
    content: str


class AnalysisRequest(BaseModel):
    image_base64: str


class ChatRequest(BaseModel):
    conversation_history: list[ConversationTurn]
    skin_profile: SkinProfile
    user_message: str


class RecommendationResponse(BaseModel):
    product_name: str
    brand: str
    amazon_asin: str
    routine_step: str
    why_recommended: str
    key_ingredients: list[str]


class AnalysisResponse(BaseModel):
    skin_profile: SkinProfile
    opening_message: str


class ChatResponse(BaseModel):
    assistant_response: str
    updated_skin_profile: SkinProfile
    recommendations: list[RecommendationResponse] = Field(default_factory=list)
    is_complete: bool = False
