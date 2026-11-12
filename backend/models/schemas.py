from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class SkinProfile(BaseModel):
    # 0 is reserved as the "image too low quality to analyze" signal; 1-10 is the Monk scale.
    monk_tone: int = Field(0, ge=0, le=10)
    observed_concerns: list[str] = Field(default_factory=list)
    skin_type: str = ""
    confirmed_concerns: list[str] = Field(default_factory=list)
    # None means "not asked yet", which is distinct from a real [] / 0 / False answer.
    allergies: list[str] | None = None
    budget_monthly: int | None = None
    goals: list[str] = Field(default_factory=list)
    fragrance_free: bool | None = None
    # Product history: what has worked (or they want to keep) and what broke them out,
    # irritated, or did nothing. Answered once either list is not None.
    products_liked: list[str] | None = None
    products_disliked: list[str] | None = None


class LifestyleAnswers(BaseModel):
    """One optional lifestyle module's answers (see LIFESTYLE_MODULES in frontend/environment.js)."""
    answers: dict[str, Any] = Field(default_factory=dict)
    context: str = Field("", max_length=500)  # the module's own plain-language summary for the consultant


class EnvironmentProfile(BaseModel):
    """Where and how the user lives, as far as it can affect skin. Every field is optional."""
    place: str | None = Field(None, max_length=120)  # "Denver, Colorado, United States"; never sent to the LLM
    # Rounded to 2 decimals (about 1 km); never more precise.
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    season: Literal["winter", "spring", "summer", "autumn"] | None = None
    climate_label: Literal["hot_humid", "hot_dry", "mild", "cold_dry", "cold_humid"] | None = None
    avg_temp_c: float | None = None
    avg_humidity: float | None = Field(None, ge=0, le=100)
    uv_index_max: float | None = Field(None, ge=0, le=20)
    indoor: Literal["heated", "ac", "mixed"] | None = None
    outdoor_time: Literal["under_30m", "30m_2h", "over_2h"] | None = None
    sunscreen_habit: Literal["daily", "sometimes", "rarely"] | None = None  # "rarely" = rarely or never
    travels: bool | None = None
    source: Literal["lookup", "manual"] | None = None
    updated_at: str | None = None
    lifestyle: dict[str, LifestyleAnswers] | None = None

    @field_validator("lat", "lon")
    @classmethod
    def _round(cls, value: float | None) -> float | None:
        return None if value is None else round(value, 2)


class OptionalEnvironment(BaseModel):
    """Mixin for requests: `environment` is optional, and a malformed one is dropped instead of failing the request."""
    environment: EnvironmentProfile | None = None

    @field_validator("environment", mode="wrap")
    @classmethod
    def _lenient(cls, value, handler):
        try:
            return handler(value)
        except ValidationError:
            return None


class ConversationTurn(BaseModel):
    role: str
    content: str


class AnalysisRequest(OptionalEnvironment):
    image_base64: str
    # "fairface" is the batch-testing prompt (scripts/fairface_harness.py); the app always uses "app".
    prompt: Literal["app", "fairface"] = "app"


class ChatRequest(OptionalEnvironment):
    conversation_history: list[ConversationTurn]
    skin_profile: SkinProfile
    user_message: str
    # The routine the user is looking at. Once present, the chat answers follow-up questions
    # about it instead of gathering profile answers.
    routine: "list[RecommendationResponse] | None" = None


class RecommendationResponse(BaseModel):
    product_name: str
    brand: str
    amazon_asin: str
    routine_step: str
    why_recommended: str
    key_ingredients: list[str]
    # Up to 3 short names ("Niacinamide", "SPF 30") for the card; cleaned by code.
    ingredient_tags: list[str] = Field(default_factory=list)
    tone_note: str = ""
    not_good_for: list[str] = Field(default_factory=list)
    # Filled by code from the KB, not the model.
    routine_time: str = ""  # "AM", "PM" or "AM & PM"
    amount: str = ""        # "a pea-sized amount"
    frequency: str = ""     # "2-3 nights a week to start"
    how_to_use: str = ""
    price: str = ""
    price_usd: float = 0
    targets: list[str] = Field(default_factory=list)
    # From environment_rules, by code: a 1-3 word card tag and one plain sentence for Details.
    climate_tag: str = ""
    climate_note: str = ""
    # Alternatives are hidden until the user removes the primary pick for that step.
    is_alternative: bool = False
    # The whole step was dropped to keep the routine within budget.
    budget_left_out: bool = False


class BudgetSummary(BaseModel):
    budget: int | None = None
    low: float | None = None  # budget - tolerance
    high: float | None = None  # budget + tolerance
    total: float = 0  # upfront cost of the products in the routine
    full_total: float = 0  # upfront cost if every step's top pick were kept
    status: str = "no_budget"  # "under" | "within" | "over" | "no_budget"
    left_out_steps: list[str] = Field(default_factory=list)
    typical_basic_low: float = 0
    typical_basic: float = 0
    typical_full: float = 0


class AnalysisResponse(BaseModel):
    skin_profile: SkinProfile
    opening_message: str
    vision_confidence: str | None = None  # "high" | "medium" | "low" | "none"; fairface prompt only


class ChatResponse(BaseModel):
    assistant_response: str
    updated_skin_profile: SkinProfile
    recommendations: list[RecommendationResponse] = Field(default_factory=list)
    budget: BudgetSummary | None = None
    is_complete: bool = False


class EnvironmentLookupRequest(BaseModel):
    """Either a city or postal code (first lookup), or saved rounded coordinates (seasonal refresh)."""
    query: str | None = Field(None, min_length=2, max_length=100)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)

    @model_validator(mode="after")
    def _one_of(self):
        if not self.query and (self.lat is None or self.lon is None):
            raise ValueError("Send a query, or both lat and lon.")
        return self


class EnvironmentLookupResponse(BaseModel):
    place: str | None  # None for a coordinates refresh; the client keeps its saved place
    lat: float
    lon: float
    season: str | None  # None in the tropics
    climate_label: str
    avg_temp_c: float
    avg_humidity: float
    uv_index_max: float
    summary: str


class RecommendationsRequest(OptionalEnvironment):
    """A complete profile, straight to a routine, with no conversation (used by the FairFace harness)."""
    skin_profile: SkinProfile


class RecommendationsResponse(BaseModel):
    recommendations: list[RecommendationResponse]
    budget: BudgetSummary | None = None


class FitBudgetRequest(BaseModel):
    recommendations: list[RecommendationResponse]
    budget: int | None = Field(None, ge=0)
    confirmed_concerns: list[str] = Field(default_factory=list)


class FitBudgetResponse(BaseModel):
    recommendations: list[RecommendationResponse]
    budget: BudgetSummary


# ChatRequest refers to RecommendationResponse, which is defined after it.
ChatRequest.model_rebuild()
