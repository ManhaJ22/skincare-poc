from pydantic import BaseModel, Field


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


class ConversationTurn(BaseModel):
    role: str
    content: str


class AnalysisRequest(BaseModel):
    image_base64: str


class ChatRequest(BaseModel):
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


class ChatResponse(BaseModel):
    assistant_response: str
    updated_skin_profile: SkinProfile
    recommendations: list[RecommendationResponse] = Field(default_factory=list)
    budget: BudgetSummary | None = None
    is_complete: bool = False


class FitBudgetRequest(BaseModel):
    recommendations: list[RecommendationResponse]
    budget: int | None = Field(None, ge=0)
    confirmed_concerns: list[str] = Field(default_factory=list)


class FitBudgetResponse(BaseModel):
    recommendations: list[RecommendationResponse]
    budget: BudgetSummary


# ChatRequest refers to RecommendationResponse, which is defined after it.
ChatRequest.model_rebuild()
