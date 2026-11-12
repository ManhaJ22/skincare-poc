import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from services import conversation_service, vision_service


class FakeClient:
    """Stands in for OpenAI(): records the system prompt and answers with a fixed JSON reply."""

    def __init__(self, reply: dict | str):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.system_prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        self.system_prompts.append(messages[0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


FAIRFACE_REPLY = {"monk_tone": 7, "observed_concerns": ["dark spots"], "skin_type_estimate": "oily", "confidence": "medium"}


@pytest.fixture
def fake(monkeypatch):
    def install(reply):
        client = FakeClient(reply)
        monkeypatch.setattr(vision_service, "_get_client", lambda: client)
        return client
    return install


def test_default_uses_app_prompt(fake):
    client = fake({"monk_tone": 5, "observed_concerns": [], "skin_type": "dry"})
    result = vision_service.analyze_photo("x")
    assert client.system_prompts == [vision_service.SYSTEM_PROMPT]
    assert result.profile.skin_type == "dry" and result.confidence is None


def test_fairface_prompt_and_parsing(fake):
    client = fake(FAIRFACE_REPLY)
    result = vision_service.analyze_photo("x", "fairface")
    assert client.system_prompts == [vision_service.FAIRFACE_PROMPT]
    assert result.profile.monk_tone == 7 and result.profile.observed_concerns == ["dark spots"]
    assert result.profile.skin_type == "oily" and result.confidence == "medium"


@pytest.mark.parametrize("reply, skin_type, confidence", [
    ({"monk_tone": 0, "observed_concerns": [], "skin_type_estimate": "unknown", "confidence": "none"}, "", "none"),
    ({"monk_tone": 4, "skin_type_estimate": "normal", "confidence": "very sure"}, "normal", None),
])
def test_fairface_unknown_and_odd_values(fake, reply, skin_type, confidence):
    fake(reply)
    result = vision_service.analyze_photo("x", "fairface")
    assert result.profile.skin_type == skin_type and result.confidence == confidence


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", {"monk_tone": 42}])
def test_unparseable_reply_means_unreadable_photo(fake, reply):
    fake(reply)
    assert vision_service.analyze_photo("x", "fairface").profile.monk_tone == 0


def test_analyze_skin_photo_still_returns_profile(fake):
    fake({"monk_tone": 3, "skin_type": "normal"})
    assert vision_service.analyze_skin_photo("x").monk_tone == 3


def test_fairface_prompt_has_neutral_tie_break():
    assert "don't lean lighter or darker" in vision_service.FAIRFACE_PROMPT
    assert "err toward deeper" not in vision_service.FAIRFACE_PROMPT
    assert "pick the higher number" not in vision_service.FAIRFACE_PROMPT


def test_analyze_endpoint_fairface_returns_confidence_and_skips_greeting(fake, monkeypatch):
    client = fake(FAIRFACE_REPLY)

    def no_llm(*args, **kwargs):
        raise AssertionError("the fairface prompt should not call the LLM greeting")
    monkeypatch.setattr(conversation_service, "get_opening_message", no_llm)

    res = TestClient(main.app).post("/analyze", json={"image_base64": "x", "prompt": "fairface"})
    assert res.status_code == 200
    body = res.json()
    assert body["vision_confidence"] == "medium" and body["skin_profile"]["monk_tone"] == 7
    assert body["opening_message"]
    assert client.system_prompts == [vision_service.FAIRFACE_PROMPT]


def test_analyze_endpoint_app_default_unchanged(fake, monkeypatch):
    client = fake({"monk_tone": 6, "observed_concerns": [], "skin_type": "dry"})
    monkeypatch.setattr(conversation_service, "get_opening_message", lambda p, env=None: "Hi!")
    body = TestClient(main.app).post("/analyze", json={"image_base64": "x"}).json()
    assert body["opening_message"] == "Hi!" and body["vision_confidence"] is None
    assert client.system_prompts == [vision_service.SYSTEM_PROMPT]


def test_analyze_endpoint_rejects_unknown_prompt():
    assert TestClient(main.app).post("/analyze", json={"image_base64": "x", "prompt": "bogus"}).status_code == 422
