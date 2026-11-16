from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from models.schemas import AnalysisRequest, ChatRequest, EnvironmentProfile, SkinProfile
from services import conversation_service, environment, environment_rules, rag_service, vision_service

DRY = EnvironmentProfile(place="Denver, Colorado, United States", lat=39.74, lon=-104.98, climate_label="cold_dry",
                         avg_temp_c=8, avg_humidity=70, uv_index_max=3, indoor="heated", sunscreen_habit="rarely")
HUMID = EnvironmentProfile(climate_label="hot_humid", avg_temp_c=29, avg_humidity=80, uv_index_max=11)


# --- Classification -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "temp, dew, humidity, label",
    [
        (29, 22, 75, "hot_humid"),
        (32, 5, 20, "hot_dry"),
        (17, 9, 55, "mild"),
        (3, -2, 80, "cold_dry"),   # cold air with high relative humidity still holds little moisture
        (7, 6, 90, "cold_humid"),
        (7, 6, 35, "cold_dry"),    # very low relative humidity counts as dry too
    ],
)
def test_climate_label(temp, dew, humidity, label):
    assert environment_rules.climate_label(temp, dew, humidity) == label


@pytest.mark.parametrize(
    "lat, month, expected",
    [(40, 1, "winter"), (40, 7, "summer"), (-33, 1, "summer"), (-33, 7, "winter"), (40, 10, "autumn"), (-33, 10, "spring"),
     (1.29, 7, None)],  # Singapore: tropics
)
def test_season_by_hemisphere(lat, month, expected):
    assert environment_rules.season(lat, date(2026, month, 15)) == expected


def test_summary_is_plain_language():
    assert environment_rules.summary(DRY) == "Currently cold and dry where you are, UV is moderate."


# --- Rules ----------------------------------------------------------------------------------
def test_guidance_without_environment():
    assert environment_rules.guidance(None).startswith("Not shared")


def test_guidance_uses_rules_but_never_the_place():
    text = environment_rules.guidance(DRY)
    assert "Dry air" in text and "rarely or never wear sunscreen" in text
    assert "Denver" not in text and "39.74" not in text and "-104.98" not in text
    assert "never say the climate or their lifestyle causes" in text


def test_guidance_includes_capped_lifestyle_context():
    env = EnvironmentProfile.model_validate({**DRY.model_dump(), "lifestyle": {"sleep": {"answers": {"hours": 6}, "context": "x" * 500}}})
    line = next(l for l in environment_rules.guidance(env).splitlines() if "Lifestyle (sleep)" in l)
    assert len(line) < 540


def test_score_boosts_hydrating_products_in_dry_air():
    rich = {"routine_step": "moisturizer", "targets": "dryness, barrier"}
    light = {"routine_step": "moisturizer", "targets": "oiliness"}
    assert environment_rules.score(rich, DRY) > 0
    assert environment_rules.score(light, DRY) == 0
    assert environment_rules.score(rich, None) == 0


def test_score_prefers_light_textures_in_humid_heat():
    rich = {"routine_step": "moisturizer", "targets": "dryness, barrier"}
    light = {"routine_step": "moisturizer", "targets": "oiliness, dehydration"}
    assert environment_rules.score(light, HUMID) > 0 > environment_rules.score(rich, HUMID)


def test_notes_match_product_and_climate():
    assert environment_rules.notes_for("moisturizer", ["dryness"], DRY)[0] == "Dry-air pick"
    tag, note = environment_rules.notes_for("moisturizer", ["oiliness"], DRY)
    assert tag == "" and "extra moisturizer at night" in note  # not a rich cream, so no "pick" tag
    assert environment_rules.notes_for("spf", ["sun protection"], HUMID)[0] == "Sun-ready"
    assert environment_rules.notes_for("spf", ["sun protection"], DRY) == ("", "")  # UV moderate, indoors
    assert environment_rules.notes_for("serum", ["acne"], DRY) == ("", "")
    assert environment_rules.notes_for("moisturizer", ["dryness"], None) == ("", "")


def test_outdoor_time_alone_triggers_sun_note():
    env = EnvironmentProfile(climate_label="mild", uv_index_max=2, outdoor_time="over_2h")
    assert environment_rules.notes_for("spf", [], env)[0] == "Sun-ready"


# --- Schema ---------------------------------------------------------------------------------
def test_coordinates_are_rounded():
    env = EnvironmentProfile(lat=39.739154, lon=-104.984703)
    assert (env.lat, env.lon) == (39.74, -104.98)


def test_requests_work_without_environment():
    assert ChatRequest(conversation_history=[], skin_profile={}, user_message="hi").environment is None
    assert AnalysisRequest(image_base64="x").environment is None


def test_malformed_environment_is_dropped_not_fatal():
    req = ChatRequest(conversation_history=[], skin_profile={}, user_message="hi", environment={"indoor": "sauna"})
    assert req.environment is None


# --- Open-Meteo client ----------------------------------------------------------------------
class _Resp:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)

    def json(self):
        return self.data


GEO = {"results": [{"name": "Denver", "admin1": "Colorado", "country": "United States",
                    "latitude": 39.73915, "longitude": -104.9847}]}
FORECAST = {"daily": {
    "temperature_2m_mean": [8.0, 9.0, 7.0],
    "relative_humidity_2m_mean": [70, 80, 75],
    "dew_point_2m_mean": [2.0, 3.0, 1.0],
    "uv_index_max": [3.5, 4.0, 2.4],
}}


def _fake_get(geo=GEO, forecast=FORECAST):
    def get(url, params, timeout):
        if url == environment.GEOCODE_URL:
            return _Resp(geo)
        assert (params["latitude"], params["longitude"]) == (39.74, -104.98)  # only rounded coordinates leave
        return _Resp(forecast)
    return get


def test_lookup_success(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get())
    result = environment.lookup("80202", today=date(2026, 11, 3))
    assert (result.place, result.lat, result.lon) == ("Denver, Colorado, United States", 39.74, -104.98)
    assert (result.season, result.climate_label, result.uv_index_max) == ("autumn", "cold_dry", 2.4)
    assert result.avg_temp_c == 8.0 and result.avg_humidity == 75
    assert result.summary == "Currently cold and dry where you are, UV is low."


def test_lookup_by_saved_coordinates_skips_geocoding(monkeypatch):
    def get(url, params, timeout):
        assert url == environment.FORECAST_URL
        assert (params["latitude"], params["longitude"]) == (39.74, -104.98)
        return _Resp(FORECAST)
    monkeypatch.setattr(httpx, "get", get)
    result = environment.lookup(lat=39.7391, lon=-104.9847, today=date(2026, 1, 10))
    assert result.place is None and result.season == "winter"
    res = TestClient(main.app).post("/environment/lookup", json={"lat": 39.74, "lon": -104.98})
    assert res.status_code == 200
    assert TestClient(main.app).post("/environment/lookup", json={"lat": 39.74}).status_code == 422


def test_lookup_not_found(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(geo={"generationtime_ms": 0.1}))
    with pytest.raises(environment.EnvironmentLookupError) as err:
        environment.lookup("zzqqxx")
    assert err.value.reason == "not_found"


def test_lookup_api_down(monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", down)
    with pytest.raises(environment.EnvironmentLookupError) as err:
        environment.lookup("Denver")
    assert err.value.reason == "unavailable"


def test_lookup_endpoint_status_codes(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(httpx, "get", _fake_get())
    ok = client.post("/environment/lookup", json={"query": "Denver"})
    assert ok.status_code == 200 and ok.json()["climate_label"] == "cold_dry"

    monkeypatch.setattr(httpx, "get", _fake_get(geo={}))
    assert client.post("/environment/lookup", json={"query": "zzqqxx"}).status_code == 404

    monkeypatch.setattr(httpx, "get", _fake_get(forecast={"daily": {}}))
    assert client.post("/environment/lookup", json={"query": "Denver"}).status_code == 503


# --- Endpoints pass the environment through, and old clients still work ----------------------
def test_chat_passes_environment_and_old_clients_work(monkeypatch):
    seen = []
    monkeypatch.setattr(rag_service, "retrieve_relevant_products", lambda profile: [])
    monkeypatch.setattr(conversation_service, "get_next_response",
                        lambda h, p, m, c, env=None, **kw: seen.append(env) or ("Any allergies?", p, False))
    client = TestClient(main.app)
    body = {"conversation_history": [], "skin_profile": {"monk_tone": 6}, "user_message": "hi"}

    assert client.post("/chat", json=body).status_code == 200  # no environment field at all
    assert client.post("/chat", json={**body, "environment": DRY.model_dump()}).status_code == 200
    assert seen[0] is None and seen[1].climate_label == "cold_dry"


def test_analyze_passes_environment(monkeypatch):
    seen = []
    monkeypatch.setattr(vision_service, "analyze_photo",
                        lambda image, prompt="app": vision_service.VisionResult(SkinProfile(monk_tone=6)))
    monkeypatch.setattr(conversation_service, "get_opening_message", lambda p, env=None: seen.append(env) or "Hi!")
    client = TestClient(main.app)

    assert client.post("/analyze", json={"image_base64": "x"}).json()["opening_message"] == "Hi!"
    client.post("/analyze", json={"image_base64": "x", "environment": {"climate_label": "hot_humid"}})
    assert seen[0] is None and seen[1].climate_label == "hot_humid"


def test_recommend_fills_climate_notes_from_code(monkeypatch):
    def hit(asin, step, targets):
        return {"doc": asin, "meta": {"asin": asin, "routine_step": step, "price": "~$10", "price_usd": 10.0,
                "targets": targets, "routine_time": "AM", "amount": "a", "frequency": "f", "how_to_use": "h"}}
    candidates = {"moisturizer": [hit("m1", "moisturizer", "dryness, barrier")], "spf": [hit("s1", "spf", "sun protection")]}
    item = {"product_name": "x", "brand": "b", "why_recommended": "w", "key_ingredients": ["Glycerin"],
            "climate_tag": "made up by the model"}
    prompts = []

    def fake(messages, temperature):
        prompts.append(messages[0]["content"])
        return {"recommendations": [{**item, "amazon_asin": "m1"}, {**item, "amazon_asin": "s1"}]}
    monkeypatch.setattr(conversation_service, "_chat_json", fake)

    recs = conversation_service.recommend(SkinProfile(monk_tone=5), candidates, DRY)
    assert [(r.climate_tag, bool(r.climate_note)) for r in recs] == [("Dry-air pick", True), ("", False)]
    assert "Dry air" in prompts[0] and "Denver" not in prompts[0]

    recs = conversation_service.recommend(SkinProfile(monk_tone=5), candidates)  # no environment
    assert all(r.climate_tag == "" and r.climate_note == "" for r in recs)
