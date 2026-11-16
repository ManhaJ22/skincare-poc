import uuid

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

import main
from models.schemas import Milestone, ProgressSession, RecommendationResponse, SkinProfile
from services import conversation_service, progress_service, progress_store, rag_service

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def rec(asin, step="serum", **kw):
    fields = dict(product_name=f"Product {asin}", brand="Brand", amazon_asin=asin, routine_step=step,
                  why_recommended="", key_ingredients=["Niacinamide 10%"], ingredient_tags=["Niacinamide"],
                  targets=["dark marks"])
    return RecommendationResponse(**{**fields, **kw})


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_store, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def timeline_calls(monkeypatch):
    calls = []

    def fake(profile, recs):
        calls.append([r.amazon_asin for r in recs])
        return [Milestone(asin=r.amazon_asin) for r in recs], "You wanted fewer dark marks."

    monkeypatch.setattr(progress_service, "build_timeline", fake)
    return calls


def body(session_id, *asins, **kw):
    return {"session_id": session_id, "monk_tone": 6, "observed_concerns": ["hyperpigmentation"],
            "skin_type_estimate": "combination", "recommendations": [rec(a).model_dump() for a in asins], **kw}


# --- Store ------------------------------------------------------------------------------------
def test_store_round_trip_and_latest():
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    progress_store.save_session(ProgressSession(session_id=a, created_at="2026-11-01T00:00:00+00:00"))
    progress_store.save_session(ProgressSession(session_id=b, created_at="2026-11-10T00:00:00+00:00"))
    assert progress_store.get_session(a).start_date == "2026-11-01"
    assert progress_store.latest_session().session_id == b


def test_upsert_keeps_photos_and_start_date():
    sid = str(uuid.uuid4())
    progress_store.save_session(ProgressSession(session_id=sid))
    progress_store.set_start_date(sid, "2026-09-01")
    progress_store.add_photo(sid, JPEG)
    saved = progress_store.save_session(ProgressSession(session_id=sid, start_date="2026-11-16"))
    assert saved.start_date == "2026-09-01" and len(saved.photos) == 1


@pytest.mark.parametrize("bad", ["../x", "not-a-uuid", "", "..\\..\\etc"])
def test_bad_session_ids_are_rejected(bad):
    with pytest.raises(progress_store.InvalidId):
        progress_store.get_session(bad)


def test_bad_photo_id_is_rejected():
    with pytest.raises(progress_store.InvalidId):
        progress_store.photo_path(str(uuid.uuid4()), "../../sessions/x")


def test_photo_add_path_delete(data_dir):
    sid = str(uuid.uuid4())
    progress_store.save_session(ProgressSession(session_id=sid))
    photo = progress_store.add_photo(sid, PNG, "2026-11-20", "week 1")
    path = progress_store.photo_path(sid, photo.photo_id)
    assert path.suffix == ".png" and path.read_bytes() == PNG and data_dir in path.parents
    assert progress_store.delete_photo(sid, photo.photo_id)
    assert progress_store.photo_path(sid, photo.photo_id) is None
    assert progress_store.get_session(sid).photos == []


def test_non_image_and_oversized_uploads_are_rejected():
    sid = str(uuid.uuid4())
    progress_store.save_session(ProgressSession(session_id=sid))
    with pytest.raises(progress_store.InvalidPhoto):
        progress_store.add_photo(sid, b"<html>not an image</html>")
    with pytest.raises(progress_store.InvalidPhoto):
        progress_store.add_photo(sid, JPEG + b"\x00" * progress_store.MAX_PHOTO_BYTES)


# --- Endpoints --------------------------------------------------------------------------------
def test_save_builds_timeline_once_and_rebuilds_on_change(client, timeline_calls):
    sid = str(uuid.uuid4())
    first = client.post("/sessions", json=body(sid, "A1", "B2")).json()
    assert [m["asin"] for m in first["timeline"]] == ["A1", "B2"]
    assert first["conversation_summary"] == "You wanted fewer dark marks."
    client.post("/sessions", json=body(sid, "B2", "A1"))  # same routine, different order
    assert len(timeline_calls) == 1
    client.post("/sessions", json=body(sid, "A1", "C3"))
    assert timeline_calls[-1] == ["A1", "C3"]


def test_left_out_and_alternative_products_are_not_tracked(client, timeline_calls):
    payload = body(str(uuid.uuid4()), "A1")
    payload["recommendations"] += [rec("B2", budget_left_out=True).model_dump(), rec("C3", is_alternative=True).model_dump()]
    saved = client.post("/sessions", json=payload).json()
    assert [r["amazon_asin"] for r in saved["recommendations"]] == ["A1"]


def test_openai_failure_still_saves_session(client, monkeypatch):
    def boom(*a):
        raise OpenAIError("no credits")

    monkeypatch.setattr(progress_service, "build_timeline", boom)
    sid = str(uuid.uuid4())
    res = client.post("/sessions", json=body(sid, "A1"))
    assert res.status_code == 200
    assert res.json()["timeline"] == [] and "no credits" in res.json()["timeline_error"]
    assert client.get(f"/sessions/{sid}").status_code == 200


def test_photo_endpoints(client, timeline_calls):
    sid = str(uuid.uuid4())
    client.post("/sessions", json=body(sid, "A1"))
    up = client.post(f"/sessions/{sid}/photos", files={"file": ("me.jpg", JPEG, "image/jpeg")},
                     data={"taken_at": "2026-11-20", "note": "baseline"})
    assert up.status_code == 200
    photo_id = up.json()["photo_id"]
    assert client.get(f"/sessions/{sid}").json()["photos"][0]["note"] == "baseline"
    got = client.get(f"/sessions/{sid}/photos/{photo_id}")
    assert got.status_code == 200 and got.content == JPEG
    fake = client.post(f"/sessions/{sid}/photos", files={"file": ("x.jpg", b"GIF89a....", "image/jpeg")})
    assert fake.status_code == 422
    assert client.delete(f"/sessions/{sid}/photos/{photo_id}").status_code == 204
    assert client.get(f"/sessions/{sid}/photos/{photo_id}").status_code == 404


def test_patch_start_date_and_errors(client, timeline_calls):
    sid = str(uuid.uuid4())
    assert client.get("/sessions/latest").status_code == 404
    client.post("/sessions", json=body(sid, "A1"))
    assert client.patch(f"/sessions/{sid}", json={"start_date": "2026-09-01"}).json()["start_date"] == "2026-09-01"
    assert client.patch(f"/sessions/{sid}", json={"start_date": "2026-02-31"}).status_code == 422
    assert client.get("/sessions/latest").json()["session_id"] == sid
    assert client.get(f"/sessions/{uuid.uuid4()}").status_code == 404
    assert client.get("/sessions/not-a-uuid").status_code == 400
    assert client.post("/sessions", json=body("../evil", "A1")).status_code == 400


# --- Timeline generation ------------------------------------------------------------------------
def test_clean_milestones_clamps_orders_and_fills_gaps():
    recs = [rec("A1"), rec("S1", step="spf", ingredient_tags=["SPF 50"])]
    raw = [
        {"asin": "A1", "first_signs_weeks": [12, 4], "full_effect_weeks": [2, 99], "kind": "active",
         "ingredient": "Niacinamide", "concern": "dark marks", "what_to_look_for": "Marks fading."},
        {"asin": "ZZZ", "first_signs_weeks": [1, 2]},  # not in the routine
        "junk",
    ]
    out = progress_service.clean_milestones(raw, recs)
    assert [m.asin for m in out] == ["A1", "S1"]
    assert out[0].first_signs_weeks == [4, 12]
    assert out[0].full_effect_weeks == [4, 52]  # never before first signs, capped at a year
    assert out[1].kind == "protective" and out[1].full_effect_weeks == [0, 0]


def test_build_timeline_grounds_in_ingredient_kb(monkeypatch):
    queries, prompts = [], []

    def fake_retrieve(q, n_results=4):
        queries.append(q)
        return ["[niacinamide - ONSET] 8-12 weeks"]

    def fake_chat(messages, temperature):
        prompts.append(messages[0]["content"])
        return {
            "conversation_summary": "You wanted even tone.",
            "milestones": [{"asin": "A1", "first_signs_weeks": [8, 12], "full_effect_weeks": [16, 24]}],
        }

    monkeypatch.setattr(rag_service, "retrieve_ingredient_context", fake_retrieve)
    monkeypatch.setattr(conversation_service, "_chat_json", fake_chat)
    profile = SkinProfile(monk_tone=7, confirmed_concerns=["dark marks"])
    milestones, summary = progress_service.build_timeline(profile, [rec("A1")])
    assert "Niacinamide" in queries[0] and "dark marks" in queries[0] and "Monk tone 7" in queries[0]
    assert "8-12 weeks" in prompts[0] and "asin A1" in prompts[0]
    assert milestones[0].full_effect_weeks == [16, 24] and summary == "You wanted even tone."
