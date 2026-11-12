# Dermalis — AI Skincare Consultant (POC)

Local proof-of-concept: photo → skin analysis (GPT-4o Vision) → conversational profile building → RAG-based product recommendations.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r backend/requirements.txt
   ```
2. Copy `backend/.env.example` to `backend/.env` and add your OpenAI API key.
3. Start the API:
   ```
   cd backend && uvicorn main:app --reload
   ```
4. Open `frontend/index.html` in a browser (or http://localhost:8000/app/, served by the backend).
5. Take or choose a photo and start the conversation.

Run the tests (no API key needed): `cd backend && pytest`

Planning a mobile app? See `docs/react-native-migration.md` for the recommended Expo + NativeWind stack and how the current design maps onto it.

## Camera capture
"Take a photo" in step 1 opens an in-page camera (`frontend/camera.js`). "Choose from library" (the old file input) stays below it.

**How it works**
1. **Intro.** A short explanation appears before the browser asks for camera access. It's skipped once access is already granted.
2. **Live view.**
   - The front camera, mirrored on screen only, with an oval face guide.
   - Every 400 ms a lighting check measures the pixels inside the oval: brightness, the left/right difference and the share of clipped pixels.
   - The status line says what to fix. Poor light restyles the shutter but never blocks it.
   - "Switch camera" appears when the device has more than one camera.
3. **Capture.** The frame is drawn at the video's native resolution, un-mirrored, with no filters or overlay, as a JPEG at 0.92 quality.
4. **Review.** The same lighting check runs on the photo. Good light offers "Use this photo" and "Retake". Poor light makes "Retake" the main button, with "Use anyway" next to it.
5. **Accept.** The photo goes through the same resize and `/analyze` request as a library pick (`{ image_base64: "data:image/jpeg;base64,…" }`), and analysis starts.

The camera turns off when the dialog closes, a photo is taken, or the page is hidden. Photos are never written to localStorage or IndexedDB, and preview URLs are revoked. When the camera isn't available (no HTTPS, permission denied, no camera, camera busy), the page says why and offers the library instead.

**Testing on a phone (HTTPS)**

Phones only allow the camera over HTTPS (desktop browsers also allow `localhost`). The backend serves the page at `/app/`, and the page then calls the API on the same origin, so one HTTPS URL covers both.
- **Quickest: a tunnel.** Run `cloudflared tunnel --url http://localhost:8000` and open `https://<random>.trycloudflare.com/app/` on the phone. Photos then pass through Cloudflare, so use it for testing only.
- **Or ngrok** (free account; run `ngrok config add-authtoken <token>` once): `ngrok http 8000 --basic-auth "dermalis:<password>"`, then open `https://<random>.ngrok-free.dev/app/`. The password stops anyone who finds the URL from spending your OpenAI credits. On the phone, tap "Visit Site" once on ngrok's warning page; the page's own API calls then work.
- **On your LAN, with no third party:**
  1. Run `mkcert -install && mkcert 192.168.x.x`, then `cd backend && uvicorn main:app --host 0.0.0.0 --ssl-keyfile ../192.168.x.x-key.pem --ssl-certfile ../192.168.x.x.pem`.
  2. Open `https://192.168.x.x:8000/app/`.
  3. The phone must trust mkcert's root CA:
     - **iPhone:** AirDrop `rootCA.pem` from `mkcert -CAROOT`, install it, then enable it under Settings › General › About › Certificate Trust Settings.
     - **Android:** Settings › Security › Install certificate.

**Tuning the lighting check**

The thresholds are named constants at the top of `frontend/camera.js`: `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`, `MAX_SIDE_DIFF`, `MAX_CLIPPED_FRACTION`, `CLIP_HIGH`/`CLIP_LOW`, `SAMPLE_MS` and `SMOOTH_SAMPLES`. Open the page with `?lightdebug=1` (e.g. `/app/?lightdebug=1`) to overlay the live readings and the measured region.

The defaults are deliberately lenient at both ends, so neither deep nor very light skin is flagged just for its tone. A flat swatch of each of the 10 Monk reference colours passes. They have **not** been validated on real faces: before relying on them, record the debug readings for people across the full Monk scale in good and poor light, then adjust.

## Climate and environment
After the photo is analyzed, an optional "Your climate" card appears in the chat. The user can type a city or ZIP code, or choose a climate, then answer four single-tap questions (indoor heating/AC, time outdoors, sunscreen habit, travel), or skip. To edit or clear it later, use Settings (the gear icon).

**What is collected and where it lives**
- **City or ZIP code:** sent once to our backend (`POST /environment/lookup`), which asks Open-Meteo (no API key) for the place and the last 14 days of temperature, humidity, dew point and UV. Coordinates are rounded to 2 decimals (about 1 km).
- **The profile:** the place name, rounded coordinates, climate label, season, averages and the four answers are saved **only in this browser** (`localStorage["dermalis.environment"]`). The backend stores nothing.
- **What gets sent:** `/analyze` and `/chat` receive an optional `environment` field with the climate facts and answers, but **not** the place or coordinates. The LLM prompts never include the place either.
- **Settings:** `localStorage["dermalis.settings"]` holds the lifestyle toggle and when the seasonal prompt was last dismissed.
- **Clearing:** Settings → "Clear my location and environment data" removes the profile, any lifestyle answers and the seasonal timer.
- **Logging:** nothing logs places or coordinates. The lookup is a POST, so they don't appear in access-log URLs either.
- **Old clients:** they keep working. `environment` is optional, and a malformed one is dropped rather than failing the request.

**How it changes suggestions** (rules: `backend/services/environment_rules.py`)
- **Dry air** (cold and dry, hot and dry, under 40% humidity, or heated indoors): richer, barrier-supporting moisturizers and gentle cleansers rank higher. Suggests a little extra moisturizer at night, and going easy on exfoliating.
- **Hot and humid:** lighter textures rank higher, with "a thin layer is usually enough".
- **Strong sun** (UV 6+ or over 2 hours outdoors): sunscreen gets a "Sun-ready" note about reapplying. A "rarely" sunscreen habit gets a gentle nudge from the consultant.
- **Air-conditioning or frequent travel:** consultant guidance only.

Each rule can give a product a short card tag, plus one plain sentence in Details and the How-to-use panel. It can also nudge ranking slightly using the KB `Targets` field, and adds guidance to the consultant's prompts. Climate is never described as a cause.

To tune the rules, edit the thresholds at the top of the file (dew point is used for dry vs humid) or add a `Rule` to `RULES`.

**Seasonal refresh.** On page open, if the saved season has turned over (computed locally) or the profile is over 60 days old, the page re-checks the climate through our backend using the saved rounded coordinates. It then shows one prompt, e.g. "Winter's here and it's gotten drier where you are. Want a seasonal routine update?". "Not now" keeps it quiet for 30 days. Seasons aren't used in the tropics.

**Adding a lifestyle module.** Settings has "Deeper lifestyle questions" (off by default). Modules are registered in `LIFESTYLE_MODULES` in `frontend/environment.js`:
- Each module is `{ id, title, questions, toContext(answers) }`, and `questions` uses the same shape as the climate questions.
- Only modules with questions are asked, and only while the setting is on.
- Answers are sent as `environment.lifestyle[id] = { answers, context }`, and the backend adds `context` (capped at 500 characters) to the consultant's prompt.
- Only a placeholder is registered today.

**Manual test checklist**
- **Lookup:** "Denver" or "80202" shows "Currently … where you are, UV is …". Answer the questions, then Save; the chat request carries `environment` without place, lat or lon.
- **Not found:** "zzqqxx" shows the not-found message with a retry. UK postcodes aren't found by Open-Meteo, so use the climate chips.
- **Weather API down** (stop your network, or stub a 503): shows "Choose your climate below" and onboarding continues.
- **Manual:** pick a climate chip, answer, Save. Also try "Not right? Choose your climate" after a lookup.
- **Skip:** the card closes, the checklist item is marked done, and recommendations work without `environment`.
- **Saved profile:** start over and analyze again; no card appears, and `/analyze` carries `environment`.
- **Clear data:** Settings → Clear; `localStorage` has no `dermalis.environment`.
- **Seasonal prompt:** set `updated_at` to 70 days ago and `season` to last season in the saved profile, then reload. Click "Not now", reload, and confirm nothing shows.
- **Old clients:** `pytest tests/test_environment.py` covers requests with no `environment` field.

## Monk tone rules
Product files declare `Monk tones best`, `Irritation potential`, `White cast` (sunscreens) and `Targets`. `backend/services/tone_rules.py` uses them to exclude and re-rank products per tone band (light 1-3, medium 4-6, deep 7-10). Tune the boosts and penalties at the top of that file.

> POC only: no auth, no database, CORS open to all origins. Product ASINs/prices were checked in Oct 2026 and may change.
