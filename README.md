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
- **On your LAN, with no third party:**
  1. Run `mkcert -install && mkcert 192.168.x.x`, then `cd backend && uvicorn main:app --host 0.0.0.0 --ssl-keyfile ../192.168.x.x-key.pem --ssl-certfile ../192.168.x.x.pem`.
  2. Open `https://192.168.x.x:8000/app/`.
  3. The phone must trust mkcert's root CA:
     - **iPhone:** AirDrop `rootCA.pem` from `mkcert -CAROOT`, install it, then enable it under Settings › General › About › Certificate Trust Settings.
     - **Android:** Settings › Security › Install certificate.

**Tuning the lighting check**

The thresholds are named constants at the top of `frontend/camera.js`: `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`, `MAX_SIDE_DIFF`, `MAX_CLIPPED_FRACTION`, `CLIP_HIGH`/`CLIP_LOW`, `SAMPLE_MS` and `SMOOTH_SAMPLES`. Open the page with `?lightdebug=1` (e.g. `/app/?lightdebug=1`) to overlay the live readings and the measured region.

The defaults are deliberately lenient at both ends, so neither deep nor very light skin is flagged just for its tone. A flat swatch of each of the 10 Monk reference colours passes. They have **not** been validated on real faces: before relying on them, record the debug readings for people across the full Monk scale in good and poor light, then adjust.

## Monk tone rules
Product files declare `Monk tones best`, `Irritation potential`, `White cast` (sunscreens) and `Targets`. `backend/services/tone_rules.py` uses them to exclude and re-rank products per tone band (light 1-3, medium 4-6, deep 7-10). Tune the boosts and penalties at the top of that file.

> POC only: no auth, no database, CORS open to all origins. Product ASINs/prices were checked in Oct 2026 and may change.
