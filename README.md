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
4. Open `frontend/index.html` in a browser.
5. Upload a photo and start the conversation.

Run the tests (no API key needed): `cd backend && pytest`

Planning a mobile app? See `docs/react-native-migration.md` for the recommended Expo + NativeWind stack and how the current design maps onto it.

## Monk tone rules
Product files declare `Monk tones best`, `Irritation potential`, `White cast` (sunscreens) and `Targets`. `backend/services/tone_rules.py` uses them to exclude and re-rank products per tone band (light 1-3, medium 4-6, deep 7-10). Tune the boosts and penalties at the top of that file.

> POC only: no auth, no database, CORS open to all origins. Product ASINs/prices were checked in Oct 2026 and may change.
