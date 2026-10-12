# SkinSage — AI Skincare Consultant (POC)

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

> POC only: no auth, no database, CORS open to all origins. Product ASINs/prices were checked in Oct 2026 and may change.
