# ID Verification Backend

FastAPI service that orchestrates document (OCR/tamper) checks and
biometric (face match/liveness) checks, then returns a single pass/fail
JSON response.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for an interactive Swagger UI
where you can upload two test images and see the response immediately —
no frontend needed yet.

## API Contract

### `POST /verify`
**Request:** `multipart/form-data`
- `id_image`: file (jpeg/png/webp)
- `selfie_image`: file (jpeg/png/webp)

**Response:** `200 OK`
```json
{
  "status": "fail",
  "document_check": {
    "tampered": true,
    "confidence": 0.87,
    "flagged_field": "date_of_birth",
    "notes": "Inconsistent font/spacing detected around 'date_of_birth' field."
  },
  "biometric_check": {
    "match_score": 0.92,
    "liveness_passed": true,
    "liveness_confidence": 0.95,
    "notes": "Face match and liveness check completed successfully."
  },
  "overall_reason": "Document appears tampered (field: date_of_birth)."
}
```

**Error responses:**
- `400` — bad/missing/oversized image
- `500` — ML model crashed
- `502` — ML model returned output that didn't match the expected schema

### `GET /history?limit=50&status=fail`
Returns recent verification results, most recent first. `status` filter is optional (`pass` or `fail`). Built for the Admin Dashboard's results list/table.

### `GET /history/{record_id}`
Returns one saved verification record by its `id` — for a dashboard detail view when an admin clicks into a specific case.

### `GET /stats`
Returns aggregate counts (`total_verifications`, `passed`, `failed`, `pass_rate`) for a dashboard summary widget.

## Database (PostgreSQL)

Users, login sessions and verification history are stored in **PostgreSQL**
(via SQLAlchemy), so nothing is lost on restart. Tables (`users`, `sessions`,
`verifications`) are created automatically on first startup.

```bash
docker compose up -d                 # start PostgreSQL (or use your own server)
cp .env.example .env                 # set DATABASE_URL if not using defaults
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`DATABASE_URL` format: `postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME`

Notes:
- Passwords are hashed with bcrypt; sessions expire after 7 days.
- API endpoints and JSON shapes are unchanged from the in-memory version.
- For schema changes later, add Alembic migrations.

## Current State: MOCK MODE

`models/document_check.py` and `models/biometric_check.py` currently
return **randomized fake results** so the frontend and dashboard teams
can build and test against a stable API right now.

## Integration Checklist for ML Teammates

**Document/OCR teammate:**
- Open `models/document_check.py`
- Replace the body of `run_document_check(image_bytes)` with real logic
- Keep the return dict shape exactly the same (see docstring in that file)

**Biometric teammate:**
- Open `models/biometric_check.py`
- Replace the body of `run_biometric_check(id_image_bytes, selfie_image_bytes)`
- Keep the return dict shape exactly the same (see docstring in that file)

No changes to `main.py` should be needed — it calls your functions by
name and validates the output against `schemas.py` automatically. If
your model is slow, that's fine: it's already run in a background thread
so it won't block other requests.

## Tuning

- Face match threshold is set in `main.py`: `FACE_MATCH_THRESHOLD = 0.75`
  — adjust based on real model calibration.
- CORS is currently wide open (`allow_origins=["*"]`) for easy dev/demo.
  Tighten before any public deployment.

## Face Analysis Features (biometric check)

`models/biometric_check.py` now does real face analysis when DeepFace is installed:

| Feature | How | Used for |
|---|---|---|
| **Face embeddings** | ArcFace (512-d vector per face) + cosine distance | Main `match_score` |
| **Facial landmarks** | RetinaFace 5 points (eyes, nose, mouth) -> face proportions compared | `landmark_score` (shown on dashboard, not part of pass/fail) |
| **Age-related changes** | Age estimated on both photos; big gap -> match threshold relaxed by up to 0.05 | `age_gap_years`, `threshold_used` |
| **Liveness** | DeepFace anti-spoofing on the selfie | `liveness_passed` |

New optional fields in `biometric_check`: `embedding_model`, `embedding_distance`,
`landmark_score`, `id_age_estimate`, `selfie_age_estimate`, `age_gap_years`,
`threshold_used`, `analysis_mode` (`real` or `mock`). Existing fields are unchanged.

Install (Python 3.10-3.12):
```bash
pip install -r requirements-ml.txt
```
First startup downloads model weights (needs internet, can take a few minutes).
Without these packages the server still starts and uses MOCK mode
(`analysis_mode: "mock"`).

Notes:
- Age estimation is approximate (about +/-5 years); the relaxation is small and capped.
- Face embeddings are never returned or saved to the database.


## Document Tamper Model (optional)

`models/document_check.py` can run a trained classifier that flags edited ID fields
(`full_name`, `date_of_birth`, `expiry_date`, `id_number`) and names the field in `flagged_field`.

- Train it with the scripts in `ml/` (see `ml/README.md`). It trains on **synthetic cards**, so treat it as a
  prototype until it is fine-tuned on real examples.
- Put the result at `models/document_tamper.pt` and run `pip install -r requirements-doc.txt`.
- Without the file (or without PyTorch) the document check stays in mock mode.
- `document_check.analysis_mode` in the API response is `"model"` or `"mock"`.
