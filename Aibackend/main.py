"""
MAIN BACKEND SERVICE (PostgreSQL-backed, hardened for deployment)

What it does:
1. Takes an ID image + selfie from a logged-in operator and validates the uploads.
2. Runs the document check and the biometric check in parallel.
3. Combines them into pass/fail, saves the result, and returns clean JSON.
4. Serves history/stats to the dashboard: operators see their own records, admins see everyone's.

Settings come from environment variables - see config.py and .env.example.

Run locally:
    python -m uvicorn main:app --reload --port 8000
"""

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

import auth
import config
from database import SessionLocal, get_db, init_db
from db_models import User, Verification
from models import biometric_check as bio_module
from models import document_check as doc_module
from models.biometric_check import BASE_MATCH_THRESHOLD, run_biometric_check, warm_up
from models.document_check import run_document_check
from schemas import BiometricCheckResult, DocumentCheckResult, VerificationResponse
from security import client_ip, limiter
from utils import MAX_FILE_SIZE_MB, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification-backend")

FACE_MATCH_THRESHOLD = BASE_MATCH_THRESHOLD     # the biometric module may lower it slightly for big age gaps
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{3,64}$")


# ------------------------------------------------------------ startup
def check_models() -> None:
    """In production, refuse to run with simulated results unless that was chosen on purpose."""
    problems = []
    if not bio_module.REAL_MODE:
        problems.append("face analysis is simulated (DeepFace is not installed)")
    if not doc_module.model_available():
        problems.append("there is no document tamper model (models/document_tamper.pt)")
    if not problems:
        return
    text = " and ".join(problems)
    if config.IS_PROD and not config.ALLOW_MOCK:
        raise RuntimeError(
            f"Refusing to start in production: {text}. Fix that, or set ALLOW_MOCK=true to run a demo "
            f"whose results are simulated."
        )
    logger.warning("RUNNING WITH SIMULATED RESULTS: %s.", text)


def purge_old_data() -> None:
    with SessionLocal() as db:
        removed = auth.purge_expired_sessions(db)
        if config.RETENTION_DAYS > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=config.RETENTION_DAYS)
            result = db.execute(delete(Verification).where(Verification.timestamp < cutoff))
            db.commit()
            if result.rowcount:
                logger.info("Retention: deleted %s verification records older than %s days.",
                            result.rowcount, config.RETENTION_DAYS)
        if removed:
            logger.info("Removed %s expired sessions.", removed)


async def maintenance_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(purge_old_data)
        except Exception:
            logger.exception("Maintenance run failed")
        await asyncio.sleep(24 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                                   # create tables / add new columns
    check_models()
    await asyncio.to_thread(warm_up)            # load face models (no-op in mock mode)
    task = asyncio.create_task(maintenance_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(
    title="ID Verification Backend",
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None if config.IS_PROD else "/docs",
    redoc_url=None,
    openapi_url=None if config.IS_PROD else "/openapi.json",
)

# While developing, any page opened from localhost / 127.0.0.1 (on ANY port, e.g. Live Server on 5500 or
# 5501) may call the API. In production only the origins listed in ALLOWED_ORIGINS are allowed.
DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_origin_regex=None if config.IS_PROD else DEV_ORIGIN_REGEX,
    allow_credentials=False,                    # we use Bearer tokens, not cookies
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


def _origin_allowed(origin: str) -> bool:
    return origin in config.ALLOWED_ORIGINS or (not config.IS_PROD and re.match(DEV_ORIGIN_REGEX, origin) is not None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # If a browser page is being blocked, say so in the terminal instead of failing silently
    origin = request.headers.get("origin")
    if origin and not _origin_allowed(origin) and origin.split("://", 1)[-1] != request.headers.get("host"):
        logger.warning("A page opened from %s was blocked by CORS. Add it to ALLOWED_ORIGINS in the .env file.", origin)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------------------- auth
class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str


def _extract_token(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ").strip()
    return None


def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Please sign in.")
    user = auth.get_user_from_token(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrators only.")
    return user


@app.post("/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, request: Request, db: Session = Depends(get_db)):
    if not config.ALLOW_SIGNUP:
        raise HTTPException(status_code=403, detail="Sign-ups are closed. Ask an administrator for an account.")
    limiter.enforce("signup", client_ip(request), limit=10, window=3600)

    username = payload.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Username must be 3-64 characters: letters, numbers and . _ @ -")
    problem = auth.password_problem(username, payload.password)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    try:
        user = auth.create_user(db, username, payload.password)
    except ValueError:
        raise HTTPException(status_code=409, detail="That username is already taken")
    return AuthResponse(token=auth.start_session(db, user), username=user.username, role=user.role)


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    username = payload.username.strip()
    ip = client_ip(request)
    pair_key, ip_key = f"{ip}|{username.lower()}", ip

    # lock out repeated wrong passwords (per username+IP, and per IP overall)
    wait = max(limiter.retry_after("login", pair_key, limit=5, window=900),
               limiter.retry_after("login-ip", ip_key, limit=30, window=900))
    if wait:
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {wait} seconds.",
                            headers={"Retry-After": str(wait)})

    token = auth.verify_login(db, username, payload.password)
    if not token:
        limiter.add("login", pair_key)
        limiter.add("login-ip", ip_key)
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    limiter.reset("login", pair_key)
    user = auth.get_user_from_token(db, token)
    return AuthResponse(token=token, username=user.username, role=user.role)


@app.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "role": user.role}


@app.post("/auth/logout")
def logout_route(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    token = _extract_token(authorization)
    if token:
        auth.logout(db, token)
    return {"status": "ok"}


# ------------------------------------------------------------- verify
@app.get("/")
def health_check():
    return {"status": "ok", "service": "ID Verification Backend"}


@app.post("/verify", response_model=VerificationResponse)
async def verify(
    id_image: UploadFile = File(..., description="Photo of the government ID"),
    selfie_image: UploadFile = File(..., description="Live selfie of the user"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    limiter.enforce("verify", user.username, limit=20, window=60)
    performed_by = user.username

    # read at most one byte over the limit, so a huge upload can't fill the memory
    cap = MAX_FILE_SIZE_MB * 1024 * 1024 + 1
    id_bytes = await id_image.read(cap)
    selfie_bytes = await selfie_image.read(cap)
    validate_image(id_image, id_bytes)
    validate_image(selfie_image, selfie_bytes)

    logger.info("Verification request from %s: id=%s, selfie=%s", performed_by, id_image.filename, selfie_image.filename)

    try:
        doc_result_raw, bio_result_raw = await asyncio.gather(
            asyncio.to_thread(run_document_check, id_bytes),
            asyncio.to_thread(run_biometric_check, id_bytes, selfie_bytes),
        )
    except Exception as e:
        logger.exception("ML model call failed")
        raise HTTPException(status_code=500,
                            detail="Verification processing failed." if config.IS_PROD
                            else f"Verification processing failed: {e}")

    try:
        doc_result = DocumentCheckResult(**doc_result_raw)
        bio_result = BiometricCheckResult(**bio_result_raw)
    except Exception as e:
        logger.exception("ML model output did not match expected schema")
        raise HTTPException(status_code=502,
                            detail="The checks returned an unexpected result." if config.IS_PROD
                            else f"Malformed model output: {e}")

    threshold = bio_result.threshold_used or FACE_MATCH_THRESHOLD
    overall_pass = (
        not doc_result.tampered
        and bio_result.match_score >= threshold
        and bio_result.liveness_passed
    )
    overall_reason = build_reason(doc_result, bio_result, overall_pass, threshold)

    response = VerificationResponse(
        status="pass" if overall_pass else "fail",
        document_check=doc_result,
        biometric_check=bio_result,
        overall_reason=overall_reason,
    )
    logger.info("Verification result for %s: %s", performed_by, response.status)

    db.add(Verification(
        id=str(uuid.uuid4()),
        performed_by=performed_by,
        id_image_filename=(id_image.filename or "")[:255],
        selfie_image_filename=(selfie_image.filename or "")[:255],
        status=response.status,
        document_check=doc_result.model_dump(),
        biometric_check=bio_result.model_dump(),
        overall_reason=overall_reason,
    ))
    db.commit()
    return response


# ---------------------------------------------------------- dashboard
def _visible(query, user: User):
    """Admins see every record; operators only their own."""
    return query if user.role == "admin" else query.where(Verification.performed_by == user.username)


@app.get("/history")
def get_history(limit: int = 50, status: str | None = None,
                user: User = Depends(current_user), db: Session = Depends(get_db)):
    limit = max(1, min(limit, 500))
    query = _visible(select(Verification), user)
    count_query = _visible(select(func.count(Verification.id)), user)
    if status:
        query = query.where(Verification.status == status)
        count_query = count_query.where(Verification.status == status)
    rows = db.scalars(query.order_by(Verification.timestamp.desc()).limit(limit)).all()
    return {"total": db.scalar(count_query), "results": [r.to_dict() for r in rows]}


@app.get("/history/{record_id}")
def get_history_record(record_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    record = db.get(Verification, record_id)
    if not record or (user.role != "admin" and record.performed_by != user.username):
        raise HTTPException(status_code=404, detail="Record not found")
    return record.to_dict()


@app.delete("/history/{record_id}")
def delete_history_record(record_id: str, user: User = Depends(admin_user), db: Session = Depends(get_db)):
    record = db.get(Verification, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"status": "deleted"}


@app.get("/stats")
def get_stats(user: User = Depends(current_user), db: Session = Depends(get_db)):
    total = db.scalar(_visible(select(func.count(Verification.id)), user)) or 0
    passed = db.scalar(_visible(select(func.count(Verification.id)), user).where(Verification.status == "pass")) or 0
    return {
        "total_verifications": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 2) if total else None,
    }


def build_reason(doc_result: DocumentCheckResult, bio_result: BiometricCheckResult,
                 overall_pass: bool, threshold: float = FACE_MATCH_THRESHOLD) -> str:
    if overall_pass:
        return "All checks passed: document is authentic and biometric match confirmed."
    reasons = []
    if doc_result.tampered:
        reasons.append(f"Document appears tampered (field: {doc_result.flagged_field}).")
    if bio_result.match_score < threshold:
        reasons.append(f"Face match score too low ({bio_result.match_score} < {threshold}).")
    if not bio_result.liveness_passed:
        reasons.append("Liveness check failed.")
    return " ".join(reasons) if reasons else "Verification failed for an unspecified reason."