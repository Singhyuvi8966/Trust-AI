"""
Central settings, all read from environment variables (or the .env file).
Nothing here is secret by itself - secrets (DB password) live in .env, never in code.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# "production" turns on stricter behaviour (docs hidden, generic errors, refuses mock results)
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PROD = APP_ENV == "production"

# Websites allowed to call the API from a browser. Not needed when the frontend and the API
# share one domain (the Docker setup does that). Comma separated.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",") if o.strip()
]

# Can anyone create an account from the signup page?
ALLOW_SIGNUP = _bool("ALLOW_SIGNUP", True)

# In production the server refuses to start if face/document checks would be simulated,
# unless you explicitly say you are running a demo.
ALLOW_MOCK = _bool("ALLOW_MOCK", not IS_PROD)

# Delete verification records older than this many days (0 = keep forever).
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "0") or 0)

MIN_PASSWORD_LENGTH = 8
