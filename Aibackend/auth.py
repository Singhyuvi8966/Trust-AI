"""
AUTH MODULE (PostgreSQL-backed)

- Passwords are hashed with bcrypt.
- Session tokens are random; only a SHA-256 fingerprint of each token is stored, so a leaked
  database does not hand out working logins.
- Sessions expire after SESSION_TTL. Roles: "operator" (sees own records) and "admin" (sees all).
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import config
from db_models import User, UserSession

SESSION_TTL = timedelta(days=7)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt()).decode()   # bcrypt reads 72 bytes


def _check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode()[:72], password_hash.encode())


_DUMMY_HASH = _hash_password("not-a-real-password")    # so unknown usernames take as long as real ones


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(UserSession(token=_digest(token), user_id=user.id, expires_at=datetime.now(timezone.utc) + SESSION_TTL))
    db.commit()
    return token


def user_exists(db: Session, username: str) -> bool:
    return db.scalar(select(User.id).where(User.username == username)) is not None


def create_user(db: Session, username: str, password: str, role: str = "operator") -> User:
    """Creates an account. Raises ValueError if the username is taken."""
    user = User(username=username, password_hash=_hash_password(password), role=role)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError("username already taken")
    db.refresh(user)
    return user


def verify_login(db: Session, username: str, password: str) -> str | None:
    """Returns a fresh session token if the credentials are right, else None."""
    user = db.scalar(select(User).where(User.username == username))
    if not user:
        _check_password(password, _DUMMY_HASH)           # spend the same time either way
        return None
    if not _check_password(password, user.password_hash):
        return None
    return start_session(db, user)


def get_user_from_token(db: Session, token: str) -> User | None:
    session = db.get(UserSession, _digest(token))
    if not session:
        return None
    if session.expires_at < datetime.now(timezone.utc):
        db.delete(session)
        db.commit()
        return None
    return session.user


def logout(db: Session, token: str) -> None:
    session = db.get(UserSession, _digest(token))
    if session:
        db.delete(session)
        db.commit()


def purge_expired_sessions(db: Session) -> int:
    result = db.execute(delete(UserSession).where(UserSession.expires_at < datetime.now(timezone.utc)))
    db.commit()
    return result.rowcount or 0


def password_problem(username: str, password: str) -> str | None:
    """A message if the password is not acceptable for a NEW account, else None."""
    if len(password) < config.MIN_PASSWORD_LENGTH:
        return f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters."
    if len(password.encode()) > 72:
        return "Password is too long (72 bytes at most)."
    if password.lower() == username.lower():
        return "Password can't be the same as the username."
    return None
