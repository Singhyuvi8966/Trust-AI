"""
DATABASE TABLES (SQLAlchemy ORM).
Named db_models.py so it doesn't clash with the ML `models/` folder.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

# JSONB on PostgreSQL (indexable/queryable), plain JSON elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default="operator", server_default="operator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserSession(Base):
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship()


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    performed_by: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    id_image_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selfie_image_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(8), index=True)  # "pass" / "fail"
    document_check: Mapped[dict] = mapped_column(JSONType)
    biometric_check: Mapped[dict] = mapped_column(JSONType)
    overall_reason: Mapped[str] = mapped_column(Text)

    def to_dict(self) -> dict:
        """Same shape the old in-memory records had, so the frontend is unaffected."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "performed_by": self.performed_by,
            "id_image_filename": self.id_image_filename,
            "selfie_image_filename": self.selfie_image_filename,
            "status": self.status,
            "document_check": self.document_check,
            "biometric_check": self.biometric_check,
            "overall_reason": self.overall_reason,
        }
