"""
DATABASE SETUP (PostgreSQL via SQLAlchemy)

Connection string comes from the DATABASE_URL environment variable
(or a .env file). Example:
    DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/idverify
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/idverify",
)

# Some hosts (Heroku/Render) give "postgres://..." - SQLAlchemy needs the driver name.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: gives each request its own DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates tables if they don't exist yet (runs on app startup)."""
    import db_models  # noqa: F401  (registers the tables on Base)

    Base.metadata.create_all(bind=engine)

    # create_all() never changes existing tables, so add what this version needs (safe to repeat)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(16) NOT NULL DEFAULT 'operator'"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_verifications_performed_by ON verifications (performed_by)"))
