"""
Pydantic models that define the exact JSON contract between:
- ML/AI team (document + biometric modules)
- Backend (this service)
- Frontend / Admin Dashboard team

Everyone should treat this file as the single source of truth
for field names and value ranges.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DocumentCheckResult(BaseModel):
    tampered: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in the tampered verdict (0-1)")
    flagged_field: Optional[str] = Field(
        None, description="Which field looked suspicious, e.g. 'date_of_birth', 'expiry_date'"
    )
    notes: Optional[str] = Field(None, description="Human-readable explanation for the dashboard")
    analysis_mode: Optional[str] = Field(None, description="'model' (trained classifier) or 'mock'")


class BiometricCheckResult(BaseModel):
    match_score: float = Field(..., ge=0.0, le=1.0, description="Selfie-to-ID face match score (0-1)")
    liveness_passed: bool
    liveness_confidence: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = Field(None, description="Human-readable explanation for the dashboard")

    # --- Extra face-analysis details (optional, for the dashboard) ---
    embedding_model: Optional[str] = Field(None, description="Face embedding model used, e.g. 'ArcFace'")
    embedding_distance: Optional[float] = Field(None, description="Cosine distance between face embeddings (lower = more similar)")
    landmark_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Facial landmark geometry similarity (0-1)")
    id_age_estimate: Optional[int] = Field(None, description="Estimated age of the face on the ID photo")
    selfie_age_estimate: Optional[int] = Field(None, description="Estimated age of the face in the selfie")
    age_gap_years: Optional[int] = Field(None, description="Absolute estimated age difference between the two photos")
    threshold_used: Optional[float] = Field(None, description="Match threshold actually applied (may be age-adjusted)")
    analysis_mode: Optional[str] = Field(None, description="'real' or 'mock'")


class VerificationResponse(BaseModel):
    status: str  # "pass" or "fail"
    document_check: DocumentCheckResult
    biometric_check: BiometricCheckResult
    overall_reason: str
