"""
Small helper utilities: basic upload validation so we fail fast with a
clean error instead of crashing inside the ML modules.
"""

from fastapi import UploadFile, HTTPException

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE_MB = 8


def validate_image(file: UploadFile, file_bytes: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}' for '{file.filename}'. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is {size_mb:.1f}MB, exceeds {MAX_FILE_SIZE_MB}MB limit.",
        )

    if size_mb == 0:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is empty.",
        )
