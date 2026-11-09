"""
BIOMETRIC CHECK MODULE - Face Match / Liveness

REAL MODE (when `deepface` is installed - see requirements-ml.txt):
    1. Face embeddings   - ArcFace model turns each face into a 512-number vector;
                           the cosine distance between the two vectors = how
                           different the faces are.
    2. Facial landmarks  - RetinaFace finds 5 key points (eyes, nose, mouth
                           corners); we compare the face *geometry* (ratios
                           between those points) of the ID photo and the selfie.
    3. Age-related changes - an age estimator runs on both photos. If the ID
                           photo is much older than the selfie, the match
                           threshold is relaxed slightly (capped) because
                           faces really do change over the years.
    4. Liveness          - DeepFace anti-spoofing on the selfie (if available).

MOCK MODE (when deepface is NOT installed): returns random fake values so
the rest of the team can still build against the API.

CONTRACT (matches schemas.BiometricCheckResult):
    Input:  id_image_bytes, selfie_image_bytes (raw bytes)
    Output: dict with match_score, liveness_passed, liveness_confidence, notes
            + optional extras: embedding_model, embedding_distance,
              landmark_score, id_age_estimate, selfie_age_estimate,
              age_gap_years, threshold_used, analysis_mode

Face embeddings are used only in memory to compute the score. They are
NOT returned or stored (they are sensitive biometric data).
"""

import logging
import os
import random
import time

logger = logging.getLogger("biometric-check")

# ------------------------------------------------------------------ config
BASE_MATCH_THRESHOLD = 0.75      # pass mark for match_score (before age relaxation)

EMBEDDING_MODEL = "ArcFace"
DETECTOR = "retinaface"
ARCFACE_COSINE_THRESHOLD = 0.68  # DeepFace's own "same person" cut-off for ArcFace+cosine

# Age tolerance: age estimators are only accurate to roughly +/-5 years, so
# small gaps are ignored; bigger gaps relax the threshold a little, with a cap.
AGE_GAP_FREE_YEARS = 5
RELAX_PER_YEAR = 0.005
MAX_RELAX = 0.05                 # threshold never drops below 0.70

# If the anti-spoofing model can't run, fail SAFE (needs manual review) instead of letting everyone pass.
# Set LIVENESS_FAIL_OPEN=true only if you accept that risk.
LIVENESS_FAIL_OPEN = os.getenv("LIVENESS_FAIL_OPEN", "false").strip().lower() in ("1", "true", "yes", "on")

# ------------------------------------------------------------ optional deps
try:
    import cv2
    import numpy as np
    from deepface import DeepFace
    from retinaface import RetinaFace

    REAL_MODE = True
except Exception as _e:  # ImportError, or TensorFlow failing to load
    REAL_MODE = False
    logger.warning(
        "DeepFace not available (%s) - biometric check running in MOCK mode. "
        "Install requirements-ml.txt for real face analysis.", _e
    )


# ------------------------------------------------------------ real helpers
def _decode(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded")
    return img


def _area(region: dict) -> int:
    return region["w"] * region["h"]


def _embedding(img):
    """512-d ArcFace embedding of the largest face. Raises ValueError if no face."""
    results = DeepFace.represent(
        img_path=img,
        model_name=EMBEDDING_MODEL,
        detector_backend=DETECTOR,
        enforce_detection=True,
        align=True,
    )
    best = max(results, key=lambda r: _area(r["facial_area"]))
    return np.asarray(best["embedding"], dtype=np.float32)


def _cosine_distance(a, b) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _distance_to_score(distance: float) -> float:
    """Maps cosine distance to a 0-1 score so that the model's own 'same person'
    cut-off (0.68) lands exactly on 0.75 (BASE_MATCH_THRESHOLD)."""
    score = 1.0 - 0.25 * distance / ARCFACE_COSINE_THRESHOLD
    return max(0.0, min(1.0, score))


def _landmarks(img):
    """5 facial landmarks of the largest face, or None if not found."""
    faces = RetinaFace.detect_faces(img)
    if not isinstance(faces, dict) or not faces:
        return None

    def box_area(f):
        x1, y1, x2, y2 = f["facial_area"]
        return (x2 - x1) * (y2 - y1)

    lm = max(faces.values(), key=box_area)["landmarks"]
    return {k: np.asarray(v, dtype=np.float32) for k, v in lm.items()}


def _geometry(lm: dict):
    """Scale-invariant face proportions (all distances divided by eye distance)."""
    eye_dist = float(np.linalg.norm(lm["left_eye"] - lm["right_eye"]))
    if eye_dist < 1e-6:
        return None
    eye_mid = (lm["left_eye"] + lm["right_eye"]) / 2
    mouth_mid = (lm["mouth_left"] + lm["mouth_right"]) / 2
    return np.array(
        [
            np.linalg.norm(lm["nose"] - eye_mid) / eye_dist,                 # eyes -> nose
            np.linalg.norm(mouth_mid - lm["nose"]) / eye_dist,               # nose -> mouth
            np.linalg.norm(lm["mouth_left"] - lm["mouth_right"]) / eye_dist, # mouth width
            np.linalg.norm(mouth_mid - eye_mid) / eye_dist,                  # eyes -> mouth
        ]
    )


def _landmark_score(g1, g2) -> float:
    """1.0 = identical proportions, lower = more different."""
    rel_diff = np.abs(g1 - g2) / ((g1 + g2) / 2)
    return float(max(0.0, 1.0 - 2.0 * rel_diff.mean()))


def _age(img):
    res = DeepFace.analyze(
        img_path=img,
        actions=["age"],
        detector_backend=DETECTOR,
        enforce_detection=True,
        silent=True,
    )
    if isinstance(res, list):
        res = res[0]
    return int(res["age"])


def _liveness(selfie_img):
    """Returns (passed, confidence), or None if anti-spoofing is unavailable."""
    try:
        faces = DeepFace.extract_faces(
            img_path=selfie_img,
            detector_backend=DETECTOR,
            enforce_detection=True,
            anti_spoofing=True,
        )
    except ValueError as e:
        if "spoof" in str(e).lower():
            return False, 0.0
        raise  # e.g. no face found
    except Exception as e:  # older DeepFace without anti_spoofing, model load issue...
        logger.warning("Anti-spoofing unavailable: %s", e)
        return None
    best = max(faces, key=lambda f: _area(f["facial_area"]))
    return True, round(float(best.get("antispoof_score", 1.0)), 2)


def _age_adjusted_threshold(id_age, selfie_age):
    """Returns (threshold, age_gap). Larger estimated age gap -> slightly lower bar."""
    if id_age is None or selfie_age is None:
        return BASE_MATCH_THRESHOLD, None
    gap = abs(id_age - selfie_age)
    relax = min(MAX_RELAX, max(0, gap - AGE_GAP_FREE_YEARS) * RELAX_PER_YEAR)
    return round(BASE_MATCH_THRESHOLD - relax, 3), gap


def _failure(message: str) -> dict:
    return {
        "match_score": 0.0,
        "liveness_passed": False,
        "liveness_confidence": 0.0,
        "notes": message,
        "threshold_used": BASE_MATCH_THRESHOLD,
        "analysis_mode": "real",
    }


def _real_biometric_check(id_image_bytes: bytes, selfie_image_bytes: bytes) -> dict:
    try:
        id_img = _decode(id_image_bytes)
        selfie_img = _decode(selfie_image_bytes)
        id_emb = _embedding(id_img)
        selfie_emb = _embedding(selfie_img)
    except ValueError as e:
        # no face detected / undecodable image
        return _failure(f"Could not find a usable face: {e}")

    # 1) Face embeddings
    distance = _cosine_distance(id_emb, selfie_emb)
    match_score = round(_distance_to_score(distance), 2)

    # 2) Facial landmarks (supporting signal - reported, not part of pass/fail)
    landmark_score = None
    try:
        lm_id, lm_selfie = _landmarks(id_img), _landmarks(selfie_img)
        if lm_id and lm_selfie:
            g1, g2 = _geometry(lm_id), _geometry(lm_selfie)
            if g1 is not None and g2 is not None:
                landmark_score = round(_landmark_score(g1, g2), 2)
    except Exception as e:
        logger.warning("Landmark analysis failed: %s", e)

    # 3) Age-related facial changes
    id_age = selfie_age = None
    try:
        id_age, selfie_age = _age(id_img), _age(selfie_img)
    except Exception as e:
        logger.warning("Age estimation failed: %s", e)
    threshold_used, age_gap = _age_adjusted_threshold(id_age, selfie_age)

    # 4) Liveness
    notes = []
    try:
        liveness = _liveness(selfie_img)
    except ValueError as e:
        return _failure(f"Could not find a usable face in selfie: {e}")
    if liveness is None:
        if LIVENESS_FAIL_OPEN:
            liveness_passed, liveness_conf = True, 0.0
            notes.append("Liveness was NOT evaluated (anti-spoofing unavailable).")
        else:
            liveness_passed, liveness_conf = False, 0.0
            notes.append("Liveness could not be evaluated, so this needs a manual review.")
    else:
        liveness_passed, liveness_conf = liveness
        if not liveness_passed:
            notes.append("Liveness check failed - possible photo/screen spoof.")

    if match_score < threshold_used:
        notes.append("Low similarity between selfie and ID photo.")
    else:
        notes.append("Selfie and ID photo faces match.")
    if age_gap is not None and threshold_used < BASE_MATCH_THRESHOLD:
        notes.append(
            f"Estimated age gap {age_gap}y - match threshold relaxed to {threshold_used}."
        )
    if landmark_score is not None and landmark_score < 0.7:
        notes.append(f"Facial proportions differ noticeably (landmark score {landmark_score}).")

    return {
        "match_score": match_score,
        "liveness_passed": liveness_passed,
        "liveness_confidence": liveness_conf,
        "notes": " ".join(notes),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_distance": round(distance, 3),
        "landmark_score": landmark_score,
        "id_age_estimate": id_age,
        "selfie_age_estimate": selfie_age,
        "age_gap_years": age_gap,
        "threshold_used": threshold_used,
        "analysis_mode": "real",
    }


# --------------------------------------------------------------- mock mode
def _mock_biometric_check(id_image_bytes: bytes, selfie_image_bytes: bytes) -> dict:
    """MOCK: random fake values (same as before, plus the new fields)."""
    time.sleep(0.7)

    match_score = round(random.uniform(0.5, 0.99), 2)
    liveness_passed = random.random() > 0.15
    liveness_confidence = round(random.uniform(0.7, 0.99), 2)
    id_age = random.randint(18, 60)
    selfie_age = max(18, id_age + random.randint(-6, 10))
    threshold_used, age_gap = _age_adjusted_threshold(id_age, selfie_age)

    notes = "Face match and liveness check completed successfully."
    if match_score < threshold_used:
        notes = "Low similarity between selfie and ID photo."
    elif not liveness_passed:
        notes = "Liveness check failed - possible photo-of-a-photo spoof attempt."

    return {
        "match_score": match_score,
        "liveness_passed": liveness_passed,
        "liveness_confidence": liveness_confidence,
        "notes": notes,
        "embedding_model": "mock",
        "embedding_distance": round(random.uniform(0.3, 1.1), 3),
        "landmark_score": round(random.uniform(0.6, 0.99), 2),
        "id_age_estimate": id_age,
        "selfie_age_estimate": selfie_age,
        "age_gap_years": age_gap,
        "threshold_used": threshold_used,
        "analysis_mode": "mock",
    }


# ------------------------------------------------------------- public API
def run_biometric_check(id_image_bytes: bytes, selfie_image_bytes: bytes) -> dict:
    if REAL_MODE:
        return _real_biometric_check(id_image_bytes, selfie_image_bytes)
    return _mock_biometric_check(id_image_bytes, selfie_image_bytes)


def warm_up() -> None:
    """Loads (and on first run downloads) the models at startup so the first
    real request isn't slow. Safe to call in mock mode (does nothing)."""
    if not REAL_MODE:
        return
    logger.info("Loading face models (first run downloads weights, may take a while)...")
    try:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(blank, model_name=EMBEDDING_MODEL, detector_backend="skip",
                           enforce_detection=False)
        DeepFace.analyze(blank, actions=["age"], detector_backend="skip",
                         enforce_detection=False, silent=True)
        RetinaFace.detect_faces(blank)
        logger.info("Face models ready.")
    except Exception as e:
        logger.warning("Model warm-up failed (will retry on first request): %s", e)
