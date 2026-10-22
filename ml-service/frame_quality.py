# ============================================================
# FRAME QUALITY VALIDATOR
#
# Validates a raw camera frame before passing it to the
# facial emotion model.  Uses the existing MTCNN detector
# (already instantiated in server.py via FaceVerifier) so
# no second detector is created.
#
# Checks (in order, fast-to-slow):
#   1. Brightness   — mean grayscale pixel value ≥ MIN_BRIGHTNESS
#   2. Blur         — Laplacian variance ≥ MIN_BLUR_VARIANCE
#   3. Face detect  — exactly ≥ 1 face found by MTCNN
#   4. Face size    — detected face bounding-box area ≥ MIN_FACE_PX²
#   5. Face clip    — bounding box is not cropped at image edges
#
# Threshold constants are defined at the top of this file.
# Change them here; they take effect on the next server restart.
#
# Return value:
#   (is_valid: bool, reason: str)
#   reason is "ok" when valid, otherwise a short snake_case label.
# ============================================================

import cv2
import numpy as np
from typing import Tuple, Optional


# ============================================================
# CONFIGURABLE THRESHOLDS
# ============================================================

# Minimum mean grayscale brightness (0–255).
# Set permissively low: only rejects pitch-black frames where no
# face information is present at all.  Candidates in dim rooms
# (mean brightness ≈ 30–80) will comfortably pass this check.
# ──────────────────────────────────────────────────────────────
# NOTE: No brightness threshold existed in this project before
# this file was added.  This is the one and only brightness
# threshold.  Adjust here if requirements change.
MIN_BRIGHTNESS: int = 20

# Minimum Laplacian variance — frames below this are considered
# too blurry for reliable emotion inference.
# A value of 30 rejects badly motion-blurred frames while
# still allowing slightly soft webcam images.
MIN_BLUR_VARIANCE: float = 30.0

# Minimum face bounding-box side length in pixels.
# A face crop smaller than this is too coarse for the ViT model.
# 48 px × 48 px is a reasonable lower bound for 224 px input.
MIN_FACE_PX: int = 48

# Fraction of the image edge margin allowed before a face is
# considered clipped.  0.02 → within 2 % of any edge is flagged.
EDGE_CLIP_MARGIN: float = 0.02


# ============================================================
# VALIDATOR
# ============================================================

class FrameQualityValidator:
    """
    Lightweight per-frame quality gate.

    Pass the MTCNN detector that is already instantiated in
    server.py (from FaceVerifier) so no duplicate model is loaded.

    Usage
    -----
    validator = FrameQualityValidator(mtcnn_detector=face_verifier.detector)
    is_valid, reason = validator.check(image_bgr)
    """

    def __init__(self, mtcnn_detector):
        """
        Parameters
        ----------
        mtcnn_detector : facenet_pytorch.MTCNN
            The MTCNN instance already loaded inside FaceVerifier.
            Passing it here avoids loading a second copy.
        """
        self._detector = mtcnn_detector

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def check(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Validate a raw BGR camera frame.

        Parameters
        ----------
        image_bgr : np.ndarray
            Full-frame BGR image as decoded by OpenCV.

        Returns
        -------
        (True, "ok")              — frame is acceptable for inference
        (False, "<reason>")       — frame rejected; reason is one of:
            "too_dark"            brightness < MIN_BRIGHTNESS
            "too_blurry"          Laplacian variance < MIN_BLUR_VARIANCE
            "no_face"             MTCNN found no face
            "face_too_small"      detected face < MIN_FACE_PX on any side
            "face_clipped"        face bounding box touches image edge
        """

        if image_bgr is None or image_bgr.size == 0:
            return False, "empty_frame"

        # ----------------------------------------------------------
        # 1. Brightness check (fast — pure NumPy, no model call)
        # Reuses the threshold defined as MIN_BRIGHTNESS above.
        # This is the only brightness check in the entire project.
        # ----------------------------------------------------------
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < MIN_BRIGHTNESS:
            return False, "too_dark"

        # ----------------------------------------------------------
        # 2. Blur check (fast — Laplacian, no model call)
        # ----------------------------------------------------------
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_var < MIN_BLUR_VARIANCE:
            return False, "too_blurry"

        # ----------------------------------------------------------
        # 3. Face detection via existing MTCNN detector
        # ----------------------------------------------------------
        try:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            )
            boxes, probs = self._detector.detect(img_pil)
        except Exception as detect_err:
            # Detection failure is treated as no-face rather than
            # crashing the pipeline.
            print(
                f"[FrameQuality] MTCNN detection error: {detect_err}"
            )
            return False, "detection_error"

        if boxes is None or len(boxes) == 0:
            return False, "no_face"

        # Keep only high-confidence detections
        valid_boxes = [
            box
            for box, prob in zip(boxes, probs)
            if prob is not None and prob >= 0.85
        ]

        if not valid_boxes:
            return False, "no_face"

        # Use the largest face (closest to camera)
        best_box = max(
            valid_boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )

        x1, y1, x2, y2 = best_box
        h, w = image_bgr.shape[:2]

        # ----------------------------------------------------------
        # 4. Face size check
        # ----------------------------------------------------------
        face_w = x2 - x1
        face_h = y2 - y1

        if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
            return False, "face_too_small"

        # ----------------------------------------------------------
        # 5. Edge-clip check
        # ----------------------------------------------------------
        margin_px_x = w * EDGE_CLIP_MARGIN
        margin_px_y = h * EDGE_CLIP_MARGIN

        if (
            x1 < margin_px_x
            or y1 < margin_px_y
            or x2 > w - margin_px_x
            or y2 > h - margin_px_y
        ):
            return False, "face_clipped"

        return True, "ok"
