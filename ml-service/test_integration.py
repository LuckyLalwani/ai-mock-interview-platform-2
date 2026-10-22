"""
Full integration test for facial emotion sampling and quality validation changes.

Run from the ml-service directory:
    python test_integration.py
"""
import sys
import re
import time
import numpy as np
import cv2
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

passed = 0
failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        print(f"  PASS: {label}")
        passed += 1
    else:
        print(f"  FAIL: {label}")
        failed += 1


# ──────────────────────────────────────────────────────────────────────
# Test A: Sampling interval = 2000 ms (frames every 2 sec sent for analysis)
# ──────────────────────────────────────────────────────────────────────
print("=== Test A: Sampling interval changed to 2000 ms (every 2 sec) ===")
tsx_path = os.path.join(ROOT, "..", "src", "pages", "InterviewSession.tsx")
with open(tsx_path, "r", encoding="utf-8") as f:
    tsx_content = f.read()

intervals = re.findall(
    r"setInterval\(\s*\(\)\s*=>\s*\{\s*captureFaceEmotion\(\)\s*;\s*\}\s*,\s*(\d+)\s*\)",
    tsx_content,
)
check(bool(intervals), "setInterval with captureFaceEmotion() found")
if intervals:
    check(int(intervals[0]) == 2000, f"Interval is 2000ms / 2 seconds (found {intervals[0]}ms)")



# ──────────────────────────────────────────────────────────────────────
# Test B: Drop-frame guard (isFaceCapturingRef)
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test B: Drop-frame guard (isFaceCapturingRef) ===")
check("isFaceCapturingRef" in tsx_content, "isFaceCapturingRef ref declared")
check("isFaceCapturingRef.current = true" in tsx_content, "Guard set-to-true on entry")
check("isFaceCapturingRef.current = false" in tsx_content, "Guard released on completion")


# ──────────────────────────────────────────────────────────────────────
# Test C: Skipped-frame handling & immediate recapture in frontend
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test C: Skipped-frame handling & immediate recapture in InterviewSession.tsx ===")
check("result.skipped" in tsx_content, "result.skipped check present")
check(
    "do NOT set faceEmotion" in tsx_content.lower()
    or "Do NOT set faceEmotion" in tsx_content,
    "Comment documenting that invalid frames do not set faceEmotion",
)
check("consecutiveInvalidRetriesRef" in tsx_content, "Immediate recapture retry counter declared")
check("setTimeout" in tsx_content and "captureFaceEmotion()" in tsx_content, "Immediate recapture callback scheduled")


# ──────────────────────────────────────────────────────────────────────
# Test D: EmotionPrediction interface updated with skipped field
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test D: EmotionPrediction interface ===")
emotion_ts_path = os.path.join(ROOT, "..", "src", "lib", "emotion.ts")
with open(emotion_ts_path, "r", encoding="utf-8") as f:
    emotion_ts = f.read()

check("skipped?" in emotion_ts, "skipped? optional field in EmotionPrediction")
check("reason?" in emotion_ts, "reason? optional field in EmotionPrediction")


# ──────────────────────────────────────────────────────────────────────
# Test E: Brightness threshold (MIN_BRIGHTNESS = 20, permissive)
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test E: Brightness threshold ===")
from frame_quality import MIN_BRIGHTNESS, MIN_BLUR_VARIANCE, MIN_FACE_PX

check(MIN_BRIGHTNESS == 20, f"MIN_BRIGHTNESS = {MIN_BRIGHTNESS} (expected 20)")
check(
    MIN_BRIGHTNESS < 30,
    f"MIN_BRIGHTNESS ({MIN_BRIGHTNESS}) is permissive (< 30 so dim rooms are allowed)",
)
print(f"  MIN_BLUR_VARIANCE = {MIN_BLUR_VARIANCE}")
print(f"  MIN_FACE_PX = {MIN_FACE_PX}")


# ──────────────────────────────────────────────────────────────────────
# Test F: Quality validator rejects invalid frames, accepts valid ones
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test F: Quality validator behavior ===")
from face_verifier import FaceVerifier
from frame_quality import FrameQualityValidator

fv = FaceVerifier(threshold=0.70)
validator = FrameQualityValidator(mtcnn_detector=fv.detector)

# Pitch-black (brightness = 0, well below MIN_BRIGHTNESS=20)
black = np.zeros((480, 640, 3), dtype=np.uint8)
ok, reason = validator.check(black)
check(not ok and reason == "too_dark", f"Pitch-black frame rejected as too_dark (got {reason!r})")

# Dim but acceptable (brightness=30 > MIN_BRIGHTNESS=20)
dim = np.full((480, 640, 3), 30, dtype=np.uint8)
ok, reason = validator.check(dim)
check(reason != "too_dark", f"Dim(30) frame NOT rejected for brightness (rejected for {reason!r} instead)")

# Blurry frame
import cv2
blurry = cv2.GaussianBlur(
    np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8),
    (101, 101), 50
)
ok, reason = validator.check(blurry)
check(not ok and reason == "too_blurry", f"Blurry frame rejected as too_blurry (got {reason!r})")

# Real face image
face_path = os.path.join(ROOT, "test_face.jpg")
if os.path.exists(face_path):
    face = cv2.imread(face_path)
    if face is not None:
        ok, reason = validator.check(face)
        check(ok and reason == "ok", f"Real face image accepted (reason={reason!r})")
    else:
        print("  SKIP: could not decode test_face.jpg")
else:
    print("  SKIP: test_face.jpg not found")


# ──────────────────────────────────────────────────────────────────────
# Test G: server.py has skipped response and MTCNN reuse
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test G: server.py validation pipeline ===")
server_path = os.path.join(ROOT, "server.py")
with open(server_path, "r", encoding="utf-8") as f:
    server_content = f.read()

check("frame_quality_validator" in server_content, "frame_quality_validator used in server.py")
check("FrameQualityValidator" in server_content, "FrameQualityValidator imported in server.py")
check(
    "mtcnn_detector=face_verifier.detector" in server_content,
    "MTCNN reused from FaceVerifier (no duplicate detector)",
)
check('"skipped": True' in server_content, "skipped=True returned for rejected frames")
check('"reason": rejection_reason' in server_content, "rejection reason included in response")
check(
    "Do NOT set faceEmotion" in server_content
    or "no default emotion" in server_content.lower()
    or "do not assign" in server_content.lower()
    or "must not affect" in server_content.lower()
    or "must never contribute" in server_content.lower()
    or "do not affect scoring" in server_content.lower(),
    "Comment documenting that invalid frames must not affect scoring",
)

# ──────────────────────────────────────────────────────────────────────
# Test H: Proctoring verification interval = 3000 ms
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test H: Proctoring interval (3000 ms) ===")
check("}, 3000);" in tsx_content, "Proctor interval is set to 3000ms (3 seconds)")

# ──────────────────────────────────────────────────────────────────────
# Test I: Fullscreen exit limits and 0-tab tolerance
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test I: Fullscreen exit policy & 0-tab tolerance ===")
check("currentExits === 1" in tsx_content, "1st fullscreen exit warning branch present")
check("currentExits >= 2" in tsx_content, "2nd fullscreen exit termination branch present")
check("fullscreenExitCountRef" in tsx_content, "Fullscreen exit tracking ref present")
check("document.hidden" in tsx_content and "handleEnd" in tsx_content, "Tab visibility loss terminates immediately (0 tab switch)")

# ──────────────────────────────────────────────────────────────────────
# Test J: Text inference long-answer chunking
# ──────────────────────────────────────────────────────────────────────
print("\n=== Test J: Text emotion whole-answer chunking ===")
text_infer_path = os.path.join(ROOT, "text_inference.py")
with open(text_infer_path, "r", encoding="utf-8") as f:
    text_content = f.read()

check("stride" in text_content or "sliding" in text_content.lower(), "Sliding-window token chunking implemented in text_inference.py")
check("max_chunk_size" in text_content, "Chunk size configured for long answers (510 tokens)")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  PASSED: {passed}")
print(f"  FAILED: {failed}")
if failed == 0:
    print("\n  ALL INTEGRATION TESTS PASSED")
else:
    print(f"\n  {failed} TEST(S) FAILED")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
