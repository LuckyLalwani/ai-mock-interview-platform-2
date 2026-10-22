import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from pathlib import Path
import os
import io

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import threading
import time
import requests
import imageio_ffmpeg
from pydantic import BaseModel
from fusion_engine_new import FusionEngine
import cv2
import numpy as np
import librosa
import subprocess

from face_verifier import FaceVerifier
from frame_quality import FrameQualityValidator
from text_inference import text_predictor
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Groq Whisper Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Local Whisper Pipeline (lazy loaded on demand or fallback)
local_whisper_pipe = None

def get_local_whisper():
    global local_whisper_pipe
    if local_whisper_pipe is None:
        import torch
        from transformers import pipeline
        print("[ML Service] Loading local Whisper fallback pipeline (openai/whisper-tiny.en)...")
        local_whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device="cpu",
            torch_dtype=torch.float32
        )
    return local_whisper_pipe


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


app = FastAPI(
    title="Interview Emotion ML Service",
    version="1.0.0",
)


cors_origins_env = os.getenv("CORS_ORIGINS", "")
explicit_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=explicit_origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


facial_model = None
voice_model = None
face_verifier = None
# Frame quality validator — reuses MTCNN from face_verifier (no duplicate model)
frame_quality_validator = None


@app.on_event("startup")
def load_models():
    global facial_model, voice_model, face_verifier, frame_quality_validator

    print("\n[ML Service] Loading ML models...")

    from facial_inference import FacialInference
    from voice_inference import VoiceInference

    facial_classes_path = (
        MODELS_DIR / "facial_emotion_classes.npy"
        if (MODELS_DIR / "facial_emotion_classes.npy").exists()
        else MODELS_DIR / "vit_facial_emotion_classes.npy"
    )

    facial_model = FacialInference(
        str(MODELS_DIR / "facial_emotion_model.pt"),
        str(facial_classes_path),
    )

    voice_model = VoiceInference(
        str(MODELS_DIR / "voice_emotion_model.pt"),
    )

    face_verifier = FaceVerifier(threshold=0.70)

    # Reuse the MTCNN detector already loaded inside FaceVerifier —
    # no second copy of the model is created.
    frame_quality_validator = FrameQualityValidator(
        mtcnn_detector=face_verifier.detector
    )

    print("[ML Service] Facial model loaded")
    print("[ML Service] Voice model loaded")
    print("[ML Service] Frame quality validator ready (reusing MTCNN)")
    print("[ML Service] ML service ready\n")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Interview Emotion ML Service",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "facial_model_loaded": facial_model is not None,
        "voice_model_loaded": voice_model is not None,
    }


@app.post("/predict/face")
async def predict_face(file: UploadFile = File(...)):
    """
    Facial emotion prediction endpoint.

    Pipeline:
      1. Decode uploaded frame.
      2. Run frame quality validation (face detection, brightness,
         blur, face size, face clipping).
      3. If invalid: return skipped=True with reason — NO emotion
         prediction is made, no default emotion is assigned.
      4. If valid: pass full frame to FacialInference.predict().

    Invalid frames are logged to console only (not to the database)
    to avoid unnecessary storage use.  They do not affect scoring.
    """
    if facial_model is None:
        return {
            "success": False,
            "error": "Facial model is not loaded",
        }

    try:
        image_bytes = await file.read()

        if not image_bytes:
            return {
                "success": False,
                "error": "Empty image received",
            }

        # Convert uploaded bytes → NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        # Decode image using OpenCV
        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            return {
                "success": False,
                "error": "Could not decode image",
            }

        # --------------------------------------------------------
        # Frame quality validation
        # Reuses the MTCNN already loaded in FaceVerifier — no
        # duplicate detector.  Runs before the ViT model so that
        # invalid frames never reach inference.
        # --------------------------------------------------------
        if frame_quality_validator is not None:
            is_valid, rejection_reason = frame_quality_validator.check(
                image_bgr
            )

            if not is_valid:
                # Log rejection to console only — not to the database.
                capture_ts = time.time()
                print(
                    f"[ML Service] Frame rejected: {rejection_reason} "
                    f"(ts={capture_ts:.3f})"
                )

                # Return skipped=True.  The frontend must NOT treat
                # this as a prediction and must NOT assign any default
                # emotion.  Invalid frames must not affect scoring.
                return {
                    "success": False,
                    "skipped": True,
                    "reason": rejection_reason,
                    "timestamp": capture_ts,
                }

        # --------------------------------------------------------
        # Frame passed quality checks — run emotion inference
        # OpenCV uses BGR, but FacialInference expects RGB
        # --------------------------------------------------------
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        predictions = facial_model.predict(image_rgb)

        # Find dominant emotion
        dominant_emotion = max(
            predictions,
            key=predictions.get,
        )

        confidence = predictions[dominant_emotion]

        return {
            "success": True,
            "skipped": False,
            "emotion": dominant_emotion,
            "confidence": confidence,
            "predictions": predictions,
        }

    except Exception as e:
        print(f"[ML Service] Face prediction error: {e}")

        return {
            "success": False,
            "error": str(e),
        }

@app.post("/proctor/register")
async def register_proctor_face(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = BASE_DIR / f"proctor_register_{time.time_ns()}.jpg"

    try:
        temp_path.write_bytes(contents)

        result = face_verifier.register(str(temp_path))

        return result

    except Exception as e:
        print(f"[Proctor] Registration error: {e}")
        return {
            "status": "ERROR",
            "message": str(e),
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.post("/proctor/verify")
async def verify_proctor_face(file: UploadFile = File(...)):
    if face_verifier is None:
        return {
            "status": "ERROR",
            "message": "Face verifier is not loaded",
        }

    contents = await file.read()
    temp_path = BASE_DIR / f"proctor_verify_{time.time_ns()}.jpg"

    try:
        temp_path.write_bytes(contents)

        result = face_verifier.verify(str(temp_path))

        return result

    except Exception as e:
        print(f"[Proctor] Verification error: {e}")
        return {
            "status": "ERROR",
            "message": str(e),
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    input_path = None
    output_path = None

    try:
        input_path = BASE_DIR / f"temp_input_{time.time_ns()}.webm"
        output_path = BASE_DIR / f"temp_output_{time.time_ns()}.wav"

        input_path.write_bytes(audio_bytes)

        subprocess.run(
            [
                FFMPEG_EXE,
                "-y",
                "-i",
                str(input_path),
                "-ar",
                str(voice_model.SR),
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        audio, _ = librosa.load(
            str(output_path),
            sr=voice_model.SR,
            mono=True,
        )

        audio = audio.astype(np.float32)

        result = voice_model.predict(audio)

        if not result:
            raise ValueError("Voice model returned no predictions.")

        emotion = max(result, key=result.get)
        confidence = float(result[emotion])

        return {
            "success": True,
            "emotion": emotion,
            "confidence": confidence,
            "predictions": {
                emotion_name: float(score)
                for emotion_name, score in result.items()
            },
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"FFmpeg conversion failed: {e.stderr}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    finally:
        if input_path and input_path.exists():
            input_path.unlink()

        if output_path and output_path.exists():
            output_path.unlink()

# Common Whisper silence hallucinations to filter out
HALLUCINATION_SUBSTRINGS = [
    "i'm going to go to the next question",
    "i'm going to go to the next one",
    "going to go to the next question",
    "going to go to the next one",
    "to the next question",
    "to the next one",
    "i'm not sure you're going to get the answer",
    "i'm not sure you're going to get",
    "i'm not sure what i'm going to do",
    "i'm not sure what i'm doing",
    "i'm going to get to date",
    "i can't wait to see the answer",
    "and i can't wait to see the answer",
    "i can't see",
    "i'm sorry",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "thanks for listening",
    "subtitles by",
    "please subscribe",
    "see you next time",
    "13, 5.",
    "13, 5",
    "13 5",
    "candidate speaking english",
    "interview answer",
]

def clean_whisper_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t in [".", "..", "...", "!", "?", "—", "-", ","]:
        return ""
    lower = t.lower()
    for h in HALLUCINATION_SUBSTRINGS:
        if h in lower:
            import re
            t = re.sub(re.escape(h), "", t, flags=re.IGNORECASE).strip()
            lower = t.lower()
    cleaned = t.strip(" .,!?:;")
    if cleaned in ["you", "bye", "so", "well"]:
        return ""
    return cleaned

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Production-grade Speech-to-Text endpoint.
    Primary: Groq Cloud Whisper API (whisper-large-v3-turbo) with greedy decoding (temperature=0).
    Fallback: Local HuggingFace Whisper (openai/whisper-tiny.en) if Groq is unreachable.
    """
    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 1500:
        return {"success": True, "text": "", "engine": "none"}

    # Attempt 1: Groq Whisper Cloud API (whisper-large-v3-turbo)
    if GROQ_API_KEY:
        try:
            t0 = time.time()
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            files = {"file": ("audio.webm", audio_bytes, file.content_type or "audio/webm")}
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
                "language": "en",
                "temperature": "0.0"
            }
            resp = requests.post(GROQ_TRANSCRIPTION_URL, headers=headers, files=files, data=data, timeout=8.0)
            if resp.status_code == 200:
                result_json = resp.json()
                raw_text = result_json.get("text", "").strip()
                transcribed = clean_whisper_text(raw_text)
                latency = time.time() - t0
                print(f"[ML Service] Groq Whisper transcribed {len(audio_bytes)}B in {latency:.2f}s: '{transcribed}'")
                return {
                    "success": True,
                    "text": transcribed,
                    "engine": "groq-whisper-large-v3-turbo",
                    "latency": round(latency, 3)
                }
            else:
                print(f"[ML Service] Groq Whisper returned HTTP {resp.status_code}: {resp.text}")
        except Exception as groq_err:
            print(f"[ML Service] Groq Whisper error: {groq_err}. Falling back to local Whisper...")

    # Attempt 2: Local Whisper pipeline fallback (runs offline)
    input_path = None
    output_path = None
    try:
        t0 = time.time()
        input_path = BASE_DIR / f"temp_transcribe_{time.time_ns()}.webm"
        output_path = BASE_DIR / f"temp_transcribe_{time.time_ns()}.wav"
        input_path.write_bytes(audio_bytes)

        subprocess.run([
            FFMPEG_EXE, "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(output_path)
        ], check=True, capture_output=True, text=True)

        audio, _ = librosa.load(str(output_path), sr=16000, mono=True)
        pipe = get_local_whisper()
        import torch
        with torch.inference_mode():
            res = pipe({"raw": audio, "sampling_rate": 16000})
        raw_text = res.get("text", "").strip()
        transcribed = clean_whisper_text(raw_text)
        latency = time.time() - t0
        print(f"[ML Service] Local Whisper fallback transcribed in {latency:.2f}s: '{transcribed}'")
        return {
            "success": True,
            "text": transcribed,
            "engine": "local-whisper-tiny.en",
            "latency": round(latency, 3)
        }
    except Exception as local_err:
        print(f"[ML Service] Local Whisper fallback error: {local_err}")
        return {
            "success": False,
            "error": str(local_err)
        }
    finally:
        if input_path and input_path.exists():
            input_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()

@app.post("/predict/text")
async def predict_text(request: dict):
    try:
        text = request.get("text", "").strip()

        if not text:
            return {
                "success": False,
                "error": "No text provided"
            }

        result = text_predictor.predict(text)

        return result

    except Exception as e:
        print(f"[ML Service] ❌ Text prediction error: {e}")

        return {
            "success": False,
            "error": str(e)
        }
        
class FusionRequest(BaseModel):
    face: dict[str, float] | None = None
    voice: dict[str, float] | None = None
    text: dict[str, float] | None = None


@app.post("/fusion")
def fuse_emotions(request: FusionRequest):
    try:
        fusion = FusionEngine()

        result = fusion.fuse(
            face=request.face,
            voice=request.voice,
            text=request.text,
        )

        return result

    except Exception as e:
        print(f"[ML Service] ❌ Fusion error: {e}")

        return {
            "success": False,
            "error": str(e),
        }
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    print(f"[ML Service] Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port
    )