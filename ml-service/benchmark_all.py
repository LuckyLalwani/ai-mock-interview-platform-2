"""
Comprehensive benchmark measuring exact inference latency and pipeline
execution time for all ML models in the platform.
"""
import os
import sys
import time
import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
os.chdir(ROOT)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print(f"BENCHMARKING ALL MODELS ON: {device.upper()}")
print(f"PyTorch Version: {torch.__version__}")
if device == "cpu":
    print(f"CPU Threads: {torch.get_num_threads()}")
print("=" * 70)


def format_stats(latencies_ms):
    return {
        "avg_ms": round(float(np.mean(latencies_ms)), 2),
        "min_ms": round(float(np.min(latencies_ms)), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
        "std_ms": round(float(np.std(latencies_ms)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(latencies_ms)), 2),
    }


# =====================================================================
# 1. FACIAL EMOTION MODEL (ViT-Base-Patch16-224 + TTA)
# =====================================================================
print("\n" + "#" * 70)
print("1. FACIAL EMOTION MODEL (ViT-Base-Patch16-224 with 2-pass TTA)")
print("#" * 70)

try:
    from facial_inference import FacialInference

    model_path = os.path.join(MODELS_DIR, "facial_emotion_model.pt")
    classes_path = os.path.join(MODELS_DIR, "vit_facial_emotion_classes.npy")

    print(f"Loading model from: {model_path}")
    t0 = time.perf_counter()
    facial_inf = FacialInference(model_path, classes_path)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Model Load Time: {load_time:.2f} ms")

    # Sample input (standard 640x480 webcam frame)
    sample_frame = cv2.imread(os.path.join(ROOT, "test_face.jpg"))
    if sample_frame is None:
        sample_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    sample_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

    # Warmup
    print("Warming up facial model...")
    for _ in range(2):
        _ = facial_inf.predict(sample_rgb)

    # Benchmark: Preprocessing
    prep_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = facial_inf.preprocess(sample_rgb)
        prep_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark: Full End-to-End predict() (preprocess + 2 forward passes ViT + softmax)
    e2e_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = facial_inf.predict(sample_rgb)
        e2e_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark: Pure Model Forward Pass (without image resize)
    dummy_tensor = facial_inf.preprocess(sample_rgb)
    fwd_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            out1 = facial_inf.model(dummy_tensor)
            out2 = facial_inf.model(torch.flip(dummy_tensor, dims=[3]))
        fwd_times.append((time.perf_counter() - t0) * 1000)

    prep_stats = format_stats(prep_times)
    fwd_stats = format_stats(fwd_times)
    e2e_stats = format_stats(e2e_times)

    print(f"Preprocessing Latency:  avg={prep_stats['avg_ms']}ms, min={prep_stats['min_ms']}ms, max={prep_stats['max_ms']}ms")
    print(f"Pure ViT (TTA 2-pass):  avg={fwd_stats['avg_ms']}ms, min={fwd_stats['min_ms']}ms, max={fwd_stats['max_ms']}ms")
    print(f"End-to-End Latency:     avg={e2e_stats['avg_ms']}ms, min={e2e_stats['min_ms']}ms, max={e2e_stats['max_ms']}ms")
    print(f"Throughput:             ~{e2e_stats['throughput_fps']} inferences/sec")
    print(f"Sample Prediction:      {max(res, key=res.get)} ({res[max(res, key=res.get)]*100:.1f}%)")

except Exception as e:
    print(f"Facial model benchmark error: {e}")


# =====================================================================
# 2. PROCTORING & FACE VERIFICATION (MTCNN + Inception-ResNet-v1 VGGFace2)
# =====================================================================
print("\n" + "#" * 70)
print("2. FACE VERIFICATION & PROCTORING (MTCNN Detector + Inception-ResNet-v1)")
print("#" * 70)

try:
    from face_verifier import FaceVerifier

    t0 = time.perf_counter()
    verifier = FaceVerifier(threshold=0.70)
    print(f"Verifier Load Time: {(time.perf_counter() - t0) * 1000:.2f} ms")

    test_img = cv2.imread(os.path.join(ROOT, "test_face.jpg"))
    if test_img is None:
        test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # Register candidate face
    print("Registering candidate face...")
    reg_res = verifier.register(test_img)
    print(f"Registration Result: status={reg_res['status']}")

    # Warmup
    for _ in range(2):
        verifier.verify(test_img)

    # Benchmark: MTCNN Face Detection
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    detect_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = verifier.detector.detect(pil_img)
        detect_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark: End-to-End Face Verification (Detect + Align + InceptionResnet + Cosine Sim)
    verify_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        v_res = verifier.verify(test_img)
        verify_times.append((time.perf_counter() - t0) * 1000)

    detect_stats = format_stats(detect_times)
    verify_stats = format_stats(verify_times)

    print(f"MTCNN Detection Latency:  avg={detect_stats['avg_ms']}ms, min={detect_stats['min_ms']}ms, max={detect_stats['max_ms']}ms")
    print(f"Full Verification Latency: avg={verify_stats['avg_ms']}ms, min={verify_stats['min_ms']}ms, max={verify_stats['max_ms']}ms")
    print(f"Throughput:               ~{verify_stats['throughput_fps']} verifications/sec")
    print(f"Sample Verification:      status={v_res['status']}, similarity={v_res.get('similarity')}")

except Exception as e:
    print(f"Face verifier benchmark error: {e}")


# =====================================================================
# 3. VOICE EMOTION MODEL (Wav2Vec2 + TemporalAttentionPooling)
# =====================================================================
print("\n" + "#" * 70)
print("3. VOICE EMOTION MODEL (Wav2Vec2 + Temporal Attention Pooling)")
print("#" * 70)

try:
    from voice_inference import VoiceInference

    voice_path = os.path.join(MODELS_DIR, "voice_emotion_model.pt")
    t0 = time.perf_counter()
    voice_inf = VoiceInference(voice_path)
    print(f"Voice Model Load Time: {(time.perf_counter() - t0) * 1000:.2f} ms")

    # Generate 3.0 seconds of audio at 16000 Hz
    duration_sec = 3.0
    sr = 16000
    dummy_audio = np.random.randn(int(duration_sec * sr)).astype(np.float32) * 0.1

    # Warmup
    print(f"Warming up voice model with {duration_sec}s audio...")
    for _ in range(2):
        voice_inf.predict(dummy_audio)

    # Benchmark
    voice_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        v_pred = voice_inf.predict(dummy_audio)
        voice_times.append((time.perf_counter() - t0) * 1000)

    voice_stats = format_stats(voice_times)
    rtf = (voice_stats['avg_ms'] / 1000.0) / duration_sec

    print(f"Audio Duration Tested:   {duration_sec} seconds ({sr} Hz mono)")
    print(f"Voice Inference Latency: avg={voice_stats['avg_ms']}ms, min={voice_stats['min_ms']}ms, max={voice_stats['max_ms']}ms")
    print(f"Real-Time Factor (RTF):  {rtf:.3f}x (values < 1.0x mean faster than real-time)")
    print(f"Sample Prediction:       {max(v_pred, key=v_pred.get)} ({v_pred[max(v_pred, key=v_pred.get)]*100:.1f}%)")

except Exception as e:
    print(f"Voice model benchmark error: {e}")


# =====================================================================
# 4. TEXT EMOTION MODEL (DistilRoBERTa Sequence Classification)
# =====================================================================
print("\n" + "#" * 70)
print("4. TEXT EMOTION MODEL (DistilRoBERTa Sequence Classification)")
print("#" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "j-hartmann/emotion-english-distilroberta-base"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    txt_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    txt_model.eval()
    print(f"Text Model Load Time: {(time.perf_counter() - t0) * 1000:.2f} ms")

    sample_answer = (
        "I am confident in my technical skills and experience with scalable cloud systems. "
        "I thrive in collaborative agile environments and enjoy solving complex algorithmic challenges."
    )

    # Warmup
    inputs = tok(sample_answer, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    for _ in range(2):
        with torch.no_grad():
            _ = txt_model(**inputs)

    # Benchmark tokenization
    tok_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = tok(sample_answer, return_tensors="pt", truncation=True, max_length=512)
        tok_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark pure model inference
    txt_fwd_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = txt_model(**inputs)
        txt_fwd_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark full end-to-end (tokenize + inference + softmax)
    txt_e2e_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        inp = tok(sample_answer, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = txt_model(**inp)
            probs = torch.softmax(out.logits, dim=-1)
        txt_e2e_times.append((time.perf_counter() - t0) * 1000)

    tok_stats = format_stats(tok_times)
    txt_fwd_stats = format_stats(txt_fwd_times)
    txt_e2e_stats = format_stats(txt_e2e_times)

    print(f"Input Length:            {len(sample_answer.split())} words / {inputs['input_ids'].shape[1]} tokens")
    print(f"Tokenization Latency:    avg={tok_stats['avg_ms']}ms, min={tok_stats['min_ms']}ms, max={tok_stats['max_ms']}ms")
    print(f"Pure Model Inference:    avg={txt_fwd_stats['avg_ms']}ms, min={txt_fwd_stats['min_ms']}ms, max={txt_fwd_stats['max_ms']}ms")
    print(f"End-to-End Latency:      avg={txt_e2e_stats['avg_ms']}ms, min={txt_e2e_stats['min_ms']}ms, max={txt_e2e_stats['max_ms']}ms")
    print(f"Throughput:              ~{txt_e2e_stats['throughput_fps']} answers/sec")

except Exception as e:
    print(f"Text model benchmark error: {e}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
