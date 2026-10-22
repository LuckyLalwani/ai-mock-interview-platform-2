import torch
from transformers import pipeline
import librosa
import time

print("Testing Whisper performance...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    device="cpu",
    torch_dtype=torch.float32
)

audio, sr = librosa.load("test_voice.wav", sr=16000)

# Warmup
with torch.inference_mode():
    pipe({"raw": audio[:16000], "sampling_rate": 16000})

# Benchmark full 5-second file
t0 = time.time()
with torch.inference_mode():
    result = pipe({"raw": audio, "sampling_rate": 16000})
elapsed = time.time() - t0

print(f"Elapsed: {elapsed:.2f}s | Result: {result['text']}")
