# 🧠 ML Inference Service - Standalone GCP Deployment Guide

This guide covers deploying **ONLY** the ML Inference Service to **Google Cloud Platform (Cloud Run)**, while leaving your frontend and other modules on **Vercel**.

---

## 🎯 Architecture: Vercel + GCP Cloud Run

```text
┌─────────────────────────────────┐
│     Frontend on VERCEL          │
│   (React / Next.js / Vite)      │
└────────────────┬────────────────┘
                 │
                 │ HTTPS REST API Calls
                 │ (VITE_ML_SERVICE_URL=https://ml-service-xyz.a.run.app)
                 ▼
┌─────────────────────────────────┐
│     Inference API on GCP        │
│          (Cloud Run)            │
│  • FastAPI Python Server        │
│  • PyTorch + Transformers       │
│  • MTCNN + Face Verification    │
│  • Wav2Vec2 + Audio Inference   │
│  • Multimodal Fusion Engine     │
└─────────────────────────────────┘
```

### Why this is the best setup:
1. **Vercel limitation**: Vercel Serverless Functions have a strict 50 MB - 250 MB size limit and short timeouts. They **cannot** run PyTorch, Librosa, FFmpeg, or large model weights.
2. **Cloud Run advantage**: GCP Cloud Run runs full Docker containers with up to 32 GB RAM, multiple vCPUs, native C/C++ audio libraries, and **scales to 0** when idle (meaning **zero cost** when you're not running tests).

---

## 📁 What Files to Upload to GCP

You **ONLY** need the `ml-service` directory. Do not upload `src/`, `node_modules/`, or frontend files.

```text
ml-service/
├── Dockerfile                  # GCP Cloud Run container definition
├── requirements.txt            # Python dependencies (PyTorch CPU, FastAPI, etc.)
├── .dockerignore               # Ignores .venv and temp files
├── .gcloudignore               # Ignores non-essential files during upload
├── server.py                   # FastAPI REST API endpoints
├── facial_inference.py         # Facial emotion inference (ViT / EfficientNet)
├── voice_inference.py          # Voice emotion inference (Wav2Vec2 / CNN-BiLSTM)
├── text_inference.py           # Text emotion inference (DistilRoBERTa)
├── face_verifier.py            # MTCNN + InceptionResnetV1 face verification
├── frame_quality.py            # Video frame quality validator
├── fusion_engine_new.py        # Multimodal fusion algorithm
├── deploy.sh                   # Deployment script for Linux / Cloud Shell
├── deploy.ps1                  # Deployment script for Windows PowerShell
└── models/                     # ALL trained model weights
    ├── facial_emotion_model.pt
    ├── facial_emotion_classes.npy
    ├── vit_facial_emotion_classes.npy
    ├── voice_emotion_model.pt
    ├── voice_model_config.json
    └── text_emotion_model/
        ├── config.json
        ├── emotion_config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
```

---

## 🚀 Quick Deploy: Cloud Shell (Browser - No Local Setup Needed)

1. Open **[Google Cloud Console](https://console.cloud.google.com/)** and select your project.
2. Open **Cloud Shell** by clicking the `>_` terminal icon in the top right.
3. Zip your `ml-service` folder locally and click **⋮ (More) > Upload** in Cloud Shell to upload `ml-service.zip`.
4. Run in Cloud Shell:
   ```bash
   unzip ml-service.zip
   cd ml-service
   chmod +x deploy.sh
   ./deploy.sh
   ```
5. In ~3 minutes, Cloud Run will print your live HTTPS API URL:
   ```text
   https://ai-mock-ml-service-xxxxxx-uc.a.run.app
   ```

---

## 💻 Alternative: Deploy from Local Terminal (PowerShell / Bash)

If you have `gcloud` installed locally:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Navigate to ml-service
cd ml-service

# Run deployment
./deploy.sh          # Linux / macOS
.\deploy.ps1         # Windows PowerShell
```

---

## 🧪 How to Test Your Deployed ML API

Once deployed, you will get an HTTPS URL (e.g. `https://ai-mock-ml-service-xyz.a.run.app`). Test it immediately:

### 1. Health Check
```bash
curl https://YOUR_URL/health
```
**Expected response**:
```json
{"status":"healthy","facial_model_loaded":true,"voice_model_loaded":true}
```

### 2. Text Emotion Inference Test
```bash
curl -X POST "https://YOUR_URL/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am confident and excited about this opportunity!"}'
```
**Expected response**:
```json
{"success": true, "emotion": "joy", "confidence": 0.96, "predictions": {...}}
```

### 3. Multimodal Fusion Test
```bash
curl -X POST "https://YOUR_URL/fusion" \
     -H "Content-Type: application/json" \
     -d '{
       "face": {"happy": 0.8, "neutral": 0.2},
       "voice": {"happy": 0.7, "neutral": 0.3},
       "text": {"happy": 0.9, "neutral": 0.1}
     }'
```
**Expected response**:
```json
{
  "success": true,
  "dominant_emotion": "happiness",
  "behavioral_metrics": {
    "confidence": 85.0,
    "stress": 12.0,
    "fluency": 80.0,
    ...
  },
  "overall_score": 88.5
}
```

---

## 🔗 How to Connect Your Frontend on Vercel

In your frontend project on Vercel:
1. Go to **Vercel Project Dashboard** > **Settings** > **Environment Variables**.
2. Add / Update:
   ```env
   VITE_ML_SERVICE_URL=https://YOUR_CLOUD_RUN_URL
   ```
3. Redeploy your frontend on Vercel.

Your Vercel frontend will now seamlessly call the high-performance ML inference models running on Google Cloud!
