# ☁️ Complete Google Cloud Platform (GCP) Deployment Guide

This guide explains step-by-step how to deploy the **AI Mock Interview Platform** to Google Cloud Platform using **Google Cloud Run** and **Artifact Registry**.

---

## 🏗️ Architecture on GCP

```text
[ Candidate Browser ]
        │
        ├────────────────────────────┐
        ▼ (HTTPS)                    ▼ (HTTPS)
┌──────────────────────┐     ┌──────────────────────┐
│  Frontend (Cloud Run)│     │ ML Service(Cloud Run)│
│  React + Vite SPA    │────>│ FastAPI + PyTorch    │
│  Served via Nginx    │     │ 4Gi RAM, 2 vCPU      │
└──────────────────────┘     └──────────┬───────────┘
        │                               │
        ▼ (HTTPS)                       ▼ (API)
┌──────────────────────┐     ┌──────────────────────┐
│  Supabase (Cloud)    │     │  Groq Whisper / AI   │
│  Database + Auth     │     │  Speech-to-Text      │
└──────────────────────┘     └──────────────────────┘
```

---

## 📋 What Gets Uploaded vs. What is Excluded

The project uses `.gcloudignore` and `.dockerignore` to ensure only necessary code and model weights are uploaded.

### ✅ Uploaded:
- `ml-service/`: Source code, FastAPI endpoints, inference scripts.
- `ml-service/models/`: Model weights (`facial_emotion_model.pt`, `voice_emotion_model.pt`, `text_emotion_model/pytorch_model.bin`, etc.).
- `src/`: React TypeScript components, hooks, router, and assets.
- `package.json`, `package-lock.json`, `vite.config.ts`, `tailwind.config.js`.
- `Dockerfile` & `nginx.conf.template`: Multi-stage production container setups.

### ❌ Excluded (Auto-ignored to save bandwidth & build time):
- `node_modules/` (rebuilt inside Docker)
- `dist/` (rebuilt inside Docker)
- `.venv/` / `venv/` / `__pycache__/`
- Local `.log` files & scratch test images/audio

---

## 🚀 Option 1: One-Click Deploy via Google Cloud Shell (Recommended)

Google Cloud Shell runs directly in your browser with `gcloud`, `docker`, and `git` pre-installed—no local software installation required!

### Step 1: Open Google Cloud Shell
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select your GCP project (ensure billing is enabled).
3. Click the **Activate Cloud Shell** icon (`>_`) in the top-right toolbar.

### Step 2: Upload or Clone the Project
In Cloud Shell, you can either:
- **Clone from GitHub**:
  ```bash
  git clone <YOUR_REPO_URL>
  cd ai-mock-interview-platform
  ```
- **OR upload as a zip**: Click the `⋮` (More) menu in Cloud Shell > **Upload** > Select your zip, then extract:
  ```bash
  unzip ai-mock-interview-platform.zip
  cd ai-mock-interview-platform
  ```

### Step 3: Run the Automated Deployment Script
```bash
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

The script will automatically:
1. Enable Cloud Run, Artifact Registry, and Cloud Build APIs.
2. Create the Docker Artifact Registry repository.
3. Build and deploy `ml-service` to Cloud Run (4GB RAM, 2 vCPUs).
4. Extract the generated ML Service HTTPS URL.
5. Build and deploy the React frontend container with the ML Service URL injected.
6. Print your public production URL!

---

## 💻 Option 2: Deploy from your Local Machine

If you prefer deploying from your local terminal:

### Step 1: Install Google Cloud SDK
- **Windows**: Download and install the [Google Cloud CLI Installer](https://cloud.google.com/sdk/docs/install#windows).
- **macOS**: `brew install google-cloud-sdk`
- **Linux**: `sudo apt-get install google-cloud-cli`

### Step 2: Authenticate and Set Project
```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

### Step 3: Run Deployment Script
- **On Windows PowerShell**:
  ```powershell
  .\deploy-gcp.ps1
  ```
- **On Linux / macOS**:
  ```bash
  chmod +x deploy-gcp.sh
  ./deploy-gcp.sh
  ```

---

## 🛠️ Option 3: Step-by-Step Manual Deployment

If you want complete control and prefer running individual commands:

### 1. Set Environment Variables
```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ARTIFACT_REPO="mock-interview"
```

### 2. Enable Required GCP APIs
```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project=$PROJECT_ID
```

### 3. Create Artifact Registry Repository
```bash
gcloud artifacts repositories create $ARTIFACT_REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for AI Mock Interview Platform" \
    --project=$PROJECT_ID
```

### 4. Build & Deploy the ML Service
Build the image using Cloud Build (no local Docker required):
```bash
gcloud builds submit ./ml-service \
    --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/ml-service:latest \
    --project=$PROJECT_ID
```

Deploy the container to Cloud Run:
```bash
gcloud run deploy ai-mock-ml-service \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/ml-service:latest \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --timeout=300s \
    --set-env-vars="GROQ_API_KEY=your_groq_api_key_here" \
    --project=$PROJECT_ID
```

Copy the generated Service URL (e.g., `https://ai-mock-ml-service-xyz-uc.a.run.app`).

### 5. Build & Deploy the Frontend
Replace `ML_URL`, `SUPABASE_URL`, and `SUPABASE_KEY` with your values:
```bash
export ML_URL="https://ai-mock-ml-service-xyz-uc.a.run.app"
export SUPABASE_URL="https://eeycrjqukhnzjgnlmavg.supabase.co"
export SUPABASE_KEY="your_supabase_anon_key"

gcloud builds submit . \
    --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/frontend:latest \
    --build-arg="VITE_SUPABASE_URL=$SUPABASE_URL" \
    --build-arg="VITE_SUPABASE_ANON_KEY=$SUPABASE_KEY" \
    --build-arg="VITE_ML_SERVICE_URL=$ML_URL" \
    --project=$PROJECT_ID
```

Deploy the frontend container to Cloud Run:
```bash
gcloud run deploy ai-mock-frontend \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/frontend:latest \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=512Mi \
    --cpu=1 \
    --project=$PROJECT_ID
```

Cloud Run will output your live frontend URL!

---

## 🧪 Local Testing with Docker Compose

Before uploading to GCP, test the multi-container setup locally:

```bash
docker compose up --build
```
- Frontend: `http://localhost:3000`
- ML Service: `http://localhost:8000`

---

## ⚙️ GCP Cloud Run Resource Sizing Recommendations

| Service | Min Recommended Memory | Min Recommended CPU | Concurrency |
| :--- | :--- | :--- | :--- |
| **ML Backend (`ml-service`)** | **4 GiB** | **2 vCPUs** | 10 - 20 |
| **Frontend (`frontend`)** | **512 MiB** | **1 vCPU** | 80 |

> **Why 4GB for the ML Service?** PyTorch and multiple deep learning models (Vision Transformer, InceptionResnetV1, Wav2Vec2, DistilRoBERTa) consume ~1.5GB to 2GB in memory when loaded. 4GB provides ample headroom for concurrent requests.

---

## 🔒 Post-Deployment Checklist

1. **Verify ML Health**:
   Visit `https://YOUR_ML_SERVICE_URL/health` in your browser. You should see:
   ```json
   {"status":"healthy","facial_model_loaded":true,"voice_model_loaded":true}
   ```
2. **Camera & Microphone Permissions**:
   Since Cloud Run serves over `https://`, browsers will allow webcam and microphone access automatically after user confirmation.
3. **Custom Domain (Optional)**:
   You can map your custom domain (e.g. `interview.yourdomain.com`) in the Cloud Run Console under **Manage Custom Domains**.
