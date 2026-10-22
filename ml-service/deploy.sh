#!/usr/bin/env bash
# ==============================================================================
# Deploy ONLY the ML Inference Service to Google Cloud Run
# ==============================================================================
set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="ai-mock-ml-service"
ARTIFACT_REPO="ml-models"
IMAGE_TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/inference-api:latest"

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "❌ No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "======================================================"
echo "🚀 Deploying ML Inference Service to Google Cloud Run"
echo "Project ID : $PROJECT_ID"
echo "Region     : $REGION"
echo "======================================================"

# 1. Enable required APIs (safely ignore if rate-limited or already active)
echo "▶ 1. Checking GCP services (Cloud Run, Artifact Registry, Cloud Build)..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID" 2>/dev/null || echo "  ℹ️ Continuing with existing service configurations..."

# 2. Create Artifact Registry repository if it doesn't exist
echo "▶ 2. Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$ARTIFACT_REPO" --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud artifacts repositories create "$ARTIFACT_REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for ML inference models" \
        --project="$PROJECT_ID"
    echo "  ✅ Artifact repository '$ARTIFACT_REPO' created."
else
    echo "  ✅ Artifact repository '$ARTIFACT_REPO' already exists."
fi

# 3. Build container with Cloud Build
echo "▶ 3. Building ML container image in Cloud Build..."
gcloud builds submit . \
    --tag="$IMAGE_TAG" \
    --project="$PROJECT_ID"

# 4. Deploy to Cloud Run
echo "▶ 4. Deploying container to Cloud Run (4GB RAM, 2 vCPUs)..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE_TAG" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --timeout=300s \
    --project="$PROJECT_ID"

URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')

echo ""
echo "======================================================"
echo "🎉 ML INFERENCE SERVICE IS LIVE!"
echo "API Endpoint : $URL"
echo "Health Check : $URL/health"
echo "======================================================"
echo ""
echo "👉 In your frontend on Vercel, set:"
echo "   VITE_ML_SERVICE_URL=$URL"
echo "======================================================"
