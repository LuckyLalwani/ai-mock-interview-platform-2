#!/usr/bin/env bash
# ==============================================================================
# Automated GCP Deployment Script for AI Mock Interview Platform
# ==============================================================================
set -e

# Configuration (Customize if needed)
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
ARTIFACT_REPO="mock-interview"
ML_SERVICE_NAME="ai-mock-ml-service"
FRONTEND_SERVICE_NAME="ai-mock-frontend"

# Check project ID
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "❌ No active GCP project found. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "================================================================="
echo "🚀 Deploying AI Mock Interview Platform to Google Cloud Platform"
echo "Project ID : $PROJECT_ID"
echo "Region     : $REGION"
echo "================================================================="

# 1. Enable required GCP APIs
echo ""
echo "▶ 1. Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

# 2. Create Artifact Registry Repository (if not already existing)
echo ""
echo "▶ 2. Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$ARTIFACT_REPO" --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud artifacts repositories create "$ARTIFACT_REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for AI Mock Interview Platform" \
        --project="$PROJECT_ID"
    echo "  ✅ Artifact repository '$ARTIFACT_REPO' created."
else
    echo "  ✅ Artifact repository '$ARTIFACT_REPO' already exists."
fi

# 3. Build & Deploy ML Service
echo ""
echo "▶ 3. Building and deploying ML Service to Cloud Run..."
ML_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/ml-service:latest"

gcloud builds submit ./ml-service \
    --tag="$ML_IMAGE" \
    --project="$PROJECT_ID"

# Read GROQ API Key from .env or fallback
GROQ_KEY="${GROQ_API_KEY:-}"

gcloud run deploy "$ML_SERVICE_NAME" \
    --image="$ML_IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --timeout=300s \
    --set-env-vars="GROQ_API_KEY=$GROQ_KEY" \
    --project="$PROJECT_ID"

# Get ML Service URL
ML_SERVICE_URL=$(gcloud run services describe "$ML_SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')
echo "  ✅ ML Service deployed at: $ML_SERVICE_URL"

# 4. Build & Deploy Frontend
echo ""
echo "▶ 4. Building and deploying Frontend to Cloud Run..."
FRONTEND_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/frontend:latest"

# Supabase Credentials (from root .env or defaults)
SUPABASE_URL="${VITE_SUPABASE_URL:-https://eeycrjqukhnzjgnlmavg.supabase.co}"
SUPABASE_KEY="${VITE_SUPABASE_ANON_KEY:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVleWNyanF1a2huempnbmxtYXZnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyNzAzNDEsImV4cCI6MjEwMzg0NjM0MX0.Ai3W8t-80COj27p2Ax4PdtN_0PJgA4-N0h5ZjA3nOkY}"

gcloud builds submit . \
    --tag="$FRONTEND_IMAGE" \
    --substitutions="_VITE_SUPABASE_URL=$SUPABASE_URL,_VITE_SUPABASE_ANON_KEY=$SUPABASE_KEY,_ML_SERVICE_URL=$ML_SERVICE_URL" \
    --project="$PROJECT_ID"

gcloud run deploy "$FRONTEND_SERVICE_NAME" \
    --image="$FRONTEND_IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=512Mi \
    --cpu=1 \
    --project="$PROJECT_ID"

FRONTEND_URL=$(gcloud run services describe "$FRONTEND_SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')

echo ""
echo "================================================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "Frontend URL : $FRONTEND_URL"
echo "ML Backend   : $ML_SERVICE_URL"
echo "================================================================="
