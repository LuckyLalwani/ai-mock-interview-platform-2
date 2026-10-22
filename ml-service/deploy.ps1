# ==============================================================================
# Deploy ONLY the ML Inference Service to Google Cloud Run (PowerShell)
# ==============================================================================
param (
    [string]$Region = "us-central1",
    [string]$ServiceName = "ai-mock-ml-service",
    [string]$ArtifactRepo = "ml-models"
)

$ErrorActionPreference = "Stop"

$ProjectId = (gcloud config get-value project 2>$null).Trim()
if (-not $ProjectId -or $ProjectId -eq "(unset)") {
    Write-Error "❌ No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "🚀 Deploying ML Inference Service to Google Cloud Run" -ForegroundColor Cyan
Write-Host "Project ID : $ProjectId"
Write-Host "Region     : $Region"
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Enable APIs
Write-Host "`n▶ 1. Enabling GCP services (Cloud Run, Artifact Registry, Cloud Build)..." -ForegroundColor Yellow
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    --project=$ProjectId

# 2. Ensure Artifact Registry repository exists
Write-Host "`n▶ 2. Checking Artifact Registry repository..." -ForegroundColor Yellow
$repoExists = gcloud artifacts repositories describe $ArtifactRepo --location=$Region --project=$ProjectId 2>$null
if (-not $repoExists) {
    gcloud artifacts repositories create $ArtifactRepo `
        --repository-format=docker `
        --location=$Region `
        --description="Docker repository for ML inference models" `
        --project=$ProjectId
    Write-Host "  ✅ Artifact repository '$ArtifactRepo' created." -ForegroundColor Green
} else {
    Write-Host "  ✅ Artifact repository '$ArtifactRepo' already exists." -ForegroundColor Green
}

# 3. Build container with Cloud Build
Write-Host "`n▶ 3. Building ML container image in Cloud Build..." -ForegroundColor Yellow
$ImageTag = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepo/inference-api:latest"

gcloud builds submit . `
    --tag=$ImageTag `
    --project=$ProjectId

# 4. Deploy to Cloud Run
Write-Host "`n▶ 4. Deploying container to Cloud Run (4GB RAM, 2 vCPUs)..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image=$ImageTag `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=4Gi `
    --cpu=2 `
    --timeout=300s `
    --project=$ProjectId

$Url = (gcloud run services describe $ServiceName --region=$Region --project=$ProjectId --format='value(status.url)').Trim()

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "🎉 ML INFERENCE SERVICE IS LIVE!" -ForegroundColor Green
Write-Host "API Endpoint : $Url" -ForegroundColor Cyan
Write-Host "Health Check : $Url/health" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green
Write-Host "`n👉 In your frontend on Vercel, set:" -ForegroundColor Yellow
Write-Host "   VITE_ML_SERVICE_URL=$Url" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Green
