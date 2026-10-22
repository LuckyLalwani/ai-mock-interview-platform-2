# ==============================================================================
# Automated GCP Deployment PowerShell Script for AI Mock Interview Platform
# ==============================================================================
param (
    [string]$Region = "us-central1",
    [string]$ArtifactRepo = "mock-interview",
    [string]$MlServiceName = "ai-mock-ml-service",
    [string]$FrontendServiceName = "ai-mock-frontend"
)

$ErrorActionPreference = "Stop"

# Get current GCP project
$ProjectId = (gcloud config get-value project 2>$null).Trim()
if (-not $ProjectId -or $ProjectId -eq "(unset)") {
    Write-Error "❌ No active GCP project found. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "🚀 Deploying AI Mock Interview Platform to Google Cloud Platform" -ForegroundColor Cyan
Write-Host "Project ID : $ProjectId"
Write-Host "Region     : $Region"
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Enable required GCP APIs
Write-Host "`n▶ 1. Enabling required GCP APIs..." -ForegroundColor Yellow
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    --project=$ProjectId

# 2. Create Artifact Registry Repository
Write-Host "`n▶ 2. Checking Artifact Registry repository..." -ForegroundColor Yellow
$repoExists = gcloud artifacts repositories describe $ArtifactRepo --location=$Region --project=$ProjectId 2>$null
if (-not $repoExists) {
    gcloud artifacts repositories create $ArtifactRepo `
        --repository-format=docker `
        --location=$Region `
        --description="Docker repository for AI Mock Interview Platform" `
        --project=$ProjectId
    Write-Host "  ✅ Artifact repository '$ArtifactRepo' created." -ForegroundColor Green
} else {
    Write-Host "  ✅ Artifact repository '$ArtifactRepo' already exists." -ForegroundColor Green
}

# 3. Build & Deploy ML Service
Write-Host "`n▶ 3. Building and deploying ML Service to Cloud Run..." -ForegroundColor Yellow
$MlImage = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepo/ml-service:latest"

gcloud builds submit ./ml-service `
    --tag=$MlImage `
    --project=$ProjectId

$GroqKey = if ($env:GROQ_API_KEY) { $env:GROQ_API_KEY } else { "" }

gcloud run deploy $MlServiceName `
    --image=$MlImage `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=4Gi `
    --cpu=2 `
    --timeout=300s `
    --set-env-vars="GROQ_API_KEY=$GroqKey" `
    --project=$ProjectId

$MlServiceUrl = (gcloud run services describe $MlServiceName --region=$Region --project=$ProjectId --format='value(status.url)').Trim()
Write-Host "  ✅ ML Service deployed at: $MlServiceUrl" -ForegroundColor Green

# 4. Build & Deploy Frontend
Write-Host "`n▶ 4. Building and deploying Frontend to Cloud Run..." -ForegroundColor Yellow
$FrontendImage = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepo/frontend:latest"

$SupabaseUrl = if ($env:VITE_SUPABASE_URL) { $env:VITE_SUPABASE_URL } else { "https://eeycrjqukhnzjgnlmavg.supabase.co" }
$SupabaseKey = if ($env:VITE_SUPABASE_ANON_KEY) { $env:VITE_SUPABASE_ANON_KEY } else { "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVleWNyanF1a2huempnbmxtYXZnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyNzAzNDEsImV4cCI6MjEwMzg0NjM0MX0.Ai3W8t-80COj27p2Ax4PdtN_0PJgA4-N0h5ZjA3nOkY" }

gcloud builds submit . `
    --tag=$FrontendImage `
    --build-arg="VITE_SUPABASE_URL=$SupabaseUrl" `
    --build-arg="VITE_SUPABASE_ANON_KEY=$SupabaseKey" `
    --build-arg="VITE_ML_SERVICE_URL=$MlServiceUrl" `
    --project=$ProjectId

gcloud run deploy $FrontendServiceName `
    --image=$FrontendImage `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=512Mi `
    --cpu=1 `
    --project=$ProjectId

$FrontendUrl = (gcloud run services describe $FrontendServiceName --region=$Region --project=$ProjectId --format='value(status.url)').Trim()

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "🎉 DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "Frontend URL : $FrontendUrl" -ForegroundColor Cyan
Write-Host "ML Backend   : $MlServiceUrl" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Green
