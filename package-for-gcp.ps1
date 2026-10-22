$ErrorActionPreference = "Stop"

$workspaceRoot = "d:\ai-mock-interview-platform-main"
$tempDir = Join-Path $env:TEMP "ml-service-upload"
$zipDest = Join-Path $workspaceRoot "ml-service-light.zip"

Write-Host "Creating lightweight package for GCP upload..." -ForegroundColor Cyan

if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Copy ml-service contents
Get-ChildItem -Path "$workspaceRoot\ml-service" -Exclude @(".venv", "__pycache__", "benchmark_results.txt", "test_*.jpg", "test_*.wav") | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $tempDir -Recurse
}

# Remove the 313MB model from the zip (Dockerfile will auto-download it on GCP fiber in 2s)
$largeModelPath = Join-Path $tempDir "models\text_emotion_model\pytorch_model.bin"
if (Test-Path $largeModelPath) {
    Remove-Item -Path $largeModelPath -Force
    Write-Host "  -> Excluded pytorch_model.bin (will be auto-downloaded on GCP Cloud Build)" -ForegroundColor Yellow
}

# Remove any stray __pycache__ or venvs in temp
Get-ChildItem -Path $tempDir -Include @("__pycache__", ".venv", "*.pyc") -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path $zipDest) { Remove-Item -Force $zipDest }

Write-Host "Compressing to ml-service-light.zip..." -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $zipDest -CompressionLevel Optimal
Remove-Item -Recurse -Force $tempDir

$item = Get-Item $zipDest
$sizeMB = [math]::Round($item.Length / 1MB, 2)
Write-Host "Done! Created: $zipDest ($sizeMB MB)" -ForegroundColor Green
