param(
    [ValidateSet('train', 'val')]
    [string]$Split = 'train',
    [int]$MaxImages = 12,
    [switch]$SkipVisual
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    $python = Join-Path $root '.venv311\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        throw "Python not found at $python. Use the .venv311 environment."
    }

    $requiredScripts = @(
        'validate_annotations.py',
        'track_annotation_progress.py',
        'visualize_annotations.py'
    )

    foreach ($script in $requiredScripts) {
        if (-not (Test-Path (Join-Path $root $script))) {
            throw "Missing required script: $script"
        }
    }

    Write-Host "[1/3] Validating $Split annotations..." -ForegroundColor Cyan
    & $python validate_annotations.py --split $Split

    Write-Host "[2/3] Updating annotation progress report..." -ForegroundColor Cyan
    & $python track_annotation_progress.py

    if (-not $SkipVisual) {
        Write-Host "[3/3] Generating annotation visualization grids..." -ForegroundColor Cyan
        & $python visualize_annotations.py
    } else {
        Write-Host "[3/3] Visualization skipped (--SkipVisual)." -ForegroundColor Yellow
    }

    Write-Host "Done. QA checks finished for split: $Split" -ForegroundColor Green
}
finally {
    Pop-Location
}
