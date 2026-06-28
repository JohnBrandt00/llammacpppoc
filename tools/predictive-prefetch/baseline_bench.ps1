param(
    [string] $Model = ".\models\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
    [string] $BuildDir = ".\build",
    [int[]] $GpuLayers = @(0, 10, 20, 30, 40, 48),
    [int] $Threads = 8,
    [int] $PromptTokens = 512,
    [int] $GenTokens = 128,
    [int] $Repetitions = 3,
    [string] $OutputDir = ".\metrics\predictive-prefetch",
    [switch] $EnableMoeMetrics
)

$ErrorActionPreference = "Stop"

$benchCandidates = @(
    Join-Path $BuildDir "bin\Release\llama-bench.exe",
    Join-Path $BuildDir "bin\llama-bench.exe",
    Join-Path $BuildDir "tools\llama-bench\Release\llama-bench.exe",
    Join-Path $BuildDir "tools\llama-bench\llama-bench.exe"
)

$Bench = $benchCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Bench) {
    throw "Could not find llama-bench.exe under $BuildDir. Build with: cmake --build $BuildDir --config Release --target llama-bench"
}

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$summary = Join-Path $OutputDir "baseline-$stamp.jsonl"

if ($EnableMoeMetrics) {
    $env:GGML_MOE_PREFETCH_METRICS = "1"
    $env:GGML_MOE_PREFETCH_METRICS_INTERVAL = "48"
}

foreach ($ngl in $GpuLayers) {
    $log = Join-Path $OutputDir "baseline-$stamp-ngl-$ngl.log"
    Write-Host "=== ngl=$ngl ==="
    Write-Host "log: $log"

    & $Bench `
        -m $Model `
        -ngl $ngl `
        -t $Threads `
        -p $PromptTokens `
        -n $GenTokens `
        -r $Repetitions `
        -o jsonl `
        2>&1 | Tee-Object -FilePath $log

    Get-Content $log | Where-Object { $_.TrimStart().StartsWith("{") } | Add-Content $summary
}

Write-Host "summary: $summary"
