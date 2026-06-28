param(
    [string] $Model = "D:\llamacpp\models\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
    [string] $Bench = "D:\llamacpp\build\bin\Release\llama-bench.exe",
    [string] $OutputDir = "D:\llamacpp\metrics\predictive-prefetch",
    [int] $PromptTokens = 512,
    [int] $GenTokens = 128
)

# Do NOT use ErrorActionPreference=Stop here: llama-bench writes normal banners
# to stderr, and on Windows PowerShell 5.1 a redirected native stderr line is
# wrapped as a NativeCommandError, which would abort the script.
$ErrorActionPreference = "Continue"

# CUDA 13.x keeps its runtime DLLs (cudart64_13, cublas64_13, ...) in bin\x64.
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

# label => llama-bench args. Only the cpu-moe config exercises the instrumented
# host->device expert-copy path, so metrics are enabled only there.
$configs = @(
    @{ Label = "ngl0-cpu";      Args = @("-ngl", "0");                  Metrics = $false; Reps = 2 },
    @{ Label = "ngl99-ncmoe48"; Args = @("-ngl", "99", "-ncmoe", "48"); Metrics = $true;  Reps = 3 }
)

foreach ($cfg in $configs) {
    $label  = $cfg.Label
    $outLog = Join-Path $OutputDir "stage0-$stamp-$label.jsonl"   # llama-bench stdout (-o jsonl)
    $errLog = Join-Path $OutputDir "stage0-$stamp-$label.err.log" # ggml stderr incl. moe metrics
    Write-Host "=== $label ==="

    if ($cfg.Metrics) {
        $env:GGML_MOE_PREFETCH_METRICS = "1"
        $env:GGML_MOE_PREFETCH_METRICS_INTERVAL = "48"
    } else {
        Remove-Item Env:GGML_MOE_PREFETCH_METRICS -ErrorAction SilentlyContinue
    }

    $allArgs = @("-m", $Model) + $cfg.Args + @("-t", "8", "-p", "$PromptTokens", "-n", "$GenTokens", "-r", "$($cfg.Reps)", "-o", "jsonl")
    $proc = Start-Process -FilePath $Bench -ArgumentList $allArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Write-Host "  exit=$($proc.ExitCode) out=$outLog err=$errLog"
}

Write-Host "DONE stamp=$stamp"
