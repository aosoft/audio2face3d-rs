param(
    [ValidateSet("regression", "diffusion", "emotion")]
    [string]$Pipeline = "regression",
    [ValidateSet("fp32", "fp16")]
    [string]$Precision = "fp32",
    [ValidateSet("standard", "interactive-random", "interactive-all", "blendshape-cpu", "blendshape-gpu")]
    [string]$Execution = "standard",
    [int]$Tracks = 1,
    [UInt64]$Seed = 0
)

$ErrorActionPreference = "Stop"
$SdkRoot = $env:AUDIO2FACE_SDK_ROOT
$TensorRtRoot = $env:TENSORRT_ROOT_DIR
$CudaRoot = $env:CUDA_PATH
if (-not $SdkRoot) {
    throw "AUDIO2FACE_SDK_ROOT must point to the local NVIDIA Audio2Face-3D-SDK checkout"
}
if (-not $TensorRtRoot) {
    throw "TENSORRT_ROOT_DIR must point to the local TensorRT installation"
}
if (-not $CudaRoot) {
    throw "CUDA_PATH must point to the local CUDA Toolkit installation"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$testRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "compatible_test"))
$Wav = Join-Path $testRoot "input.wav"
if (-not (Test-Path -LiteralPath $Wav -PathType Leaf)) {
    throw "Evaluation audio was not found: place a mono PCM16 16 kHz WAV at $Wav"
}

$Model = $env:AUDIO2FACE3D_REFERENCE_MODEL
if (-not $Model) {
    $relativeModel = switch ($Pipeline) {
        "regression" {
            if ($Precision -eq "fp16") {
                "_data/generated/audio2face-sdk/samples/data/mark/model_fp16.json"
            } else {
                "_data/generated/audio2face-sdk/samples/data/mark/model.json"
            }
        }
        "diffusion" { "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json" }
        "emotion" { "_data/generated/audio2emotion-sdk/samples/model/model.json" }
    }
    $Model = Join-Path $SdkRoot $relativeModel
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Reference model was not found: $Model"
}

$caseName = "$Pipeline-$Execution-$Precision-tracks$Tracks-seed$Seed"
$fixture = Join-Path $testRoot "fixture"
$outputRoot = Join-Path (Join-Path $testRoot "results") $caseName
$cppArtifact = Join-Path $outputRoot "cpp"
$rustArtifact = Join-Path $outputRoot "rust"
$report = Join-Path $outputRoot "comparison.json"
$cppRunner = & (Join-Path $PSScriptRoot "build-cpp-runner.ps1")

$env:PATH = @(
    (Join-Path $SdkRoot "_build/release/audio2x-sdk/bin")
    (Join-Path $TensorRtRoot "bin")
    (Join-Path $CudaRoot "bin")
    $env:PATH
) -join ";"

$fixtureLicense = if ($env:AUDIO2FACE3D_REFERENCE_WAV_LICENSE) {
    $env:AUDIO2FACE3D_REFERENCE_WAV_LICENSE
} else {
    "user-provided-not-for-redistribution"
}
$fixtureArguments = @(
    "run", "-p", "audio2face3d-cli", "--", "reference", "fixture", "wav",
    $Wav, $fixture, "--name", "compatible-test-input", "--license", $fixtureLicense
)
if ($env:AUDIO2FACE3D_REFERENCE_WAV_SHA256) {
    $fixtureArguments += @("--sha256", $env:AUDIO2FACE3D_REFERENCE_WAV_SHA256)
}
& cargo @fixtureArguments
if ($LASTEXITCODE -ne 0) { throw "fixture preparation failed" }

cargo run -p audio2face3d-cli --features runtime -- reference capture `
    $Model $fixture $rustArtifact --execution $Execution --precision $Precision `
    --tracks $Tracks --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Rust reference capture failed" }

& $cppRunner $Pipeline $Execution $Model (Join-Path $fixture "samples.f32le") `
    $cppArtifact $Precision $Tracks $Seed
if ($LASTEXITCODE -ne 0) { throw "C++ reference capture failed" }

cargo run -p audio2face3d-cli -- reference compare $cppArtifact $rustArtifact `
    --tolerances (Join-Path $PSScriptRoot "tolerances.json") --report $report
if ($LASTEXITCODE -ne 0) {
    throw "reference parity failed; inspect $report"
}

Write-Output "SDK compatibility comparison passed: $report"
