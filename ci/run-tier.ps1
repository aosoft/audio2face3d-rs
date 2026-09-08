param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("portable", "cuda-lifetime", "tensorrt-model", "reference-parity", "release")]
    [string]$Tier
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string[]]$Command)
    $program = $Command[0]
    $arguments = if ($Command.Length -gt 1) { $Command[1..($Command.Length - 1)] } else { @() }
    & $program @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "release tier command failed ($LASTEXITCODE): $($Command -join ' ')"
    }
}

function Require-Environment {
    param([Parameter(Mandatory = $true)][string[]]$Names)
    foreach ($name in $Names) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ([string]::IsNullOrWhiteSpace($value)) {
            throw "$name is required by the $Tier release tier"
        }
    }
}

switch ($Tier) {
    "portable" {
        & (Join-Path $PSScriptRoot "test-public-api-policy.ps1")
        & (Join-Path $PSScriptRoot "check-public-api.ps1") -Tier portable
        Invoke-Checked @("cargo", "fmt", "--all", "--", "--check")
        Invoke-Checked @("cargo", "check", "--workspace")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", "animation")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", "emotion")
        Invoke-Checked @("cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings")
        Invoke-Checked @("cargo", "test", "--workspace")
        foreach ($features in @("", "animation", "emotion")) {
            $arguments = @("cargo", "test", "-p", "audio2face3d", "--no-default-features")
            if ($features) { $arguments += @("--features", $features) }
            Invoke-Checked $arguments
        }
        $previousRustdocFlags = $env:RUSTDOCFLAGS
        try {
            $env:RUSTDOCFLAGS = "-D warnings"
            Invoke-Checked @("cargo", "doc", "--workspace", "--no-deps")
            foreach ($features in @("", "animation", "emotion")) {
                $arguments = @("cargo", "doc", "-p", "audio2face3d", "--no-deps", "--no-default-features")
                if ($features) { $arguments += @("--features", $features) }
                Invoke-Checked $arguments
            }
        } finally {
            $env:RUSTDOCFLAGS = $previousRustdocFlags
        }
    }
    "cuda-lifetime" {
        Require-Environment @("CUDA_PATH", "AUDIO2FACE3D_CUDA_ARCHS")
        & (Join-Path $PSScriptRoot "check-public-api.ps1") -Tier cuda
        Invoke-Checked @("cargo", "clippy", "-p", "audio2face3d", "--features", "animation,cuda", "--all-targets", "--", "-D", "warnings")
        Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--features", "animation,cuda")
        Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--features", "animation,cuda", "--test", "compile_fail")
        foreach ($features in @("cuda", "emotion,cuda")) {
            Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--no-default-features", "--features", $features)
        }
    }
    "tensorrt-model" {
        Require-Environment @("CUDA_PATH", "TENSORRT_ROOT_DIR", "AUDIO2FACE3D_TEST_FACADE_MODELS")
        $env:PATH = "$(Join-Path $env:CUDA_PATH 'bin');$(Join-Path $env:TENSORRT_ROOT_DIR 'bin');$env:PATH"
        & (Join-Path $PSScriptRoot "check-public-api.ps1") -Tier tensorrt
        foreach ($features in @("tensorrt", "animation,tensorrt", "emotion,tensorrt")) {
            Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", $features, "--all-targets")
            Invoke-Checked @("cargo", "test", "--release", "-p", "audio2face3d", "--no-default-features", "--features", $features, "--test", "compile_fail", "--test", "api_contracts", "--test", "tokio_runtime")
        }
        Invoke-Checked @("cargo", "check", "--workspace", "--all-features")
        Invoke-Checked @("cargo", "clippy", "--workspace", "--all-features", "--all-targets", "--", "-D", "warnings")
        # Optimized test binaries avoid a known MSVC 14.51 debug-linker LNK1000
        # when the large TensorRT import libraries are present.
        Invoke-Checked @("cargo", "test", "--release", "--workspace", "--all-features")
        $previousRustdocFlags = $env:RUSTDOCFLAGS
        try {
            $env:RUSTDOCFLAGS = "-D warnings"
            Invoke-Checked @("cargo", "doc", "--workspace", "--all-features", "--no-deps")
        } finally {
            $env:RUSTDOCFLAGS = $previousRustdocFlags
        }
    }
    "reference-parity" {
        Require-Environment @("AUDIO2FACE_SDK_ROOT", "CUDA_PATH", "TENSORRT_ROOT_DIR", "AUDIO2FACE3D_REFERENCE_WAV_LICENSE")
        Invoke-Checked @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "reference/run-sdk-compatibility.ps1", "-Pipeline", "emotion", "-Execution", "standard", "-Precision", "fp32", "-Tracks", "1", "-Seed", "0")
        Invoke-Checked @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "reference/run-sdk-compatibility.ps1", "-Pipeline", "regression", "-Execution", "teeth-standalone", "-Precision", "fp32", "-Tracks", "2", "-Seed", "0")
    }
    "release" {
        Require-Environment @("CUDA_PATH", "TENSORRT_ROOT_DIR")
        Invoke-Checked @("cargo", "run", "-p", "audio2face3d-cli", "--", "release", "audit", "--report", "target/release-audit.json")
        Invoke-Checked @("cargo", "package", "--workspace", "--allow-dirty", "--no-verify", "--exclude-lockfile")
    }
}

Write-Output "Release tier passed: $Tier"
