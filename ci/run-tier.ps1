param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("portable", "cuda-lifetime", "tensorrt-model", "reference-parity", "release")]
    [string]$Tier,
    [string]$PlatformConfig
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string[]]$Command)
    $program = $Command[0]
    $arguments = if ($Command.Length -gt 1) { $Command[1..($Command.Length - 1)] } else { @() }
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $program
    $start.WorkingDirectory = $repoRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    # Windows argv quoting works on Windows PowerShell 5.1 as well as PowerShell 7.
    $quoted = foreach ($argument in $arguments) {
        '"' + [regex]::Replace([regex]::Replace([string]$argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
    }
    $start.Arguments = $quoted -join ' '
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    if ($PlatformConfig) { $start.EnvironmentVariables['AUDIO2FACE3D_PLATFORM_CONFIG'] = [IO.Path]::GetFullPath([IO.Path]::Combine($repoRoot, $PlatformConfig)) }
    if ($script:runtimeInfo) {
        # Legacy native tests consume SDK roots. These settings belong only to this child.
        $start.EnvironmentVariables['CUDA_PATH'] = $script:runtimeInfo.cuda_root
        $start.EnvironmentVariables['TENSORRT_ROOT_DIR'] = $script:runtimeInfo.tensorrt_root
        $directories = @($script:runtimeInfo.libraries | ForEach-Object { Split-Path -Parent $_.path } | Select-Object -Unique)
        $start.EnvironmentVariables['PATH'] = ($directories + @($start.EnvironmentVariables['PATH'])) -join [IO.Path]::PathSeparator
    }
    $process = [System.Diagnostics.Process]::Start($start)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $outputText = $stdout.GetAwaiter().GetResult()
    $errorText = $stderr.GetAwaiter().GetResult()
    if ($outputText) { Write-Output $outputText }
    if ($errorText) { [Console]::Error.WriteLine($errorText) }
    $code = $process.ExitCode
    $process.Dispose()
    if ($code -ne 0) { throw "release tier command failed ($code): $($Command -join ' ')" }
}

$script:runtimeInfo = $null
if ($PlatformConfig) {
    # Parse through the same Rust CLI implementation used by applications; this does not load DLLs.
    $diagnostic = & cargo run --quiet --locked -p audio2face3d --no-default-features --features cli -- --platform-config $PlatformConfig doctor --json
    if ($LASTEXITCODE -ne 0) { throw 'runtime configuration discovery failed' }
    $script:runtimeInfo = $diagnostic | ConvertFrom-Json
    if (-not $script:runtimeInfo.cuda_root -or -not $script:runtimeInfo.tensorrt_root) {
        throw 'Native CI legacy tests require SDK roots (cuda-root and tensorrt-root) resolved from PlatformConfig; library directory layouts are tested through the explicit client API.'
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
        Invoke-Checked @("cargo", "fmt", "--all", "--", "--check")
        Invoke-Checked @("cargo", "check", "--workspace", "--no-default-features", "--features", "audio2face3d/cli,audio2face3d/mock,audio2face3d/client-grpc,audio2face3d-server/cli,audio2face3d-server/mock")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", "animation")
        Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", "emotion")
        Invoke-Checked @("cargo", "clippy", "--workspace", "--no-default-features", "--features", "audio2face3d/cli,audio2face3d/mock,audio2face3d/client-grpc,audio2face3d-server/cli,audio2face3d-server/mock", "--all-targets", "--", "-D", "warnings")
        Invoke-Checked @("cargo", "test", "--workspace", "--no-default-features", "--features", "audio2face3d/cli,audio2face3d/mock,audio2face3d/client-grpc,audio2face3d-server/cli,audio2face3d-server/mock")
        Invoke-Checked @("cargo", "test", "--locked", "-p", "audio2face3d-gui-core", "--features", "gltf-read,gltf-write")
        Invoke-Checked @("cargo", "test", "--locked", "-p", "audio2face3d-headgen")
        Invoke-Checked @("cargo", "test", "--locked", "-p", "audio2face3d-gui", "--no-default-features", "--features", "ui-egui,grpc,mock")
        Invoke-Checked @("cargo", "test", "--locked", "-p", "audio2face3d-gui", "--no-default-features", "--features", "standalone-app,grpc,mock", "--test", "startup")
        Invoke-Checked @("cargo", "test", "--locked", "-p", "audio2face3d-gui", "--no-default-features", "--features", "standalone-app,grpc,mock", "--lib")
        Invoke-Checked @("cargo", "clippy", "--locked", "-p", "audio2face3d-gui", "--no-default-features", "--features", "capture,grpc,mock", "--all-targets", "--", "-D", "warnings")
        foreach ($features in @("", "animation", "emotion")) {
            $arguments = @("cargo", "test", "-p", "audio2face3d", "--no-default-features")
            if ($features) { $arguments += @("--features", $features) }
            Invoke-Checked $arguments
        }
        $previousRustdocFlags = $env:RUSTDOCFLAGS
        try {
            $env:RUSTDOCFLAGS = "-D warnings"
            Invoke-Checked @("cargo", "doc", "--workspace", "--no-default-features", "--features", "audio2face3d/cli,audio2face3d/mock,audio2face3d/client-grpc,audio2face3d-server/cli,audio2face3d-server/mock", "--no-deps")
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
        Invoke-Checked @("cargo", "clippy", "-p", "audio2face3d", "--features", "animation,cuda", "--all-targets", "--", "-D", "warnings")
        Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--features", "animation,cuda")
        Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--features", "animation,cuda", "--test", "compile_fail")
        foreach ($features in @("cuda", "emotion,cuda")) {
            Invoke-Checked @("cargo", "test", "-p", "audio2face3d", "--no-default-features", "--features", $features)
        }
    }
    "tensorrt-model" {
        Require-Environment @("AUDIO2FACE3D_TEST_FACADE_MODELS")
        foreach ($features in @("tensorrt", "animation,tensorrt", "emotion,tensorrt")) {
            Invoke-Checked @("cargo", "check", "-p", "audio2face3d", "--no-default-features", "--features", $features, "--all-targets")
            Invoke-Checked @("cargo", "test", "--release", "-p", "audio2face3d", "--no-default-features", "--features", $features, "--test", "compile_fail", "--test", "api_contracts", "--test", "tokio_runtime")
        }
        Invoke-Checked @("cargo", "check", "--workspace", "--all-features")
        Invoke-Checked @("cargo", "clippy", "--workspace", "--all-features", "--all-targets", "--", "-D", "warnings")
        # Keep optimized real-model tests for representative GPU execution.
        Invoke-Checked @("cargo", "test", "--release", "--workspace", "--all-features")
        Invoke-Checked @("cargo", "test", "--release", "-p", "audio2face3d", "--all-features", "--test", "native_facade", "--", "--ignored", "--exact", "acquired_models_execute_through_completed_facades", "--test-threads=1")
        $previousRustdocFlags = $env:RUSTDOCFLAGS
        try {
            $env:RUSTDOCFLAGS = "-D warnings"
            Invoke-Checked @("cargo", "doc", "--workspace", "--all-features", "--no-deps")
        } finally {
            $env:RUSTDOCFLAGS = $previousRustdocFlags
        }
    }
    "reference-parity" {
        Require-Environment @("AUDIO2FACE_SDK_ROOT", "AUDIO2FACE3D_REFERENCE_WAV_LICENSE")
        Invoke-Checked @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "reference/run-sdk-compatibility.ps1", "-Pipeline", "emotion", "-Execution", "standard", "-Precision", "fp32", "-Tracks", "1", "-Seed", "0")
        Invoke-Checked @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "reference/run-sdk-compatibility.ps1", "-Pipeline", "regression", "-Execution", "teeth-standalone", "-Precision", "fp32", "-Tracks", "2", "-Seed", "0")
    }
    "release" {
        $previousRustdocFlags = $env:RUSTDOCFLAGS
        try {
            $env:RUSTDOCFLAGS = "-D warnings"
            Invoke-Checked @("cargo", "doc", "--workspace", "--all-features", "--no-deps", "--locked")
        } finally {
            $env:RUSTDOCFLAGS = $previousRustdocFlags
        }
        Invoke-Checked @("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "test-release-packages.ps1"))
    }
}

Write-Output "Release tier passed: $Tier"
