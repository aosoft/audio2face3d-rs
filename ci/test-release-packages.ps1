param(
    [switch]$SkipPublishDryRun
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
        throw "package command failed ($LASTEXITCODE): $($Command -join ' ')"
    }
}

function Get-PackageFiles {
    param([Parameter(Mandatory = $true)][string]$Package)
    $files = @(& cargo package -p $Package --locked --allow-dirty --list)
    if ($LASTEXITCODE -ne 0) {
        throw "could not list package contents for $Package"
    }
    return $files
}

$metadata = cargo metadata --locked --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "could not read Cargo metadata"
}
$packages = @($metadata.packages | Where-Object { $_.name -in @("audio2face3d", "audio2face3d-server", "audio2face3d-gui") })
if ($packages.Count -ne 3) {
    throw "expected the inference library, server and GUI packages"
}
foreach ($package in $packages) {
    if ("crates-io" -notin $package.publish) {
        throw "package $($package.name) must allow publishing to crates-io"
    }
    if ($package.version -ne "0.2.0") {
        throw "package $($package.name) has version $($package.version), expected 0.2.0"
    }
}
$cli = $packages | Where-Object name -eq "audio2face3d-server"
$library = $packages | Where-Object name -eq "audio2face3d"
if ($library.license -ne 'MIT AND MPL-2.0 AND Apache-2.0') {
    throw 'The library package must declare the Eigen-derived source license'
}
$mplText = [IO.File]::ReadAllText((Join-Path $repoRoot 'LICENSE-MPL-2.0')).Replace("`r`n", "`n").Trim()
$packagedLicense = [IO.File]::ReadAllText((Join-Path $repoRoot 'crates/audio2face3d/LICENSE-MPL-2.0')).Replace("`r`n", "`n")
if (-not $packagedLicense.Contains($mplText)) {
    throw 'Packaged LICENSE-MPL-2.0 must retain the full root MPL-2.0 text'
}
foreach ($consumer in @($cli, ($packages | Where-Object name -eq "audio2face3d-gui"))) {
    $libraryDependency = $consumer.dependencies | Where-Object { $_.name -eq "audio2face3d" -and $null -eq $_.kind }
    if ($null -eq $libraryDependency -or $libraryDependency.req -notin @("^0.2.0", "0.2.0")) {
        throw "$($consumer.name) must depend on audio2face3d 0.2.0"
    }
}

$forbiddenPath = '(?i)(^|/)(reference/compatible_test|models)(/|$)|\.audio2x-|\.(onnx(?:[._]data)?|trt|engine|plan|wav|pdb|dll|so|dylib|lib|exe|bin|npz|npy)$'
foreach ($packageName in @("audio2face3d", "audio2face3d-server", "audio2face3d-gui")) {
    $files = @(Get-PackageFiles $packageName)
    if ($packageName -eq 'audio2face3d-gui') {
        foreach ($required in @('README.md', 'assets/default-head.glb', 'presets/ict-facekit.toml', 'src/obj2morph.rs')) {
            if ($required -notin $files) { throw "GUI package is missing $required" }
        }
    }
    if ($packageName -eq 'audio2face3d') {
        foreach ($required in @('LICENSE-MPL-2.0', 'LICENSE-APACHE')) {
            if ($required -notin $files) { throw "Library package is missing $required" }
        }
    }
    if ('LICENSE' -notin $files) { throw "$packageName is missing LICENSE" }
    if ($packageName -eq 'audio2face3d' -and 'src/animation/blendshape/bvls/svd.rs' -notin $files) {
        throw 'The library package must retain its MPL-covered SVD source'
    }
    $invalid = @($files | ForEach-Object { $_ -replace '\\', '/' } | Where-Object { $_ -match $forbiddenPath })
    if ($invalid.Count -ne 0) {
        throw "$packageName contains forbidden local/native artifacts: $($invalid -join ', ')"
    }
    Write-Output "$packageName package contents accepted ($($files.Count) files)"
}

# Modern Cargo stages workspace packages in a temporary local registry, so
# all packaged manifests can be verified before their initial publication.
Invoke-Checked @("cargo", "package", "--workspace", "--locked", "--allow-dirty", "--no-default-features")


# Verify every packaged crate retains the shared root license text.
$rootLicense = [IO.File]::ReadAllText((Join-Path $repoRoot 'LICENSE')).Replace("`r`n", "`n")
foreach ($package in $packages) {
    $archiveRoot = Join-Path $metadata.target_directory "package/$($package.name)-$($package.version)"
    $archiveLicense = [IO.File]::ReadAllText((Join-Path $archiveRoot 'LICENSE')).Replace("`r`n", "`n")
    if ($archiveLicense -ne $rootLicense) {
        throw "$($package.name) packaged LICENSE differs from the shared root LICENSE"
    }
}

if (-not $SkipPublishDryRun) {
    Invoke-Checked @("cargo", "publish", "--workspace", "--locked", "--allow-dirty", "--no-default-features", "--dry-run", "--registry", "crates-io")
} else {
    Write-Warning "Skipped the networked workspace publish dry-run; this gate is unverified."
}

Write-Output "Library, server and GUI package verification passed. No packages were uploaded."
