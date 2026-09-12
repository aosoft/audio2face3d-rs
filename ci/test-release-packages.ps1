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
$packages = @($metadata.packages | Where-Object { $_.name -in @("audio2face3d", "audio2face3d-cli") })
if ($packages.Count -ne 2) {
    throw "expected exactly the audio2face3d library and CLI packages"
}
foreach ($package in $packages) {
    if ($package.version -ne "0.1.0") {
        throw "package $($package.name) has version $($package.version), expected 0.1.0"
    }
}
$cli = $packages | Where-Object name -eq "audio2face3d-cli"
$library = $packages | Where-Object name -eq "audio2face3d"
if ($library.license -ne 'MIT AND MPL-2.0') {
    throw 'The library package must declare the Eigen-derived source license'
}
$mplText = [IO.File]::ReadAllText((Join-Path $repoRoot 'LICENSE-MPL-2.0')).Replace("`r`n", "`n").Trim()
$packagedLicense = [IO.File]::ReadAllText((Join-Path $repoRoot 'crates/audio2face3d/LICENSE')).Replace("`r`n", "`n")
if (-not $packagedLicense.Contains($mplText)) {
    throw 'Packaged LICENSE must retain the full root MPL-2.0 text'
}
$libraryDependency = $cli.dependencies | Where-Object name -eq "audio2face3d"
if ($null -eq $libraryDependency -or $libraryDependency.req -notin @("^0.1.0", "0.1.0")) {
    throw "audio2face3d-cli must depend on audio2face3d 0.1.0"
}

$forbiddenPath = '(?i)(^|/)(reference/compatible_test|models)(/|$)|\.audio2x-|\.(onnx(?:[._]data)?|trt|engine|plan|wav|pdb|dll|so|dylib|lib|exe|bin|npz|npy)$'
foreach ($packageName in @("audio2face3d", "audio2face3d-cli")) {
    $files = @(Get-PackageFiles $packageName)
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
# both packaged manifests can be verified before their initial publication.
Invoke-Checked @("cargo", "package", "--workspace", "--locked", "--allow-dirty")

if (-not $SkipPublishDryRun) {
    Invoke-Checked @("cargo", "publish", "--workspace", "--locked", "--allow-dirty", "--dry-run", "--registry", "crates-io")
} else {
    Write-Warning "Skipped the networked workspace publish dry-run; this gate is unverified."
}

Write-Output "Library and CLI package verification passed. No packages were uploaded."
