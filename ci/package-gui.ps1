param(
    [ValidateSet('grpc', 'local-grpc')][string]$Mode = 'grpc',
    [string]$OutputDirectory,
    [string]$PlatformConfig
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
$previousConfig = $env:AUDIO2FACE3D_PLATFORM_CONFIG
try {
    if (-not $OutputDirectory) { $OutputDirectory = "temp/gui-dist-$Mode" }
    $destination = [IO.Path]::GetFullPath([IO.Path]::Combine($repoRoot, $OutputDirectory))
    if (Test-Path -LiteralPath $destination) { throw "Choose a new output directory; already exists: $destination" }
    if ($PlatformConfig) { $env:AUDIO2FACE3D_PLATFORM_CONFIG = [IO.Path]::GetFullPath([IO.Path]::Combine($repoRoot, $PlatformConfig)) }
    $features = 'desktop,grpc'
    if ($Mode -eq 'local-grpc') { $features += ',local' }
    $target = 'x86_64-pc-windows-msvc'
    & cargo build --release --locked -p audio2face3d-gui --no-default-features --features $features --target $target
    if ($LASTEXITCODE -ne 0) { throw 'GUI build failed' }
    $qualifiedFeatures = ($features.Split(',') | ForEach-Object { "audio2face3d-gui/$_" }) -join ','
    $metadataText = & cargo metadata --locked --format-version 1 --no-default-features --features $qualifiedFeatures --filter-platform $target
    if ($LASTEXITCODE -ne 0) { throw 'Dependency metadata failed' }
    $metadata = $metadataText | ConvertFrom-Json
    $packages = @{}; $nodes = @{}
    foreach ($package in $metadata.packages) { $packages[$package.id] = $package }
    foreach ($node in $metadata.resolve.nodes) { $nodes[$node.id] = $node }
    $root = $metadata.packages | Where-Object name -eq 'audio2face3d-gui'
    $pending = [Collections.Generic.Stack[string]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new()
    $pending.Push($root.id)
    while ($pending.Count) {
        $id = $pending.Pop()
        if (-not $seen.Add($id)) { continue }
        foreach ($dependency in $nodes[$id].deps) {
            if (@($dependency.dep_kinds | Where-Object kind -ne 'dev').Count) { $pending.Push($dependency.pkg) }
        }
    }
    New-Item -ItemType Directory -Path (Join-Path $destination 'assets'),(Join-Path $destination 'licenses'),(Join-Path $destination 'source') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $metadata.target_directory "$target/release/audio2face3d-gui.exe") -Destination $destination
    Copy-Item -LiteralPath 'crates/audio2face3d-gui/assets/default-head.glb','crates/audio2face3d-gui/assets/README.md' -Destination (Join-Path $destination 'assets')
    Copy-Item -LiteralPath 'LICENSE','LICENSE-MPL-2.0','THIRD-PARTY-NOTICES.md' -Destination $destination
    Copy-Item -LiteralPath 'crates/audio2face3d/LICENSE-APACHE' -Destination $destination
    Copy-Item -LiteralPath 'gui.example.toml','platform.example.toml' -Destination $destination
    Copy-Item -LiteralPath 'docs/gui.md' -Destination (Join-Path $destination 'README.md')
    # Include exact MPL-covered source alongside native binaries, not only a mutable URL.
    Copy-Item -LiteralPath 'crates/audio2face3d/src/animation/blendshape/bvls/svd.rs' -Destination (Join-Path $destination 'source/svd.rs')
    $rows = @('# Dependency notices', '', 'Conservative dependency inventory, including build tools. See each directory for license texts and bundled font/data notices.', '', '| Package | Version | License expression |', '| --- | --- | --- |')
    foreach ($package in @($seen | ForEach-Object { $packages[$_] } | Sort-Object name,version)) {
        $rows += "| $($package.name) | $($package.version) | $($package.license) |"
        if (-not $package.source) { continue }
        $packageRoot = Split-Path -Parent $package.manifest_path
        $noticeRoot = Join-Path $destination "licenses/$($package.name)-$($package.version)"
        $noticeFiles = @(Get-ChildItem -LiteralPath $packageRoot -File -Recurse | Where-Object { $_.Name -match '(^|[.\-_])(LICENSE|LICENCE|COPYING|NOTICE|OFL|UFL|UNLICENSE|AUTHORS)([.\-_]|$)' })
        if ($package.name -eq 'epaint_default_fonts') {
            $noticeFiles += Get-ChildItem -LiteralPath (Join-Path $packageRoot 'fonts') -File -Filter '*.txt'
        }
        if ($package.license_file) {
            $explicitLicense = [IO.Path]::GetFullPath([IO.Path]::Combine($packageRoot,$package.license_file))
            if (Test-Path -LiteralPath $explicitLicense) { $noticeFiles += Get-Item -LiteralPath $explicitLicense }
        }
        New-Item -ItemType Directory -Path $noticeRoot -Force | Out-Null
        if ($package.license -match 'Apache-2.0') {
            Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'gui-licenses/Apache-2.0.txt') -Destination $noticeRoot
        }
        # Preserve upstream attribution even when a crate archive omits license files.
        Copy-Item -LiteralPath $package.manifest_path -Destination $noticeRoot
        $readme = Join-Path $packageRoot 'README.md'
        if (Test-Path -LiteralPath $readme) { Copy-Item -LiteralPath $readme -Destination $noticeRoot }
        if (-not $noticeFiles.Count) {
            $fallback = switch -Regex ($package.license) {
                '^(MIT OR )?Apache-2.0$' { 'Apache-2.0.txt'; break }
                '^BSL-1.0$' { 'BSL-1.0.txt'; break }
                '^CC0-1.0$' { 'CC0-1.0.txt'; break }
                '^MIT$' {
                    if ($package.name -match '^tonic-prost') { 'tonic-MIT.txt' }
                    elseif ($package.name -match '^protoc-bin-vendored') { 'protoc-MIT.txt' }
                    break
                }
            }
            if (-not $fallback) { throw "No license text found for $($package.name); review before distributing $destination" }
            Copy-Item -LiteralPath (Join-Path $PSScriptRoot "gui-licenses/$fallback") -Destination $noticeRoot
        }
        foreach ($file in $noticeFiles | Sort-Object FullName -Unique) {
            $relative = $file.FullName.Substring($packageRoot.Length).TrimStart('\','/')
            $noticePath = Join-Path $noticeRoot $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $noticePath) -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $noticePath
        }
    }
    Set-Content -LiteralPath (Join-Path $destination 'licenses/INDEX.md') -Value $rows -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $destination 'BUILD.txt') -Value @("audio2face3d-gui $($root.version)","Features: $features","Target: $target",'Models, CUDA, TensorRT and driver binaries are not included.') -Encoding UTF8
    Write-Output "GUI package: $destination"
} finally {
    $env:AUDIO2FACE3D_PLATFORM_CONFIG = $previousConfig
    Pop-Location
}
