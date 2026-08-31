param()

$ErrorActionPreference = "Stop"
$SdkRoot = $env:AUDIO2FACE_SDK_ROOT
if (-not $SdkRoot) {
    throw "AUDIO2FACE_SDK_ROOT must point to the local NVIDIA Audio2Face-3D-SDK checkout"
}
if (-not (Test-Path -LiteralPath $SdkRoot -PathType Container)) {
    throw "AUDIO2FACE_SDK_ROOT does not name a directory: $SdkRoot"
}

$source = Join-Path $PSScriptRoot "cpp/main.cpp"
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "compatible_test/tools"))
$sdkInclude = Join-Path $SdkRoot "_build/release/audio2x-sdk/include"
$sdkLibrary = Join-Path $SdkRoot "_build/release/audio2x-sdk/lib/audio2x.lib"
if (-not (Test-Path -LiteralPath $sdkInclude -PathType Container)) {
    throw "Built audio2x SDK headers were not found under AUDIO2FACE_SDK_ROOT"
}
if (-not (Test-Path -LiteralPath $sdkLibrary -PathType Leaf)) {
    throw "Built audio2x.lib was not found under AUDIO2FACE_SDK_ROOT"
}

$toolchainRoots = Get-ChildItem "C:/Program Files/Microsoft Visual Studio" -Directory |
    ForEach-Object { Join-Path $_.FullName "Community/VC/Tools/MSVC" } |
    Where-Object { Test-Path $_ }
$toolchains = foreach ($toolchainRoot in $toolchainRoots) {
    Get-ChildItem $toolchainRoot -Directory
}
$msvc = $toolchains | Sort-Object Name -Descending | Select-Object -First 1
if (-not $msvc) {
    throw "Visual Studio Community with the MSVC toolchain was not found"
}
$compiler = Join-Path $msvc.FullName "bin/HostX64/x64/cl.exe"
$linker = Join-Path $msvc.FullName "bin/HostX64/x64/link.exe"

$kitRoot = "C:/Program Files (x86)/Windows Kits/10"
$kit = Get-ChildItem (Join-Path $kitRoot "Include") -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $kit) {
    throw "Windows SDK was not found"
}
$kitVersion = $kit.Name
$env:INCLUDE = @(
    (Join-Path $msvc.FullName "include")
    (Join-Path $kitRoot "Include/$kitVersion/ucrt")
    (Join-Path $kitRoot "Include/$kitVersion/shared")
    (Join-Path $kitRoot "Include/$kitVersion/um")
    (Join-Path $kitRoot "Include/$kitVersion/winrt")
    $sdkInclude
) -join ";"
$env:LIB = @(
    (Join-Path $msvc.FullName "lib/x64")
    (Join-Path $kitRoot "Lib/$kitVersion/ucrt/x64")
    (Join-Path $kitRoot "Lib/$kitVersion/um/x64")
) -join ";"

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$object = Join-Path $outputRoot "main.obj"
$executable = Join-Path $outputRoot "audio2face3d-cpp-reference.exe"

& $compiler /nologo /std:c++20 /EHsc /MD /W4 /GL- /c $source "/Fo$object" | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "C++ reference runner compilation failed with exit code $LASTEXITCODE"
}

# This small standalone link deliberately avoids the SDK aggregate target and
# its incremental/LTCG state, which is the reproducible workaround for LNK1000.
& $linker /NOLOGO "/OUT:$executable" /INCREMENTAL:NO /LTCG:OFF /DEBUG:NONE `
    /SUBSYSTEM:CONSOLE $object $sdkLibrary bcrypt.lib | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "C++ reference runner link failed with exit code $LASTEXITCODE"
}

Write-Output $executable
