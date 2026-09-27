param(
    [Parameter(Mandatory = $true)][string]$SdkDirectory,
    [Parameter(Mandatory = $true)][string]$TensorRtRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not ($IsWindows -or $IsLinux)) { throw 'Native build CI supports Windows and Linux only' }
$sdk = [IO.Path]::GetFullPath($SdkDirectory)
$trt = [IO.Path]::GetFullPath($TensorRtRoot)
if (-not (Test-Path -LiteralPath (Join-Path $trt 'include/NvInfer.h'))) { throw 'TensorRT headers missing' }
$cuda = Join-Path $sdk 'cuda-12.9.1'
$downloads = Join-Path $sdk 'downloads'
New-Item -ItemType Directory -Force $cuda,$downloads | Out-Null
# NVIDIA CUDA 12.9.1 redistrib manifest; only compiler, runtime headers and CCCL.
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json
$platform = if ($IsWindows) { 'windows-x86_64' } else { 'linux-x86_64' }
$extension = if ($IsWindows) { 'zip' } else { 'tar.xz' }
$components = @(
    @('cuda_nvcc', '12.9.86', '227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85', '7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
    @('cuda_cudart', '12.9.79', '179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8', '1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
    @('cuda_cccl', '12.9.27', '17aaa7c6b8f94a417d8f3261780b7e34b9cbdfab7513bce86768623b06aa28b5', '8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa')
)
foreach ($component in $components) {
    $name,$version,$windowsHash,$linuxHash = $component
    $folder = "$name-$platform-$version-archive"
    $archive = Join-Path $downloads "$folder.$extension"
    $expected = if ($IsWindows) { $windowsHash } else { $linuxHash }
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/$name/$platform/$folder.$extension" -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw "Checksum mismatch: $archive" }
    & tar -xf $archive -C $downloads
    if ($LASTEXITCODE -ne 0) { throw "Could not extract $archive" }
    # Merge component layouts, including nvvm/libdevice, headers and compiler tools.
    Get-ChildItem -LiteralPath (Join-Path $downloads $folder) | Copy-Item -Destination $cuda -Recurse -Force
}
foreach ($header in @('include/cuda.h', 'include/cuda_runtime.h', 'include/cuda_runtime_api.h')) {
    if (-not (Test-Path -LiteralPath (Join-Path $cuda $header))) { throw "CUDA header missing: $header" }
}
$config = @("cuda-root = '$($cuda.Replace('\','/'))'", "tensorrt-root = '$($trt.Replace('\','/'))'")
if ($IsWindows) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $vs) { throw 'Visual Studio 2022 C++ tools are required' }
    $toolset = Get-ChildItem -LiteralPath (Join-Path $vs 'VC/Tools/MSVC') -Directory |
        Where-Object Name -Like '14.4*' | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $toolset) { throw 'CUDA 12.9 CI requires an MSVC 14.4x toolset' }
    $config += @('[build-cuda.windows]', "visual-studio-root = '$($vs.Replace('\','/'))'", "msvc-toolset-version = '$($toolset.Name)'")
} else {
    if (-not (Test-Path -LiteralPath '/usr/bin/g++-13')) { throw 'GCC 13 is required' }
    $config += @('[build-cuda.linux]', "cuda-host-compiler = '/usr/bin/g++-13'")
    $env:CC = '/usr/bin/gcc-13'
    $env:CXX = '/usr/bin/g++-13'
    if ($env:GITHUB_ENV) { @('CC=/usr/bin/gcc-13', 'CXX=/usr/bin/g++-13') | Add-Content -LiteralPath $env:GITHUB_ENV }
}
$configFile = Join-Path $sdk 'platform.toml'
$config | Set-Content -LiteralPath $configFile -Encoding utf8NoBOM
$env:AUDIO2FACE3D_PLATFORM_CONFIG = $configFile
if ($env:GITHUB_ENV) { "AUDIO2FACE3D_PLATFORM_CONFIG=$configFile" | Add-Content -LiteralPath $env:GITHUB_ENV }
Write-Output "Native build configuration: $configFile"
