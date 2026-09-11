param(
    [Parameter(Mandatory = $true)][string]$BaselineDirectory,
    [Parameter(Mandatory = $true)][string]$ReportDirectory
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$captureRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "compatible_test"))
$baseline = [IO.Path]::GetFullPath($BaselineDirectory)
$reportRoot = [IO.Path]::GetFullPath($ReportDirectory)
foreach ($path in @($baseline, $reportRoot)) {
    if (!$path.StartsWith($captureRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Baseline and report must be under ignored reference/compatible_test"
    }
}
$summary = Get-Content -Raw -Encoding UTF8 (Join-Path $reportRoot "summary.json") | ConvertFrom-Json
foreach ($case in $summary.cases) {
    $case.structural_differences = @($case.structural_differences | Where-Object { $null -ne $_ })
    if ($case.classification -ne "unexplained-difference" -or !$case.baseline_inputs_match) { continue }
    $comparison = Join-Path $reportRoot "$($case.case).baseline.json"
    $actual = Join-Path $reportRoot "results/$($case.case)/rust"
    if (!(Test-Path -LiteralPath (Join-Path $actual "artifact.json"))) {
        throw "Report has no preserved capture for $($case.case); recapture with run-api-regression.ps1"
    }
    if (Test-Path -LiteralPath $comparison) { throw "Baseline comparison already exists; preserve it and use a new report directory" }
    & cargo run -p audio2face3d-cli -- reference compare (Join-Path $baseline "results/$($case.case)/rust") $actual --tolerances reference/tolerances.json --report $comparison
    $exitCode = $LASTEXITCODE
    if (!(Test-Path -LiteralPath $comparison)) { throw "Baseline comparison did not produce a report" }
    $result = Get-Content -Raw -Encoding UTF8 $comparison | ConvertFrom-Json
    $case | Add-Member -NotePropertyName baseline_maximum_absolute_error -NotePropertyValue $result.maximum_absolute_error
    $case | Add-Member -NotePropertyName baseline_within_tolerance -NotePropertyValue ($exitCode -eq 0 -and $result.compatible)
    if ($case.baseline_within_tolerance) {
        # Preserve the failed SDK exit code, SDK error and exact-hash result.
        # Numerical stability against a local baseline is NOT SDK parity.
        $case.classification = "baseline-stable-within-existing-tolerance"
    }
}
[IO.File]::WriteAllText((Join-Path $reportRoot "assessed-summary.json"), ($summary | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
Write-Host "Baseline assessment written; SDK parity failures remain failures."
