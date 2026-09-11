param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineDirectory,
    [string]$ReportDirectory = "reference/compatible_test/step9-report"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$captureRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "compatible_test"))
$baseline = [IO.Path]::GetFullPath($BaselineDirectory)
$reportRoot = [IO.Path]::GetFullPath($ReportDirectory)
foreach ($path in @($baseline, $reportRoot)) {
    if (!$path.StartsWith($captureRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Baseline and reports must stay under ignored reference/compatible_test"
    }
}
if (!(Test-Path -LiteralPath (Join-Path $baseline "results"))) { throw "Preserved baseline results are required" }
if (Test-Path -LiteralPath $reportRoot) { throw "Report directory already exists; choose a new directory" }
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null
$baselineProvenance = $null
$provenancePath = Join-Path $baseline "capture-summary.json"
if (Test-Path -LiteralPath $provenancePath) {
    $baselineProvenance = Get-Content -Raw -Encoding UTF8 $provenancePath | ConvertFrom-Json
    Copy-Item -LiteralPath $provenancePath -Destination (Join-Path $reportRoot "baseline-provenance.json")
}

function Same-Inputs($Expected, $Actual) {
    if (($Expected.case | ConvertTo-Json -Compress) -cne ($Actual.case | ConvertTo-Json -Compress)) { return $false }
    if ($Expected.fixture.sha256 -ne $Actual.fixture.sha256) { return $false }
    $expectedModels = @($Expected.model_files.psobject.Properties | ForEach-Object { "$($_.Name)=$($_.Value.sha256)" } | Sort-Object)
    $actualModels = @($Actual.model_files.psobject.Properties | ForEach-Object { "$($_.Name)=$($_.Value.sha256)" } | Sort-Object)
    return ($expectedModels -join "`n") -ceq ($actualModels -join "`n")
}

$cases = @()
foreach ($pipeline in @("emotion", "regression", "diffusion")) {
    foreach ($tracks in @(1, 2)) {
        $cases += [pscustomobject]@{ pipeline = $pipeline; execution = "standard"; tracks = $tracks }
    }
    if ($pipeline -ne "emotion") {
        foreach ($execution in @("interactive-random", "interactive-all", "blendshape-cpu", "blendshape-gpu", "interactive-blendshape-random", "interactive-blendshape-all")) {
            $cases += [pscustomobject]@{ pipeline = $pipeline; execution = $execution; tracks = 1 }
        }
    }
}
$cases += [pscustomobject]@{ pipeline = "regression"; execution = "teeth-standalone"; tracks = 2 }
$results = @()
foreach ($case in $cases) {
    $name = "$($case.pipeline)-$($case.execution)-fp32-tracks$($case.tracks)-seed0"
    $output = Join-Path $captureRoot "results/$name"
    $old = Join-Path $baseline "results/$name"
    $comparison = Join-Path $output "comparison.json"
    # A failed capture must not be confused with a stale comparison file.
    $started = [DateTime]::UtcNow
    Write-Host "Step 9 capture: $name"
    $previousErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell wraps ordinary Cargo stderr progress in error
        # records. Capture it without treating a successful build as a failure;
        # the child process exit code and fresh comparison decide the outcome.
        $ErrorActionPreference = "Continue"
        & powershell -NoProfile -ExecutionPolicy Bypass -File reference/run-sdk-compatibility.ps1 -Pipeline $case.pipeline -Execution $case.execution -Precision fp32 -Tracks $case.tracks -Seed 0 *> (Join-Path $reportRoot "$name.log")
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $entry = [ordered]@{ case = $name; exit_code = $exitCode; classification = "not-run"; maximum_absolute_error = $null; baseline_identical = $null; baseline_inputs_match = $null; baseline_records_identical = $null; baseline_values_identical = $null; sdk_metadata_identical = $null; first_difference = $null; structural_differences = @(); records_compared = 0; values_compared = 0 }
    if ((Test-Path -LiteralPath $comparison) -and (Get-Item -LiteralPath $comparison).LastWriteTimeUtc -ge $started) {
        $value = Get-Content -Raw -Encoding UTF8 $comparison | ConvertFrom-Json
        $entry.maximum_absolute_error = $value.maximum_absolute_error
        $entry.first_difference = $value.first_difference
        $entry.structural_differences = @($value.structural_differences | Where-Object { $null -ne $_ })
        $entry.records_compared = $value.records_compared
        $entry.values_compared = $value.values_compared
        $actual = Get-Content -Raw -Encoding UTF8 (Join-Path $output "rust/artifact.json") | ConvertFrom-Json
        $sdk = Get-Content -Raw -Encoding UTF8 (Join-Path $output "cpp/artifact.json") | ConvertFrom-Json
        # Match same_record_identity in the existing comparator, but inspect
        # all records even when numerical comparison stops on an earlier one.
        $identityFields = @("layer", "component", "track", "frame", "inference", "timestamp", "next_timestamp", "dtype", "shape")
        $sdkIdentity = ConvertTo-Json -InputObject @($sdk.records | Select-Object -Property $identityFields) -Depth 12 -Compress
        $rustIdentity = ConvertTo-Json -InputObject @($actual.records | Select-Object -Property $identityFields) -Depth 12 -Compress
        $entry.sdk_metadata_identical = $sdkIdentity -ceq $rustIdentity
        # Keep each report tied to its own captures. Later harness runs replace
        # results/<case>, and must not silently change baseline assessments.
        $saved = Join-Path $reportRoot "results/$name"
        [IO.Directory]::CreateDirectory($saved) | Out-Null
        foreach ($producer in @("cpp", "rust")) {
            Copy-Item -LiteralPath (Join-Path $output $producer) -Destination (Join-Path $saved $producer) -Recurse
        }
        Copy-Item -LiteralPath $comparison -Destination (Join-Path $saved "comparison.json")
        $entry.classification = if ($value.compatible) { "within-sdk-tolerance" } else { "unexplained-difference" }
        if (Test-Path -LiteralPath (Join-Path $old "rust/artifact.json")) {
            $expected = Get-Content -Raw -Encoding UTF8 (Join-Path $old "rust/artifact.json") | ConvertFrom-Json
            $entry.baseline_inputs_match = Same-Inputs $expected $actual
            $entry.baseline_records_identical = ($expected.records | ConvertTo-Json -Depth 20 -Compress) -ceq ($actual.records | ConvertTo-Json -Depth 20 -Compress)
            $entry.baseline_values_identical = (Get-FileHash (Join-Path $old "rust/values.f32le")).Hash -eq (Get-FileHash (Join-Path $output "rust/values.f32le")).Hash
            $entry.baseline_identical = $entry.baseline_inputs_match -and $entry.baseline_records_identical -and $entry.baseline_values_identical
            if (!$value.compatible -and $entry.baseline_identical) { $entry.classification = "unchanged-local-baseline" }
        }
    }
    $results += [pscustomobject]$entry
    $report = [ordered]@{
        schema_version = 1
        baseline_scope = "Preserved captures; source-revision claims require the accompanying baseline provenance"
        baseline_provenance = $baselineProvenance
        precision = "fp32"
        seed = 0
        tolerance_sha256 = (Get-FileHash reference/tolerances.json).Hash
        audio_sha256 = (Get-FileHash (Join-Path $captureRoot "input.wav")).Hash
        maximum_error_scope = "Existing comparator reports records through the first mismatching record, not necessarily all records"
        unimplemented_reference_cases = @("emotion-interactive-random", "emotion-interactive-all")
        cases = $results
    }
    [IO.File]::WriteAllText((Join-Path $reportRoot "summary.json"), ($report | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
    Write-Host "$name : $($entry.classification)"
}
if (@($results | Where-Object { $_.classification -in @("not-run", "unexplained-difference") }).Count -gt 0) {
    throw "Some cases require investigation; inspect the report. SDK tolerances were not changed."
}
Write-Host "All captures classified; known baseline differences are not SDK parity passes."
