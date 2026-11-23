param([ValidateSet('height','timestep','ddpg')] [string] $StandingComparison='ddpg',
      [string] $Experiment, [switch] $Headless)
$ErrorActionPreference='Stop'
$launcher = Join-Path $PSScriptRoot '../../sac_isaac_lab/launch.ps1'
$groupId = [guid]::NewGuid().ToString('N')
$directory = Join-Path $PSScriptRoot "checks/standing/$groupId"
Write-Host "Standing comparison: $directory"
$results = [ordered]@{}
$incomplete = $false
$cases = switch ($StandingComparison) {
    'timestep' { 'step_20ms','step_5ms' }
    'ddpg' { 'baseline','ddpg_settled' }
    'height' { 'baseline','foot_geometry' }
}
foreach ($case in $cases) {
    $caseArgs = @()
    if ($Experiment) { $caseArgs += @('-Experiment', $Experiment) }
    if ($Headless) { $caseArgs += '-Headless' }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode standing -StandingCase $case -StandingGroup $groupId @caseArgs
    $resultPath = Join-Path $directory "$case/standing.json"
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $resultPath)) {
        $results[$case] = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json
    } else {
        $results[$case] = @{status='incomplete'; exit_code=$LASTEXITCODE}
        $incomplete = $true
    }
}
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$results | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $directory 'comparison.json')
Write-Host "Comparison: $directory/comparison.json"
if ($incomplete) { exit 1 }
exit 0
