param([switch]$Check, [string]$Resume, [string]$Experiment)
$ErrorActionPreference = 'Stop'
$projectRoot = Join-Path $PSScriptRoot 'data'
$projectConfig = Get-Content -Raw -Encoding UTF8 (Join-Path $projectRoot 'config.json') | ConvertFrom-Json
$experimentId = [string]$projectConfig.current_experiment
if ($projectConfig.backend_overrides.webots.current_experiment) {
    $experimentId = [string]$projectConfig.backend_overrides.webots.current_experiment
}
if ($Experiment) { $experimentId = $Experiment }
if ($experimentId -notmatch '^\d{3}$') {
    throw 'current_experiment must be a three-digit ID.'
}
$experimentConfigPath = Join-Path $projectRoot (Join-Path $projectConfig.experiments_root (Join-Path $experimentId 'training_config.json'))
$experimentConfig = Get-Content -Raw -Encoding UTF8 $experimentConfigPath | ConvertFrom-Json
$python = & (Join-Path $projectRoot 'resolve_python.ps1') -Configured ([string]$projectConfig.python)
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable in the SAC config was not found: $python"
}
$entry = Join-Path $PSScriptRoot 'train_sac.py'
$entryArgs=@()
if ($Experiment) { $entryArgs += @('--experiment', $experimentId) }
if ($Check) { $entryArgs+='--check' }
if ($Resume) { $entryArgs+=@('--resume', $Resume) }
& $python -u -X utf8 $entry @entryArgs
exit $LASTEXITCODE
