param([ValidateSet('start','continue','check')][string]$Mode='check', [string]$CheckOutput)
$ErrorActionPreference='Stop'
$moduleRoot = Split-Path $PSScriptRoot -Parent
$config = Get-Content -Raw -LiteralPath (Join-Path $moduleRoot 'config.json') | ConvertFrom-Json
$experiment = [string]$config.current_experiment
if ($experiment -notmatch '^[A-Za-z0-9_]+$') { throw 'Invalid current_experiment' }
$action = if ($Mode -eq 'check') { 'offline' } else { 'train' }
$params = @{Experiment=$experiment; Mode=$action; Headless=[bool]$config.headless}
if ($Mode -eq 'continue') { $params.ContinueConfiguredRun = $true }
if ($CheckOutput) { $params.CheckOutput=$CheckOutput }
& (Join-Path $moduleRoot 'launch.ps1') @params
exit $LASTEXITCODE
