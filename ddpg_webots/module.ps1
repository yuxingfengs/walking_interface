param([ValidateSet('start','continue','check')][string]$Mode='check')
$ErrorActionPreference='Stop'
$config = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'config.json') | ConvertFrom-Json
$experiment = [string]$config.current_experiment
if ($experiment -notmatch '^[A-Za-z0-9_]+$') { throw 'Invalid current_experiment' }
if ($Mode -eq 'continue') {
    if (-not $config.resume_from) { throw 'Set resume_from in this module config.json to the run or checkpoint to continue.' }
    $resume = [string]$config.resume_from
    if (-not [IO.Path]::IsPathRooted($resume)) { $resume = Join-Path $PSScriptRoot $resume }
    $resume = (Resolve-Path -LiteralPath $resume).Path
}
$profile = Join-Path $PSScriptRoot "experiments/$experiment/config.json"
$action = if ($Mode -eq 'check') { 'check' } else { 'train' }
$params = @{Config=$profile; Mode=$action}
if ($Mode -eq 'continue') { $params.ResumeCheckpoint=$resume }
& (Join-Path $PSScriptRoot 'launch.ps1') @params
exit $LASTEXITCODE
