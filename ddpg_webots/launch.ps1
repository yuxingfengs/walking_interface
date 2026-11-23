param([Parameter(Mandatory=$true)][string]$Config,
      [ValidateSet('train','pretrain','check','report')][string]$Mode='check', [string]$ResumeCheckpoint)
$ErrorActionPreference='Stop'
$settings=Get-Content -Raw -Encoding UTF8 -LiteralPath $Config | ConvertFrom-Json
$python=& (Join-Path $PSScriptRoot 'data/runtime/resolve_python.ps1') -Configured $settings.training_config.webots.python
switch ($Mode) {
    'train' { $entry='train.py'; $extra=@() }
    'pretrain' { $entry='pretrain.py'; $extra=@() }
    'check' { $entry='train.py'; $extra=@('--check') }
    'report' { $entry='report.py'; $extra=@('--no-compare','--open') }
}
if ($ResumeCheckpoint) { $extra += @('--resume-checkpoint', $ResumeCheckpoint) }
& $python -u -X utf8 (Join-Path $PSScriptRoot $entry) --config $Config @extra
exit $LASTEXITCODE
