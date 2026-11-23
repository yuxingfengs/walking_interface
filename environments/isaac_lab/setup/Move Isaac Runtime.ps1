$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime_settings.ps1')
& $settings.isaac_python (Join-Path $PSScriptRoot 'external_runtime.py') @args
exit $LASTEXITCODE
