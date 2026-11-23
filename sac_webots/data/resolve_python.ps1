param([string]$Configured='python')
$ErrorActionPreference='Stop'
if ($Configured -and $Configured -ne 'python') {
    $candidate=[Environment]::ExpandEnvironmentVariables($Configured)
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw 'Configured Python does not exist.' }
    return $candidate
}
$candidates=@()
if ($env:CONDA_PREFIX) { $candidates+=Join-Path $env:CONDA_PREFIX 'python.exe' }
if ($env:USERPROFILE) {
    $candidates+=Join-Path $env:USERPROFILE 'anaconda3\envs\webot\python.exe'
    $candidates+=Join-Path $env:USERPROFILE 'miniconda3\envs\webot\python.exe'
}
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
}
return (Get-Command python -ErrorAction Stop).Source
